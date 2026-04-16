import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from transformers import Swinv2ForImageClassification
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

def train_model(model, dataloader, optimizer, criterion, device, start_epoch, num_epochs=80):
    model.train()
    loss_history = []

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        optimizer.zero_grad()

        for i, (images, heatmaps, star_features, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            star_features = star_features.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images, heatmaps, star_features)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            del images, heatmaps, star_features, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

class SwinWithStarPositions(nn.Module):
    def __init__(self, swin_model, num_classes):
        super(SwinWithStarPositions, self).__init__()
        self.swin = swin_model
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        swin_output_features = self.swin.config.hidden_size
        heatmap_output_features = 64 * 4 * 4
        star_features_size = 6
        combined_features = swin_output_features + heatmap_output_features + star_features_size
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, image, heatmap, star_features):
        outputs = self.swin(image)
        swin_output = outputs.hidden_states[-1].mean(dim=1)
        heatmap_output = self.heatmap_conv(heatmap)
        combined = torch.cat([swin_output, heatmap_output, star_features], dim=1)
        output = self.fc(combined)
        return output

class StarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.star_map_paths = []
        self.labels = []
        self.class_labels = [] # Placeholder for keypoint labels if needed
        num_classes=12

        # Iterate through class folders in images directory
        for class_name in os.listdir(os.path.join(root_dir, "images")):
            class_image_dir = os.path.join(root_dir, "processed", class_name)
            class_star_map_dir = os.path.join(root_dir, "processed", class_name)

            if os.path.isdir(class_image_dir) and os.path.isdir(class_star_map_dir):
                class_num=int(class_name.split('_')[1].strip("N"))
                label = (class_num // 30) % num_classes

                for img_file in os.listdir(class_image_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(img_file)[0]
                        star_map_file = f"{base_name}_star_map.json"

                        image_path = os.path.join(class_image_dir, img_file)
                        star_map_path = os.path.join(class_star_map_dir, star_map_file)

                        if os.path.exists(star_map_path):
                            self.image_paths.append(image_path)
                            self.star_map_paths.append(star_map_path)
                            self.labels.append(label)
                            self.class_labels.append(0) # Placeholder label for keypoints

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load star map data
        with open(self.star_map_paths[idx]) as f:
            star_data = json.load(f)

        # Extract keypoints (star positions)
        star_positions_list = [star_data['centroid']['position']]
        if 'reference_stars' in star_data:
            star_positions_list.extend([star['position'] for star in star_data['reference_stars']])

        keypoints = []
        for x, y in star_positions_list:
            keypoints.append((x, y))

        # Create class labels for each keypoint (if you don't have specific ones, use a placeholder like 0)
        keypoint_class_labels = [0] * len(keypoints)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), keypoints=keypoints, class_labels=keypoint_class_labels)

            image = transformed['image']
            transformed_keypoints = transformed['keypoints']

            # Re-create heatmap and star features from transformed keypoints and image size
            heatmap = self._create_heatmap_from_keypoints(image.shape[:2][::-1], transformed_keypoints) # Pass image size (width, height)
            star_features = self._create_star_features_from_keypoints(image.shape[:2][::-1], transformed_keypoints) # Pass image size (width, height)
        else:
            heatmap = self._create_heatmap(image.size, star_data)
            star_features = self._create_star_features(star_data)

        return image, heatmap, star_features, self.labels[idx]
        
        
    def _create_heatmap(self, image_size, star_data):
        star_positions = [star_data['centroid']['position']]
        if 'reference_stars' in star_data:
            star_positions.extend([star['position'] for star in star_data['reference_stars']])

        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        for (x, y) in star_positions:
            heatmap = cv2.circle(heatmap, (int(x), int(y)), 5, 1, -1)
        return torch.tensor(heatmap).unsqueeze(0)

    def _create_star_features(self, star_data):
        width, height = star_data['image_shape'][1], star_data['image_shape'][0]
        star_positions = [star_data['centroid']['position']]
        if 'reference_stars' in star_data:
            star_positions.extend([star['position'] for star in star_data['reference_stars']])

        normalized_positions = []
        for (x, y) in star_positions:
            normalized_positions.extend([x / width, y / height])

        while len(normalized_positions) < 6:
            normalized_positions.append(0.0)

        return torch.tensor(normalized_positions, dtype=torch.float32)

    def _create_heatmap_from_keypoints(self, image_size, keypoints):
        heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        for (x, y) in keypoints:
            x_clipped = int(round(np.clip(x, 0, image_size[0] - 1)))
            y_clipped = int(round(np.clip(y, 0, image_size[1] - 1)))
            if 0 <= x_clipped < image_size[0] and 0 <= y_clipped < image_size[1]:
                heatmap = cv2.circle(heatmap, (x_clipped, y_clipped), 5, 1, -1)
        return torch.tensor(heatmap).unsqueeze(0)

    def _create_star_features_from_keypoints(self, image_size, keypoints):
        width, height = image_size
        normalized_positions = []
        for (x, y) in keypoints:
            x_clipped = np.clip(x, 0, width - 1)
            y_clipped = np.clip(y, 0, height - 1)
            normalized_positions.extend([x_clipped / width, y_clipped / height])

        while len(normalized_positions) < 6:
            normalized_positions.append(0.0)

        return torch.tensor(normalized_positions, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# Configuration
image_size = 480  # Assuming your resize target is 128x128

transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Affine(
        scale=(0.9, 1.1),
        rotate=(-5, 5),
        shear=(-5, 5),
        keep_ratio=True,
        p=0.5,
        interpolation=cv2.INTER_LINEAR
    ),
    A.Perspective(
        scale=(0.01, 0.05),
        keep_size=True,
        p=0.3,
        interpolation=cv2.INTER_LINEAR
    ),
    A.ElasticTransform(
        alpha=50.0,
        sigma=4.0,
        alpha_affine=50.0 * 0.03,
        p=0.2,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101
    ),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))

# Initialize dataset
train_dataset = StarDataset(
    root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed",
    transform=transform
)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.labels),
    y=train_dataset.labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum()

# Initialize components
criterion = FocalLoss(alpha=class_weights, gamma=2.0)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

# Load Swin model
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=12,
    ignore_mismatched_sizes=True,
    output_hidden_states=True
)
model = SwinWithStarPositions(swin_model, num_classes=12).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
accumulation_steps = 4

# Checkpoint loading
save_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed_model"
os.makedirs(save_dir, exist_ok=True)

checkpoint_path = ""
start_epoch = 0 # Reset start epoch for training with new augmentations
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load only the model state dict, as optimizer state might not be compatible with new augmentation
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded pre-trained model weights from epoch {checkpoint['epoch']}")
    # If you want to continue training the optimizer as well, uncomment the following:
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"]
    # print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch.")

# Start training
train_model(model, train_loader, optimizer, criterion, device, start_epoch)
