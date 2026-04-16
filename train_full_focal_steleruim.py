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
import pandas as pd
from pathlib import Path

import re

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
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir


        self.star_map_paths = []
        self.labels = []
        num_classes=12

        self.annotation_df = pd.read_csv("/home/student/star_tracker/image_labels.csv")
        #self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.annotation_image_names = self.annotation_df['image_name'].tolist()
        self.labels = self.annotation_df['orientation_class'].tolist()
        self.all_image_files = self.annotation_df['full_image_path'].tolist()
        self.all_image_files = [os.path.abspath(p) for p in self.all_image_files]
        
        #self.all_image_files=[name.replace(".jpg", ".png") for name in self.all_image_files]
        #self.all_image_files=[Path(str(f)) for f in self.all_image_files]
        
        self.star_maps = self.annotation_df['full_star_map_path'].tolist()     
        
        self.star_maps=[Path(str(f)) for f in self.star_maps]
        
    def _find_matching_image(self, annotation_img_name):
        # Strip "img_" and replace with "stars_"
        potential_base_name = annotation_img_name
        print("potential_base_name",potential_base_name)
        # Look for files that start with the potential base name and are image files
        matching_files = [f for f in self.all_image_files if f.startswith(potential_base_name) ]

        if matching_files:
            # If multiple matches (e.g., different augmentations), you might need a more specific rule.
            # For now, we'll return the first one found. You might need to refine this.
            return matching_files[0]
        return None        
    def _modify_filename(self, filename):
        """Removes trailing zero if the number part ends in zero."""
        match = re.search(r'stars-(\d+)\b', filename)
        if match:
            number_str = match.group(1)
            if "00" in number_str and len(number_str) ==3:  # Ensure we don't strip the only zero
                #modified_number_str = re.sub(r'0', '', number_str) #number_str.rstrip('0')
                #modified_filename = filename.replace(f'stars-{number_str}', f'stars-{modified_number_str}')
                modified=re.sub(r'00', '0', filename)
                print(modified)
                return modified                
            elif   "0" in number_str  and len(number_str) > 3:
                modified=re.sub(r'0', '', filename)
                return modified
        return filename
    def _get_class_to_idx(self):
        classes = sorted(self.annotation_df['orientation_class'].unique())
        return {str(cls): idx for idx, cls in enumerate(classes)}        
        # Iterate through class folders in images directory

    def __len__(self):
        return len(self.all_image_files)

    def __getitem__(self, idx):
      max_idx = len(self.all_image_files)
    
      while idx < max_idx:
        path = self.all_image_files[idx]
        #print(f"Trying: {path}")
        
        if os.path.exists(path):
            image = Image.open(path).convert("RGB")
            annotation_img_name = self.annotation_image_names[idx]
            label = self.labels[idx]

            # Load star map JSON
            with open(self.star_maps[idx]) as f:
                star_data = json.load(f)

            # Create heatmap and star features
            heatmap = self._create_heatmap(image.size, star_data)
            star_features = torch.tensor(self._create_star_features(star_data), dtype=torch.float)


            # Only apply transform to image
            if self.transform:
                image = self.transform(image)

            return image, heatmap, star_features, label
        else:
            #print(f"File not found: {path}, skipping to next index...")
            idx += 1

      raise IndexError("No valid image found from given index onward.")

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
            normalized_positions.extend([x/width, y/height])
        
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
def custom_collate(batch):
    import torch
    images = torch.stack([item[0] for item in batch])
    heatmaps = torch.stack([item[1] for item in batch])
    
    features = [item[2] for item in batch]  # If variable length, keep as list
    features = torch.stack([item[2] for item in batch])  # Now a tensor
    #features = features.to(device)

    labels = torch.tensor([item[3] for item in batch])
    return images, heatmaps, features, labels
    
    
    
# Configuration
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Initialize dataset
train_dataset = StarDataset(
    root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/",

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
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,collate_fn=custom_collate)

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

checkpoint_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed_model/swin_epoch_27.pth"
start_epoch = 28
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch}")

# Start training
train_model(model, train_loader, optimizer, criterion, device, start_epoch)
