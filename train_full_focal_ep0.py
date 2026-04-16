import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from transformers import Swinv2ForImageClassification
from torch.cuda.amp import GradScaler, autocast
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()  # Correct initialization
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

            with autocast():  # Mixed precision training
                outputs = model(images, heatmaps, star_features)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # Clear unused variables and free GPU memory
            del images, heatmaps, star_features, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        # Save loss history
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)
# Define the model
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
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce feature map size
            nn.Flatten()
        )
        swin_output_features = self.swin.config.hidden_size  # Hidden size of Swin Transformer
        heatmap_output_features = 64 * 4 * 4  # Output size of the heatmap branch
        star_features_size = 6  # 3 stars * 2 coordinates each
        combined_features = swin_output_features + heatmap_output_features + star_features_size
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, image, heatmap, star_features):
        # Pass the image through the Swin Transformer
        outputs = self.swin(image)
        swin_output = outputs.hidden_states[-1].mean(dim=1)  # Use the last hidden state and pool it
        
        # Pass the heatmap through the heatmap branch
        heatmap_output = self.heatmap_conv(heatmap)
        
        # Concatenate the Swin output, heatmap features, and star position features
        combined = torch.cat([swin_output, heatmap_output, star_features], dim=1)
        
        # Pass through a fully connected layer
        output = self.fc(combined)
        return output

# Define the dataset class
class AugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.heatmap_paths = []
        self.star_features_paths = []
        self.labels = []
        
        # Iterate through the class directories
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                # Extract label from class name (e.g., "N0" -> 0, "N5" -> 1, etc.)
                print(class_name)
                label = int(class_name[1:]) // 5
                
                # Collect all files in the class directory
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(".jpg"):
                        # Paths for image, heatmap, and star features
                        image_path = os.path.join(class_dir, file_name)
                        heatmap_path = os.path.join(class_dir, file_name.replace(".jpg", "_heatmap.pt"))
                        star_features_path = os.path.join(class_dir, file_name.replace(".jpg", "_star_features.pt"))
                        
                        # Add to lists
                        self.image_paths.append(image_path)
                        self.heatmap_paths.append(heatmap_path)
                        self.star_features_paths.append(star_features_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load heatmap and star features on CPU
        heatmap = torch.load(self.heatmap_paths[idx], map_location="cpu", weights_only=True)
        star_features = torch.load(self.star_features_paths[idx], map_location="cpu", weights_only=True)
        
        # Load label
        label = self.labels[idx]
        
        return image, heatmap, star_features, label

# Define Focal Loss
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

# Define the transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the dataset
train_dataset = AugmentedDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized", transform=transform)

# Compute class weights for Focal Loss
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.labels),
    y=train_dataset.labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weights = class_weights / class_weights.sum()  # Normalize weights

# Initialize Focal Loss
criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")

# Create the DataLoader for the resampled dataset
batch_size = 1  # Reduce batch size further
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

# Load the Swin model
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",  # Use a smaller model
    num_labels=72,
    ignore_mismatched_sizes=True,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True
)

# Initialize the model
model = SwinWithStarPositions(swin_model, num_classes=72)
model.to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Train the model
save_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/new_model"
os.makedirs(save_dir, exist_ok=True)

# Start training
accumulation_steps = 4  # Gradient accumulation steps

checkpoint_path="None"#"/media/student/B076126976123098/my_data/SiT/model_swin7_aug/swin_epoch_47.pth"
start_epoch = 0
if torch.cuda.is_available() :

    
     checkpoint = torch.load(checkpoint_path) if os.path.exists(checkpoint_path) else None
    
else:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) if os.path.exists(checkpoint_path) else None
 
if checkpoint:

    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    print(f"🔄 Resuming training from epoch {start_epoch}...")
train_model(model, train_loader, optimizer, criterion, device,start_epoch)
train_model(model, train_loader, optimizer, criterion, device, start_epoch=18, num_epochs=80)
