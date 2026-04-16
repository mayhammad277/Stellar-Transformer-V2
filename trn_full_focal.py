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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SwinForImageClassification, SwinConfig
import os
from PIL import Image
from tqdm import tqdm
import numpy as np 
import cv2
from transformers import AutoImageProcessor, Swinv2ForImageClassification
from transformers import Swinv2Config, Swinv2Model
from torch.cuda.amp import GradScaler, autocast
import json
# Define the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()
class SwinWithStarPositions(nn.Module):
    def __init__(self, swin_model, num_classes):
        super(SwinWithStarPositions, self).__init__()
        self.swin = swin_model
        # Add a separate branch for processing the heatmap
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce feature map size
            nn.Flatten()
        )
        # Calculate the number of input features for the fully connected layer
        swin_output_features = self.swin.config.hidden_size  # Hidden size of Swin Transformer
        heatmap_output_features = 64 * 4 * 4  # Output size of the heatmap branch
        star_features_size = 6  # 3 stars * 2 coordinates each
        combined_features = swin_output_features + heatmap_output_features + star_features_size
        
        # Define the fully connected layer
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
        
        # Load heatmap
        heatmap = torch.load(self.heatmap_paths[idx], weights_only=True)  # Set weights_only=True
        
        # Load star features
        star_features = torch.load(self.star_features_paths[idx], weights_only=True)  # Set weights_only=True
        
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

# Function to flatten features
def flatten_features(image, heatmap, star_features):
    """
    Flatten the image, heatmap, and star features into a single feature vector.
    """
    # Flatten the image tensor (C, H, W) -> (C*H*W,)
    image_flat = image.view(-1).cpu().numpy()
    
    # Flatten the heatmap tensor (1, H, W) -> (H*W,)
    heatmap_flat = heatmap.view(-1).cpu().numpy()
    
    # Flatten the star features tensor (6,) -> (6,)
    star_features_flat = star_features.view(-1).cpu().numpy()
    
    # Concatenate all features into a single vector
    features_flat = np.concatenate([image_flat, heatmap_flat, star_features_flat])
    return features_flat

# Function to reconstruct features
def reconstruct_features(features_flat, image_shape, heatmap_shape, star_features_shape):
    """
    Reconstruct the image, heatmap, and star features from a flattened feature vector.
    """
    # Extract the image, heatmap, and star features from the flattened vector
    image_size = np.prod(image_shape)
    heatmap_size = np.prod(heatmap_shape)
    star_features_size = np.prod(star_features_shape)
    
    image_flat = features_flat[:image_size]
    heatmap_flat = features_flat[image_size:image_size + heatmap_size]
    star_features_flat = features_flat[image_size + heatmap_size:image_size + heatmap_size + star_features_size]
    
    # Reshape to original dimensions
    image = torch.tensor(image_flat).view(image_shape)
    heatmap = torch.tensor(heatmap_flat).view(heatmap_shape)
    star_features = torch.tensor(star_features_flat).view(star_features_shape)
    
    return image, heatmap, star_features

# Define the transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the dataset
train_dataset = AugmentedDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented_moise", transform=transform)

# Flatten the dataset for resampling
X_flat = []  # List of flattened feature vectors
y = []       # List of labels

for image, heatmap, star_features, label in train_dataset:
    X_flat.append(flatten_features(image, heatmap, star_features))
    y.append(label)

# Convert to numpy arrays
X_flat = np.array(X_flat)
y = np.array(y)

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, y)

# Reconstruct the resampled dataset
train_dataset_resampled = []
for features_flat, label in zip(X_resampled, y_resampled):
    image, heatmap, star_features = reconstruct_features(
        features_flat,
        image_shape=(3, 128, 128),  # Shape of the image tensor
        heatmap_shape=(1, 128, 128),  # Shape of the heatmap tensor
        star_features_shape=(6,)  # Shape of the star features tensor
    )
    train_dataset_resampled.append((image, heatmap, star_features, label))

# Create the DataLoader for the resampled dataset

batch_size = 1
train_loader = DataLoader(train_dataset_resampled, batch_size=batch_size, shuffle=True)

# Load the Swin model
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=72,
    ignore_mismatched_sizes=True,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True
)

# Initialize the model
model = SwinWithStarPositions(swin_model, num_classes=72)
model.to(device)

# Compute class weights for Focal Loss
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_resampled),
    y=y_resampled
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Initialize Focal Loss
criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, dataloader, optimizer, criterion, device, start_epoch,num_epochs=80):
    model.train()
    loss_history = []

    for epoch in range(start_epoch,num_epochs):
        total_loss = 0
        for images, heatmaps, star_features, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            star_features = star_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, heatmaps, star_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        #checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        #torch.save(model.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        torch.save({
        "epoch": epoch + 1,  # Save the last completed epoch
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),}, checkpoint_path)
        # Save loss history
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

# Train the model
save_dir ="/media/student/B076126976123098/my_data/SiT/model_swin7_aug"
#checkpoint_path="/media/student/B076126976123098/my_data/SiT/model_swin6_aug/swin_epoch_17.pth"
checkpoint_path=None
start_epoch = 0
if torch.cuda.is_available() :

    
     checkpoint = torch.load(checkpoint_path) if os.path.exists(checkpoint_path) else None
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))

else:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) if os.path.exists(checkpoint_path) else None
 
if checkpoint:
    #model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #torch.save({
    #    "epoch": epoch + 1,  # Save the last completed epoch
    #    "model_state_dict": model.state_dict(),
    #    "optimizer_state_dict": optimizer.state_dict(),}, checkpoint_path)    
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    print(f"🔄 Resuming training from epoch {start_epoch}...")
train_model(model, train_loader, optimizer, criterion, device,start_epoch)
