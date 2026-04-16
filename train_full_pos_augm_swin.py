import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
import json 

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the augmented dataset


# Create the DataLoader
def create_star_heatmap(self, image_shape, star_positions, sigma=5):
        """
        Create a heatmap for star positions.
        :param image_shape: Shape of the image (height, width).
        :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
        :param sigma: Standard deviation for Gaussian blobs.
        :return: Heatmap with Gaussian blobs at star positions.
        """
        heatmap = np.zeros(image_shape, dtype=np.float32)
        for (x, y) in star_positions:
            heatmap = cv2.circle(heatmap, (int(x), int(y)), sigma, 1, -1)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return heatmap

def encode_star_positions(self, star_positions, image_size):
        """
        Encode star positions as a normalized feature vector.
        :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
        :param image_size: Size of the image (width, height).
        :return: Normalized feature vector of star positions.
        """
        width, height = image_size
        normalized_positions = []
        for (x, y) in star_positions:
            normalized_x = x / width
            normalized_y = y / height
            normalized_positions.extend([normalized_x, normalized_y])
        
        # Pad with zeros if fewer than 3 stars are detected
        while len(normalized_positions) < 6:  # 3 stars * 2 coordinates each
            normalized_positions.append(0.0)
        
        return torch.tensor(normalized_positions, dtype=torch.float32)
        
        
        
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
        print(f"Swin output shape: {swin_output.shape}")  # Debugging
        
        # Pass the heatmap through the heatmap branch
        heatmap_output = self.heatmap_conv(heatmap)
        print(f"Heatmap output shape: {heatmap_output.shape}")  # Debugging
        
        # Concatenate the Swin output, heatmap features, and star position features
        combined = torch.cat([swin_output, heatmap_output, star_features], dim=1)
        print(f"Combined shape: {combined.shape}")  # Debugging
        
        # Pass through a fully connected layer
        output = self.fc(combined)
        return output
         
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
        heatmap = torch.load(self.heatmap_paths[idx])
        
        # Load star features
        star_features = torch.load(self.star_features_paths[idx])
        
        # Load label
        label = self.labels[idx]
        
        return image, heatmap, star_features, label
        
# Load the model
num_classes=72
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
augmented_dataset = AugmentedDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented_moise", transform=transform)
batch_size = 8
train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=num_classes ,ignore_mismatched_sizes=True,output_hidden_states=True,output_attentions=True,return_dict=True)       
model = SwinWithStarPositions(swin_model, num_classes=num_classes)
model.to(device)

test_dataset = AugmentedDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented_moise", transform=transform)

save_dir = "/media/student/B076126976123098/my_data/SiT/model_swin5_aug/"
#mkdir $save_dir





# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=80):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
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
        checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Save loss history
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

# Train the model
train_model(model, train_loader, optimizer, criterion, device)        


# Define the transform

