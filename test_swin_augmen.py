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

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the model (same as during training)
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

# Define the dataset class (same as during training)
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

# Define the transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the test dataset
test_dataset = AugmentedDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented", transform=transform)

# Create the DataLoader for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# Load the trained model weights
model_path = "/media/student/B076126976123098/my_data/SiT/model_swin5_aug/swin_epoch_59.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)

# Test function
def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, heatmaps, star_features, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            star_features = star_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, heatmaps, star_features)
            _, predicted = torch.max(outputs, dim=1)
            
            # Update counts
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Compute accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Visualization function
def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    class_names = ["N" + str(i) for i in range(0, 360, 5)]  # Example class names
    
    with torch.no_grad():
        for i, (images, heatmaps, star_features, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            star_features = star_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, heatmaps, star_features)
            _, predicted = torch.max(outputs, dim=1)
            
            # Convert tensor to PIL image for visualization
            image = to_pil_image(images[0].cpu())
            true_label = class_names[labels[0].item()]
            predicted_label = class_names[predicted[0].item()]
            
            # Plot the image with true and predicted labels
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            plt.title(f"True: {true_label}, Predicted: {predicted_label}")
            plt.axis("off")
            plt.show()

# Test the model
test_accuracy = test_model(model, test_loader, device)

# Visualize predictions
visualize_predictions(model, test_loader, device, num_samples=5)

# 

# Test the model
test_accuracy = test_model(model, test_loader, device)

# Visualize predictions
visualize_predictions(model, test_loader, device, num_samples=5)
