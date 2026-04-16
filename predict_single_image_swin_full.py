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
configuration = Swinv2Config()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def encode_star_positions(star_positions, image_size):
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
def detect_star_positions(image):
    """
    Detect star positions in an image using color thresholding and blob detection.
    :param image: Input image (PIL or NumPy array).
    :return: List of star positions [(x1, y1), (x2, y2), ...].
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define color ranges for red and blue stars
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])
    
    # Create masks for red and blue stars
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine masks
    star_mask = cv2.bitwise_or(red_mask, blue_mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(star_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get centroids of the contours
    star_positions = []
    for contour in contours:
        # Calculate moments to find the centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            star_positions.append((cx, cy))
    
    return star_positions
    
    
def create_star_heatmap(image_shape, star_positions, sigma=5):
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
data_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder"
batch_size = 8
num_classes = 72


save_dir = "/media/student/B076126976123098/my_data/SiT/model_swin4/"
#mkdir $save_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dataset


# Load Swin model and redefine classifier head
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=num_classes ,ignore_mismatched_sizes=True,output_hidden_states=True,output_attentions=True,return_dict=True)
model = SwinWithStarPositions(swin_model, num_classes=num_classes)
model.to(device)



        




def predict_single_image(model, img_path, transform, device):
    """
    Predict the class of a single image using the trained model.
    
    Args:
        model: Trained model.
        img_path: Path to the input image.
        transform: Image preprocessing transform.
        device: Device to run the model on (e.g., "cuda" or "cpu").
    
    Returns:
        pred: Predicted class probabilities.
        predicted_label: Predicted class label.
    """
    # Load and preprocess the image
    img = Image.open(img_path).convert("RGB")
    
    # Detect star positions
    star_positions = detect_star_positions(img)
    
    # Create a heatmap for star positions
    heatmap = create_star_heatmap((256, 256), star_positions)
    
    # Encode star positions as a feature vector
    star_features = encode_star_positions(star_positions, img.size)
    
    # Apply the transform to the image
    img = transform(img)
    
    # Convert to tensors and add batch dimension
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    if not isinstance(heatmap, torch.Tensor):
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
    if not isinstance(star_features, torch.Tensor):
        star_features = torch.tensor(star_features, dtype=torch.float32)
    
    img = img.unsqueeze(0)  # Add batch dimension: (C, H, W) -> (1, C, H, W)
    heatmap = heatmap.unsqueeze(0)  # Add batch dimension: (1, H, W) -> (1, 1, H, W)
    star_features = star_features.unsqueeze(0)  # Add batch dimension: (6,) -> (1, 6)
    
    # Move tensors to the same device as the model
    img, heatmap, star_features = img.to(device), heatmap.to(device), star_features.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Pass the image, heatmap, and star features through the model
        outputs = model(img, heatmap, star_features)
        
    # Get predicted class probabilities
    pred = torch.softmax(outputs, dim=1)
    
    # Get the predicted class label
    predicted_class = torch.argmax(pred, dim=1).item()
    class_names = ["N" + str(i) for i in range(0, 360, 5)]  # Example class names
    predicted_label = class_names[predicted_class]
    
    print(f"Predicted class: {predicted_label}")
    print(f"Class probabilities: {pred.squeeze().cpu().numpy()}")
    
    return pred, predicted_label
    
    
    

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the trained model
model = SwinWithStarPositions(swin_model, num_classes=72)
model.load_state_dict(torch.load("/media/student/B076126976123098/my_data/SiT/model_swin4/swin_epoch_80.pth"))
model.to(device)

# Predict a single image
img_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder/N25/N25_4.png"
pred, predicted_label = predict_single_image(model, img_path, transform, device)

print(f"Predicted class: {predicted_label}")
print(f"Class probabilities: {pred.squeeze().cpu().numpy()}")            
