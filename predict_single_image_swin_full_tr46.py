import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
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



import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

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
        heatmap[int(y), int(x)] = 1.0  # Mark star positions
    heatmap = gaussian_filter(heatmap, sigma=sigma)  # Apply Gaussian blur
    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return heatmap

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

def process_single_image(image_path, output_dir):
    """
    Process a single image to generate heatmap and star features.
    :param image_path: Path to the input image.
    :param output_dir: Directory to save the heatmap and star features.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_size = image.size  # (width, height)
    
    # Detect star positions
    star_positions = detect_star_positions(image)
    print(f"Detected star positions: {star_positions}")
    
    # Generate heatmap
    heatmap = create_star_heatmap((256, 256), star_positions, sigma=5)
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Encode star features
    star_features = encode_star_positions(star_positions, image_size)
    print(f"Star features: {star_features}")
    
    # Save heatmap and star features
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "heatmap.pt")
    star_features_path = os.path.join(output_dir, "star_features.pt")
    
    torch.save(heatmap, heatmap_path)
    torch.save(star_features, star_features_path)
    print(f"Heatmap saved to: {heatmap_path}")
    print(f"Star features saved to: {star_features_path}")
    return star_features,heatmap


# Define the model (same as in training)
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

# Load the trained model
def load_model(checkpoint_path, num_classes=72):
    # Load the Swin model
    swin_model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )
    model = SwinWithStarPositions(swin_model, num_classes=num_classes)
    
    # Load the checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input
def preprocess_input(image, heatmap, star_features, transform):
    """
    Preprocess the input for inference.
    :param image: Input image (PIL or path).
    :param heatmap: Heatmap (Tensor or path to .pt file).
    :param star_features: Star features (Tensor or path to .pt file).
    :param transform: Image transformation pipeline.
    :return: Preprocessed inputs.
    """
    # Load and transform the image
    if isinstance(image, str):  # If image is a file path
        image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load heatmap if it's a file path, otherwise use the Tensor directly
    if isinstance(heatmap, str):  # If heatmap is a file path
        heatmap = torch.load(heatmap, weights_only=True)
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 2 else heatmap  # Add batch dimension if needed
    
    # Load star features if it's a file path, otherwise use the Tensor directly
    if isinstance(star_features, str):  # If star_features is a file path
        star_features = torch.load(star_features, weights_only=True)
    star_features = star_features.unsqueeze(0) if star_features.dim() == 1 else star_features  # Add batch dimension if needed
    
    return image, heatmap, star_features

# Define the transform (same as in training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Inference function
def infer_single_image(model, image_path, heatmap_path, star_features_path, transform, device):
    # Preprocess the input
    image, heatmap, star_features = preprocess_input(image_path, heatmap_path, star_features_path, transform)
    
    # Move inputs to the device
    image = image.to(device)
    heatmap = heatmap.to(device)
    star_features = star_features.to(device)
    
    # Run the model
    with torch.no_grad():  # Disable gradient calculation
        output = model(image, heatmap, star_features)
        _, predicted_class = torch.max(output, 1)  # Get the predicted class
    
    return predicted_class.item()

# Main function for inference
if __name__ == "__main__":
    # Define paths
    checkpoint_path = "/media/student/B076126976123098/my_data/SiT/model_swin6_aug/swin_epoch_52.pth"
    image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder/N225/N225_0.png"
    heatmap_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented/N225/augmented_728_0_heatmap.pt"
    star_features_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented/N225/augmented_728_0_star_features.pt"
    # Path to the input image

    # Directory to save the heatmap and star features
    output_dir = "/media/student/B076126976123098/my_data/SiT/test_output"
    
    # Process the single image
    #star_features,heatmap=process_single_image(image_path, output_dir)    
    heatmap = torch.load(heatmap_path, weights_only=True).unsqueeze(0)  # Add batch dimension
    star_features = torch.load(star_features_path, weights_only=True).unsqueeze(0)  # Add batch dimension
    print(heatmap.shape)    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path).to(device)
    
    # Perform inference
    predicted_class = infer_single_image(model, image_path, heatmap, star_features, transform, device)
    print(f"Predicted Class: {predicted_class}")
