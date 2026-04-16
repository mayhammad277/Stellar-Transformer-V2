import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import os
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import os

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
        heatmap_output_features = 64 * 4 * 4 #64 *4*4 
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

# Star detection and feature generation
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

def create_star_heatmap(image_shape, star_positions, original_image_size, sigma=5):
    """
    Create a heatmap for star positions.
    :param image_shape: Shape of the heatmap (height, width).
    :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
    :param original_image_size: Size of the original image (width, height).
    :param sigma: Standard deviation for Gaussian blobs.
    :return: Heatmap with Gaussian blobs at star positions.
    """
    heatmap = np.zeros(image_shape, dtype=np.float32)
    heatmap_height, heatmap_width = image_shape
    original_width, original_height = original_image_size
    
    for (x, y) in star_positions:
        # Scale star positions to heatmap dimensions
        scaled_x = int((x / original_width) * heatmap_width)
        scaled_y = int((y / original_height) * heatmap_height)
        
        # Ensure the scaled positions are within the heatmap bounds
        if 0 <= scaled_x < heatmap_width and 0 <= scaled_y < heatmap_height:
            heatmap[scaled_y, scaled_x] = 1.0  # Mark star positions
    
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
class StarFeatureGenerator:
    def __init__(self, heatmap_size=(256, 256), gaussian_sigma=5):
        """
        Initialize the star feature generator.
        
        Args:
            heatmap_size: Tuple (height, width) for the output heatmap size
            gaussian_sigma: Sigma value for Gaussian blur applied to heatmap
        """
        self.heatmap_size = heatmap_size
        self.gaussian_sigma = gaussian_sigma
        
        # Define color ranges for star detection in HSV space
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        self.lower_blue = np.array([100, 120, 70])
        self.upper_blue = np.array([130, 255, 255])
    
    def detect_star_positions(self, image):
        """
        Detect star positions in an image using color thresholding and contour detection.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            List of (x,y) tuples representing star positions
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create masks for red and blue stars
        red_mask = cv2.inRange(hsv, self.lower_red, self.upper_red)
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        
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
    
    def create_star_heatmap(self, star_positions, original_image_size):
        """
        Create a heatmap with Gaussian blobs at star positions.
        
        Args:
            star_positions: List of (x,y) tuples of star positions
            original_image_size: Tuple (width, height) of original image
            
        Returns:
            torch.Tensor: Heatmap tensor of shape (1, height, width)
        """
        heatmap = np.zeros(self.heatmap_size, dtype=np.float32)
        heatmap_height, heatmap_width = self.heatmap_size
        original_width, original_height = original_image_size
        
        for (x, y) in star_positions:
            # Scale star positions to heatmap dimensions
            scaled_x = int((x / original_width) * heatmap_width)
            scaled_y = int((y / original_height) * heatmap_height)
            
            # Ensure positions are within bounds
            if 0 <= scaled_x < heatmap_width and 0 <= scaled_y < heatmap_height:
                heatmap[scaled_y, scaled_x] = 1.0
        
        # Apply Gaussian blur and convert to tensor
        heatmap = gaussian_filter(heatmap, sigma=self.gaussian_sigma)
        return torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)
    
    def encode_star_positions(self, star_positions, image_size):
        """
        Encode star positions as a normalized feature vector.
        
        Args:
            star_positions: List of (x,y) tuples of star positions
            image_size: Tuple (width, height) of original image
            
        Returns:
            torch.Tensor: Feature vector of shape (6,) containing normalized positions
        """
        width, height = image_size
        normalized_positions = []
        
        for (x, y) in star_positions:
            normalized_x = x / width
            normalized_y = y / height
            normalized_positions.extend([normalized_x, normalized_y])
        
        # Pad with zeros if fewer than 3 stars are detected
        while len(normalized_positions) < 6:  # 3 stars * 2 coordinates
            normalized_positions.append(0.0)
        
        return torch.tensor(normalized_positions, dtype=torch.float32)
    
    def process_image(self, image):
        """
        Process an image to generate star features and heatmap.
        
        Args:
            image: Input image (PIL Image or file path)
            
        Returns:
            tuple: (star_features, heatmap, original_image)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        image_size = image.size  # (width, height)
        star_positions = self.detect_star_positions(image)
        
        heatmap = self.create_star_heatmap(star_positions, image_size)
        star_features = self.encode_star_positions(star_positions, image_size)
        
        return star_features, heatmap, image

class SkyClassifier:
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize the sky classifier.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on (None for auto-detection)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.feature_generator = StarFeatureGenerator()
        
        # Define transforms (match your training setup)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Initial resize to match heatmap size
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # For Swin input (after feature extraction)
        self.swin_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Swin input size
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _load_model(self, checkpoint_path):
        """Load the trained model from checkpoint."""
        # Load base Swin model
        swin_model = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            num_labels=72,
            ignore_mismatched_sizes=True,
            output_hidden_states=True
        )
        
        # Create custom model
        model = SwinWithStarPositions(swin_model, num_classes=72)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image):
        """
        Perform inference on an image.
        
        Args:
            image: Input image (PIL Image or file path)
            
        Returns:
            int: Predicted class index
            torch.Tensor: Class probabilities
        """
        # Generate features
        star_features, heatmap, image = self.feature_generator.process_image(image)
        
        # Transform image for Swin input
        image_tensor = self.swin_transform(image).unsqueeze(0).to(self.device)
        heatmap = heatmap.unsqueeze(0).to(self.device)  # Add batch dim
        star_features = star_features.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor, heatmap, star_features)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return predicted_class, probabilities.squeeze()

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = SkyClassifier(
        checkpoint_path="/media/student/B076126976123098/my_data/SiT/model_swin7_aug/swin_epoch_63.pth"
    )
    
    # Example image path
    image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated/N0_4.png"
    
    # Perform prediction
    predicted_class, probabilities = classifier.predict(image_path)
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities.cpu().numpy()}")
