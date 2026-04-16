from PIL import Image, ImageEnhance
import numpy as np
import os
import cv2
import torch
from torchvision import transforms
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
from torchvision.transforms.functional import to_pil_image

def tensor_to_pil(tensor):
    """
    Convert a PyTorch tensor to a PIL image.
    :param tensor: Input tensor (C, H, W).
    :return: PIL image.
    """
    return to_pil_image(tensor)
def detect_star_positions(image):
    """
    Detect star positions in an image using color thresholding and blob detection.
    :param image: Input image (PIL or NumPy array).
    :return: List of star positions [(x1, y1), (x2, y2), ...].
    """
    # Convert PIL image to NumPy array (if necessary)
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
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.star_positions = []  # Store star positions for each image
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(os.listdir(root_dir))}

        for cls_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Detect star positions
        star_positions = detect_star_positions(img)
        
        # Encode star positions as a feature vector
        star_features = self.encode_star_positions(star_positions, img.size)
        
        # Create a heatmap for star positions
        heatmap = self.create_star_heatmap((256, 256), star_positions)
        
        if self.transform:
            img = self.transform(img)
        
        return img, heatmap, star_features, label

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
lass SafeAugmentation:
    def __init__(self):
        pass

    def add_noise(self, image, noise_level=0.01):
        """
        Add Gaussian noise to the image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_low_random_noise(self, image, noise_level=0.01):
        """
        Add low random Gaussian noise to the image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def add_bright_spots(self, image, num_spots=3, max_radius=10, max_intensity=50):
        """
        Add subtle bright spots to the image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w, _ = image.shape
        output_image = np.copy(image)
        
        for _ in range(num_spots):
            # Randomly choose the center of the bright spot
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            
            # Randomly choose the radius and intensity
            radius = np.random.randint(3, max_radius)
            intensity = np.random.randint(20, max_intensity)
            
            # Create a non-circular bright spot using a random shape
            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    if 0 <= cx + dx < w and 0 <= cy + dy < h:
                        # Add intensity with a random decay factor
                        decay = random.uniform(0.5, 1.0)
                        output_image[cy + dy, cx + dx] = np.clip(
                            output_image[cy + dy, cx + dx] + int(intensity * decay), 0, 255
                        )
        
        return Image.fromarray(output_image)

    def adjust_brightness_contrast(self, image, brightness_factor=1.0, contrast_factor=1.0):
        """
        Adjust the brightness and contrast of the image.
        """
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image

    def add_occlusions(self, image, num_patches=3, patch_size=50):
        """
        Add random occlusions to the image.
        """
        image = np.array(image)
        h, w, _ = image.shape
        
        for _ in range(num_patches):
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            image[y:y+patch_size, x:x+patch_size] = 0  # Set patch to black
        
        return Image.fromarray(image)

    def change_background(self, image, color=(0, 0, 0)):
        """
        Change the background of the image to a solid color.
        """
        image = np.array(image)
        mask = np.all(image == [0, 0, 0], axis=-1)  # Mask for black pixels (background)
        image[mask] = color
        return Image.fromarray(image)

    def __call__(self, image):
        """
        Apply all augmentations to the image in a single forward pass.
        """
        # Apply subtle brightness and contrast adjustment
        brightness_factor = random.uniform(0.95, 1.05)  # Very small brightness adjustment
        contrast_factor = random.uniform(0.95, 1.05)  # Very small contrast adjustment
        image = self.adjust_brightness_contrast(image, brightness_factor, contrast_factor)
        
        # Add low random noise
        image = self.add_low_random_noise(image, noise_level=0.01)
        
        # Add subtle bright spots
        image = self.add_bright_spots(image, num_spots=random.randint(1, 3), max_radius=8, max_intensity=50)
        
        # Add Gaussian noise (optional, can be removed if not needed)
        image = self.add_noise(image, noise_level=0.01)
        
        # Add occlusions (optional, can be removed if not needed)
        image = self.add_occlusions(image, num_patches=2, patch_size=30)
        
        # Change background (optional, can be removed if not needed)
        image = self.change_background(image, color=(10, 10, 30))  # Dark blue background
        
        return image
        
        
def save_augmented_dataset(original_dataset, augmentation, output_dir, num_augmented_samples=5):
    """
    Save an augmented dataset with the same format as the original dataset.
    
    Args:
        original_dataset: Original dataset (CustomDataset object).
        augmentation: Augmentation pipeline (SafeAugmentation object).
        output_dir: Directory to save the augmented dataset.
        num_augmented_samples: Number of augmented samples to generate per original sample.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in range(len(original_dataset)):
        image_tensor, heatmap, star_features, label = original_dataset[idx]
        
        # Convert the image tensor to a PIL image
        image = tensor_to_pil(image_tensor)
        
        # Create a subdirectory for the class
        class_name = f"N{label * 5}"  # Assuming labels are 0, 1, 2, ..., 71
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save the original sample
        original_image_path = os.path.join(class_dir, f"original_{idx}.jpg")
        image.save(original_image_path)
        
        # Save the original heatmap and star features
        original_heatmap_path = os.path.join(class_dir, f"original_{idx}_heatmap.pt")
        torch.save(heatmap, original_heatmap_path)
        
        original_star_features_path = os.path.join(class_dir, f"original_{idx}_star_features.pt")
        torch.save(star_features, original_star_features_path)
        
        # Generate and save augmented samples
        for aug_idx in range(num_augmented_samples):
            augmented_image = augmentation(image)
            
            # Save the augmented image
            augmented_image_path = os.path.join(class_dir, f"augmented_{idx}_{aug_idx}.jpg")
            augmented_image.save(augmented_image_path)
            
            # Save the same heatmap and star features for the augmented image
            augmented_heatmap_path = os.path.join(class_dir, f"augmented_{idx}_{aug_idx}_heatmap.pt")
            torch.save(heatmap, augmented_heatmap_path)
            
            augmented_star_features_path = os.path.join(class_dir, f"augmented_{idx}_{aug_idx}_star_features.pt")
            torch.save(star_features, augmented_star_features_path)
            
            
            
            
# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the original dataset
original_dataset = CustomDataset(root_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder", transform=transform)

# Define the augmentation pipeline
augmentation = SafeAugmentation()

# Save the augmented dataset
output_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder_augmented"
save_augmented_dataset(original_dataset, augmentation, output_dir, num_augmented_samples=5)                     
