import os
import torch
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import Swinv2ForImageClassification
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random  # Import the random module

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

def create_heatmap_from_keypoints(image_size, keypoints):
    """
    Creates a heatmap from a list of keypoint coordinates.

    Args:
        image_size (tuple): (width, height) of the image.
        keypoints (list): List of (x, y) keypoint coordinates.

    Returns:
        torch.Tensor: The heatmap tensor.
    """
    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for (x, y) in keypoints:
        x_clipped = int(round(np.clip(x, 0, image_size[0] - 1)))
        y_clipped = int(round(np.clip(y, 0, image_size[1] - 1)))
        if 0 <= x_clipped < image_size[0] and 0 <= y_clipped < image_size[1]:
            heatmap = cv2.circle(heatmap, (x_clipped, y_clipped), 5, 1, -1)
    return torch.tensor(heatmap).unsqueeze(0)

def create_star_features_from_keypoints(image_size, keypoints):
    """
    Creates star features from a list of keypoint coordinates.

    Args:
        image_size (tuple): (width, height) of the image.
        keypoints (list): List of (x, y) keypoint coordinates.

    Returns:
        torch.Tensor: The star features tensor.
    """
    width, height = image_size
    normalized_positions = []
    for (x, y) in keypoints:
        x_clipped = np.clip(x, 0, width - 1)
        y_clipped = np.clip(y, 0, height - 1)
        normalized_positions.extend([x_clipped / width, y_clipped / height])

    while len(normalized_positions) < 6:
        normalized_positions.append(0.0)

    return torch.tensor(normalized_positions, dtype=torch.float32)

def process_star_map(star_map_path, image_size):
    """Process star map JSON file into heatmap and features"""
    with open(star_map_path) as f:
        star_data = json.load(f)

    # Create heatmap
    star_positions = [star_data['centroid']['position']]
    if 'reference_stars' in star_data:
        star_positions.extend([star['position'] for star in star_data['reference_stars']])

    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for (x, y) in star_positions:
        heatmap = cv2.circle(heatmap, (int(x), int(y)), 5, 1, -1)
    heatmap = torch.tensor(heatmap).unsqueeze(0)

    # Create star features
    width, height = star_data['image_shape'][1], star_data['image_shape'][0]
    normalized_positions = []
    for (x, y) in star_positions:
        normalized_positions.extend([x / width, y / height])

    # Pad to 6 features (3 stars × 2 coordinates)
    while len(normalized_positions) < 6:
        normalized_positions.append(0.0)

    star_features = torch.tensor(normalized_positions, dtype=torch.float32)

    return heatmap, star_features



def augment_and_save(image_path, star_map_path, output_dir, num_augmentations=50, image_size=128):
    """
    Augments an image and its corresponding star map, and saves the augmented images and star maps.

    Args:
        image_path (str): Path to the input image.
        star_map_path (str): Path to the input star map JSON file.
        output_dir (str): Path to the directory where augmented images and star maps will be saved.
        num_augmentations (int, optional): Number of augmentations to generate per image. Defaults to 50.
        image_size (int, optional): The size to resize the image to before augmentation. Defaults to 128.
    """

    # Load image
    image = Image.open(image_path).convert("RGB")
    original_image_width, original_image_height = image.size

    # Load star map data
    with open(star_map_path) as f:
        star_data = json.load(f)

    # Extract keypoints (star positions) from the original star map
    original_star_positions_list = [star_data['centroid']['position']]
    if 'reference_stars' in star_data:
        original_star_positions_list.extend([star['position'] for star in star_data['reference_stars']])
    original_keypoints = [(x, y) for x, y in original_star_positions_list]

    # Define the augmentation pipeline.  Use the same one as in training.
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

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_augmentations):
        # Create a unique filename for each augmented image and star map
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        augmented_image_filename = f"{base_filename}_aug_{i}.png"
        augmented_star_map_filename = f"{base_filename}_aug_{i}_star_map.json"

        # Apply the augmentation
        #  The key is to pass the original keypoints.
        transformed = transform(image=np.array(image), keypoints=original_keypoints, class_labels=[0] * len(original_keypoints))  # Use dummy class labels

        augmented_image = transformed['image']  # This is a tensor
        transformed_keypoints = transformed['keypoints'] # Get the transformed keypoints

        # Convert the augmented image tensor back to a PIL Image and save it
        augmented_image_pil = transforms.ToPILImage()(augmented_image)
        augmented_image_path = os.path.join(output_dir, augmented_image_filename)
        augmented_image_pil.save(augmented_image_path)

        # Create the augmented star map data.  Crucially, use the *transformed* keypoints.
        augmented_star_positions = [(x, y) for x, y in transformed_keypoints]
        augmented_star_map_data = {
            'image_shape': [original_image_width, original_image_height], # Use the *original* image shape
            'centroid': {'position': augmented_star_positions[0]},
            'reference_stars': [{'position': pos} for pos in augmented_star_positions[1:]],
        }

        # Save the augmented star map data to a JSON file
        augmented_star_map_path = os.path.join(output_dir, augmented_star_map_filename)
        with open(augmented_star_map_path, 'w') as f:
            json.dump(augmented_star_map_data, f, indent=4)

        print(f"Saved augmented image and star map: {augmented_image_filename}, {augmented_star_map_filename}")



if __name__ == "__main__":
    # Define the classes to process
    classes = ["N_N0", "N_N30", "N_N60", "N_N90", "N_N120", "N_N150", "N_N180", "N_N210", "N_N240", "N_N270", "N_N300", "N_N330"]
    base_image_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/images"
    base_star_map_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/starmaps"
    base_output_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/augmented_data"  # Parent directory for all augmented classes
    num_augmentations = 50
    image_size = 128

    for class_name in classes:
        # Construct the input and output directories for the current class
        image_dir = os.path.join(base_image_dir, class_name)
        star_map_dir = os.path.join(base_star_map_dir, class_name)
        output_dir = os.path.join(base_output_dir, f"{class_name}_augmented")  # e.g., augmented_data/N_N0_augmented

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through the images in the current class directory
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, img_file)
                star_map_file = f"{os.path.splitext(img_file)[0]}_star_map.json"
                star_map_path = os.path.join(star_map_dir, star_map_file)

                # Check if the star map file exists
                if os.path.exists(star_map_path):
                    print(f"Processing: {image_path}, {star_map_path}")
                    augment_and_save(image_path, star_map_path, output_dir, num_augmentations, image_size)
                else:
                    print(f"Warning: Star map file not found for {image_path}. Skipping.")

    print("Augmentation and star map generation complete for all classes.")

