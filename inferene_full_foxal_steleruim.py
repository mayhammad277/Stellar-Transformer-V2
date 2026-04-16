import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import cv2
import numpy as np
import pandas as pd
import re # For filename modification
from transformers import Swinv2ForImageClassification

# --- Model Definition (copied from your training script for self-containment) ---
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
        # Ensure that the swin_output_features correctly reflects the output dimension of the Swin backbone
        # When `output_hidden_states=True` and taking `hidden_states[-1].mean(dim=1)`,
        # this corresponds to the hidden_size from the config.
        swin_output_features = self.swin.config.hidden_size
        heatmap_output_features = 64 * 4 * 4
        star_features_size = 6
        combined_features = swin_output_features + heatmap_output_features + star_features_size
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, image, heatmap, star_features):
        # Swin model typically outputs a BaseModelOutputWithPooling or similar
        # We need the last hidden state for classification
        outputs = self.swin(image, output_hidden_states=True) # Ensure output_hidden_states is True
        swin_output = outputs.hidden_states[-1].mean(dim=1) # Global average pooling over sequence length
        heatmap_output = self.heatmap_conv(heatmap)
        combined = torch.cat([swin_output, heatmap_output, star_features], dim=1)
        output = self.fc(combined)
        return output

# --- Helper Functions (copied from your dataset for consistency) ---
def _create_heatmap(image_size, star_data):
    star_positions = [star_data['centroid']['position']]
    if 'reference_stars' in star_data:
        star_positions.extend([star['position'] for star in star_data['reference_stars']])

    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for (x, y) in star_positions:
        # Ensure positions are within bounds for cv2.circle
        x = max(0, min(image_size[0] - 1, int(x)))
        y = max(0, min(image_size[1] - 1, int(y)))
        heatmap = cv2.circle(heatmap, (x, y), 5, 1, -1)
    return torch.tensor(heatmap).unsqueeze(0)

def _create_star_features(star_data):
    width, height = star_data['image_shape'][1], star_data['image_shape'][0]
    star_positions = [star_data['centroid']['position']]
    if 'reference_stars' in star_data:
        star_positions.extend([star['position'] for star in star_data['reference_stars']])

    normalized_positions = []
    for (x, y) in star_positions:
        # Normalize and ensure valid range [0, 1]
        normalized_positions.extend([max(0.0, min(1.0, x/width)), max(0.0, min(1.0, y/height))])

    while len(normalized_positions) < 6:
        normalized_positions.append(0.0)

    return torch.tensor(normalized_positions, dtype=torch.float32)

def _modify_filename_for_inference(filename):
    """
    Removes trailing zero if the number part ends in zero and has more than one digit.
    This is the same logic as in your StarDataset.
    """
    match = re.search(r'stars-(\d+)\b', filename)
    if match:
        number_str = match.group(1)
        if number_str.endswith('0') and len(number_str) > 1:
            modified_number_str = number_str.rstrip('0')
            modified_filename = filename.replace(f'stars-{number_str}', f'stars-{modified_number_str}')
            return modified_filename
    return filename

# --- Inference Function ---
def infer_image(model, image_path, star_map_path, transform, device, class_mapping):
    """
    Performs inference on a single image.

    Args:
        model (nn.Module): The trained model.
        image_path (str): Full path to the input image.
        star_map_path (str): Full path to the corresponding star map JSON file.
        transform (torchvision.transforms.Compose): Image transformation pipeline.
        device (torch.device): Device to run inference on (e.g., 'cuda' or 'cpu').
        class_mapping (dict): Dictionary mapping class indices to human-readable labels.

    Returns:
        tuple: (predicted_class_label, predicted_probability)
    """
    model.eval() # Set model to evaluation mode

    # 1. Load Image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Skipping.")
        return None, None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Skipping.")
        return None, None

    # 2. Load Star Map Data
    try:
        with open(star_map_path) as f:
            star_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Star map not found at {star_map_path}. Skipping.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {star_map_path}. Skipping.")
        return None, None
    except Exception as e:
        print(f"Error loading star map {star_map_path}: {e}. Skipping.")
        return None, None

    # 3. Create Heatmap and Star Features
    heatmap = _create_heatmap(image.size, star_data)
    star_features = _create_star_features(star_data)

    # 4. Apply Image Transforms
    if transform:
        image = transform(image)

    # 5. Prepare Tensors for Model Input (add batch dimension and move to device)
    image = image.unsqueeze(0).to(device)
    heatmap = heatmap.unsqueeze(0).to(device)
    star_features = star_features.unsqueeze(0).to(device)

    # 6. Make Prediction
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model(image, heatmap, star_features)
        probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
        predicted_prob, predicted_idx = torch.max(probabilities, 1)

    # 7. Post-process and return
    predicted_label = class_mapping.get(predicted_idx.item(), f"Unknown Class {predicted_idx.item()}")
    return predicted_label, predicted_prob.item()

# --- Main Inference Script Execution ---
if __name__ == "__main__":
    # --- Configuration (MUST MATCH TRAINING CONFIG) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 12 # From your training script

    transform = transforms.Compose([
        transforms.Resize((256, 256)), # Must match training input size for Swin
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Define your class mapping (adjust based on your actual labels and how you derive them)
    # This should be derived from your `_get_class_to_idx` logic in the dataset
    # For example, if your labels are 0, 1, ..., 11 and correspond to specific orientations:
    class_mapping = {
        0: "Orientation_0",
        1: "Orientation_1",
        2: "Orientation_2",
        3: "Orientation_3",
        4: "Orientation_4",
        5: "Orientation_5",
        6: "Orientation_6",
        7: "Orientation_7",
        8: "Orientation_8",
        9: "Orientation_9",
        10: "Orientation_10",
        11: "Orientation_11"
    }

    # --- Initialize Model and Load Checkpoint ---
    swin_base_model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=num_classes, # This num_labels is for the pre-trained head, which we ignore
        ignore_mismatched_sizes=True,
        output_hidden_states=True
    )
    model = SwinWithStarPositions(swin_base_model, num_classes=num_classes).to(device)

    checkpoint_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed_model/swin_epoch_45.pth" # Path to your saved model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded model from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit() # Exit if model cannot be loaded

    # --- Prepare Data for Inference ---
    # This example demonstrates inference for a few specific files.
    # In a real scenario, you might iterate through a test dataset.

    # Base directories for images and star maps (should match your actual data layout)
    base_image_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/images"
    base_star_map_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/star_maps"

    # Example files for inference
    # You need to provide the 'raw' image name from your annotation file before any modifications
    # The script will then apply the necessary modifications to find the actual file.
    inference_files = [
        "stars-179.png", # Example: Will become stars-1.png and stars-1_star_map.json
        "stars-232.png", # Example: No change to number part, just .jpg to .png
        "stars-372.png", # Example: Will become stars-1_star_map.json (if that's desired)
        "stars-574.png", # Example: Will become stars-5.png and stars-5_star_map.json
        # Add more filenames you want to test
    ]

    print("\n--- Running Inference ---")
    for original_img_name in inference_files:
        # Apply the same filename modifications as in the dataset loading
        #modified_img_name = _modify_filename_for_inference(original_img_name).replace(".jpg", ".png")
        #modified_star_map_base_name = _modify_filename_for_inference(os.path.splitext(original_img_name)[0])
        original_img=original_img_name.split(".")[0]
        actual_image_path = os.path.join(base_image_dir, original_img_name)
        actual_star_map_path = os.path.join(base_star_map_dir, f"{original_img}_star_map.json")
        print(actual_star_map_path)
        print(f"\nProcessing {original_img_name}:")
        print(f"  Looking for image: {actual_image_path}")
        print(f"  Looking for star map: {actual_star_map_path}")

        predicted_label, predicted_prob = infer_image(model, actual_image_path, actual_star_map_path, transform, device, class_mapping)

        if predicted_label is not None:
            print(f"  Predicted Class: {predicted_label}")
            print(f"  Predicted Probability: {predicted_prob:.4f}")
        else:
            print(f"  Failed to process {original_img_name}")
