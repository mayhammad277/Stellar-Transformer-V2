import os
import torch
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import Swinv2ForImageClassification
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_model(checkpoint_path, num_classes=12, device='cuda'):
    """Load the trained model from checkpoint"""
    swin_model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_hidden_states=True
    )
    model = SwinWithStarPositions(swin_model, num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

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


def prepare_image_and_star_data(image_path, star_map_path, image_size=128):
    image = Image.open(image_path).convert("RGB")

    # Process star map *before* applying image transforms
    star_map_data = process_star_map(star_map_path, image.size)
    heatmap, star_features = star_map_data

    transform = transforms.Compose([ # Use torchvision transforms
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transformed_image = transform(image).unsqueeze(0).to(device)
    heatmap = heatmap.unsqueeze(0).to(device)  # Add batch dimension and move to device
    star_features = star_features.unsqueeze(0).to(device) # Add batch dimension and move to device

    return transformed_image, heatmap, star_features

def inference(model_path, image_path, star_map_path, num_classes=12, image_size=128):
    # Load the trained model
    swin_model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_hidden_states=True
    )
    model = SwinWithStarPositions(swin_model, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # Prepare the image and star data
    image, heatmap, star_features = prepare_image_and_star_data(image_path, star_map_path, image_size)

    # Perform inference
    with torch.no_grad():
        outputs = model(image, heatmap, star_features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage: Replace with your actual paths
    model_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/new_model30_augmented_modify/swin_epoch_43.pth"
    image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/images/N_N330/N330_0.png"
    star_map_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/starmaps/N_N330/N330_0_star_map.json"
    num_classes = 12
    image_size = 128

    predicted_class, confidence = inference(model_path, image_path, star_map_path, num_classes, image_size)

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

