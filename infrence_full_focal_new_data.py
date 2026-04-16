import os
import torch
from PIL import Image
import json
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import Swinv2ForImageClassification

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from transformers import Swinv2ForImageClassification
import json
import cv2
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


def _create_heatmap_from_keypoints(image_size, keypoints):
    heatmap = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    for (x, y) in keypoints:
        x_clipped = int(round(np.clip(x, 0, image_size[0] - 1)))
        y_clipped = int(round(np.clip(y, 0, image_size[1] - 1)))
        if 0 <= x_clipped < image_size[0] and 0 <= y_clipped < image_size[1]:
            heatmap = cv2.circle(heatmap, (x_clipped, y_clipped), 5, 1, -1)
    return torch.tensor(heatmap).unsqueeze(0).to(device)

def _create_star_features_from_keypoints(image_size, keypoints):
    width, height = image_size
    normalized_positions = []
    for (x, y) in keypoints:
        x_clipped = np.clip(x, 0, width - 1)
        y_clipped = np.clip(y, 0, height - 1)
        normalized_positions.extend([x_clipped / width, y_clipped / height])

    while len(normalized_positions) < 6:
        normalized_positions.append(0.0)

    return torch.tensor(normalized_positions, dtype=torch.float32).to(device)

def prepare_image_and_star_data(image_path, star_map_path, image_size=128):
    image = Image.open(image_path).convert("RGB")
    with open(star_map_path) as f:
        star_data = json.load(f)

    star_positions_list = [star_data['centroid']['position']]
    if 'reference_stars' in star_data:
        star_positions_list.extend([star['position'] for star in star_data['reference_stars']])

    keypoints = []
    for x, y in star_positions_list:
        keypoints.append((x, y))

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy'))

    transformed = transform(image=np.array(image), keypoints=keypoints)
    transformed_image = transformed['image'].unsqueeze(0).to(device)
    transformed_keypoints = transformed['keypoints']

    heatmap = _create_heatmap_from_keypoints((image_size, image_size), transformed_keypoints)
    star_features = _create_star_features_from_keypoints((image_size, image_size), transformed_keypoints).unsqueeze(0).to(device)

    return transformed_image, heatmap, star_features

def inference(model_path, image_path, star_map_path, num_classes=12, image_size=128):
    # Load the trained model
    swin_model = Swinv2ForImageClassification.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        output_hidden_states=True
    )
    model = SwinWithStarPositions(swin_model, num_classes).to(device)
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
    image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/images/N_N270/N270_3.png"
    star_map_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/starmaps/N_N270/N270_3_star_map.json"
    num_classes = 12
    image_size = 128

    predicted_class, confidence = inference(model_path, image_path, star_map_path, num_classes, image_size)

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
