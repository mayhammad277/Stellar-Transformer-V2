import os
import torch
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import Swinv2ForImageClassification
import torch.nn as nn

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

def load_model(checkpoint_path, num_classes=72, device='cuda'):
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
        normalized_positions.extend([x/width, y/height])
    
    # Pad to 6 features (3 stars × 2 coordinates)
    while len(normalized_positions) < 6:
        normalized_positions.append(0.0)
    
    star_features = torch.tensor(normalized_positions, dtype=torch.float32)
    
    return heatmap, star_features

def predict(image_path, star_map_path, model, device='cuda'):
    """Run inference on a single image-star map pair"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process star map
    heatmap, star_features = process_star_map(star_map_path, image.size)
    heatmap = heatmap.unsqueeze(0).to(device)
    star_features = star_features.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor, heatmap, star_features)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs).item()
    
    # Convert prediction to angle (assuming classes represent 5° increments)
    predicted_angle = pred_class * 30  #pred_class * 5
    confidence = probs[0, pred_class].item()
    
    return predicted_angle, confidence

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/new_model30/swin_epoch_80.pth"  
    test_image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/images/N_N180/N180_3.png"
    test_star_map_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized_30/starmaps/N_N180/N180_2_star_map.json"
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, num_classes=12, device=device)
    
    # Run prediction
    print("Running inference...")
    angle, confidence = predict(test_image_path, test_star_map_path, model, device)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Angle: {angle}°")
    print(f"Confidence: {confidence:.2%}")
    print(f"Class: N_{angle}")

    # Optional: Print top-3 predictions
    with torch.no_grad():
        image = Image.open(test_image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        heatmap, star_features = process_star_map(test_star_map_path, image.size)
        heatmap = heatmap.unsqueeze(0).to(device)
        star_features = star_features.unsqueeze(0).to(device)
        
        outputs = model(image_tensor, heatmap, star_features)
        probs = torch.softmax(outputs, dim=1)
        
        top3_probs, top3_classes = torch.topk(probs, 3)
        
        print("\nTop 3 Predictions:")
        for i in range(3):
            print(f"{i+1}. Class N_{top3_classes[0,i].item()*5} ({top3_classes[0,i].item()}): {top3_probs[0,i].item():.2%}")
