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
# Initializing a Swinv2 microsoft/swinv2-tiny-patch4-window8-256 style configuration
configuration = Swinv2Config()
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
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
        swin_output_features = self.swin.config.hidden_size#96   Hidden size of Swin Transformer
        heatmap_output_features = 64 * 4 * 4   # Output size of the heatmap branch
        combined_features = swin_output_features + heatmap_output_features
        
        # Define the fully connected layer
        self.fc = nn.Linear(combined_features, num_classes)

    def forward(self, image, heatmap):
        # Pass the image through the Swin Transformer
        #print( self.swin(image).hidden_states[-1])
        swin_output = self.swin(image).hidden_states[-1].mean(dim=1)  # Use pooled output
        #print("swin_output.shape",swin_output.shape)
        print(f"Swin output shape: {swin_output.shape}")  # Debugging
        
        # Pass the heatmap through the heatmap branch
        heatmap_output = self.heatmap_conv(heatmap)
        print(f"Heatmap output shape: {heatmap_output.shape}")  # Debugging
        
        # Concatenate the Swin output and heatmap features
        combined = torch.cat([swin_output, heatmap_output], dim=1)
        print(f"Combined shape: {combined.shape}")  # Debugging
        
        # Pass through a fully connected layer
        output = self.fc(combined)
        return output
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=80):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        for images, heatmaps, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, heatmaps)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Save loss history
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, heatmaps, labels in dataloader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            labels = labels.to(device)
            outputs = model(images, heatmaps)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
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
        
        # Create a heatmap for star positions
        heatmap = self.create_star_heatmap((256, 256), star_positions)
        
        if self.transform:
            img = self.transform(img)
        
        return img, heatmap, label

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
def create_star_heatmap( image_shape, star_positions, sigma=5):
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
        
def visualize_star_positions(image, star_positions):
    """
    Visualize star positions on the image.
    :param image: Input image (PIL or NumPy array).
    :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    for (x, y) in star_positions:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw green circles at star positions
    
    # Display the image
    cv2.imshow("Star Positions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    


def predict_single_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
        
        # Detect star positions
    star_positions = detect_star_positions(img)
        
        # Create a heatmap for star positions
    heatmap = create_star_heatmap((256, 256), star_positions)
        
      
    img = transform(img)

    
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    if not isinstance(heatmap, torch.Tensor):
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image = img.unsqueeze(0)
    heatmap = heatmap.unsqueeze(0)

    # Move to the same device as the model (GPU if available)
    device = next(model.parameters()).device
    image, heatmap = image.to(device), heatmap.to(device)

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model(image, heatmap)
        
    pred=torch.softmax(prediction, dim=1)
    predicted_class = torch.argmax(pred, dim=1).item()
    class_names = ["N"+str(i) for i in range(0,360,5)]
    predicted_label = class_names[predicted_class]
    print(f"Predicted class: {predicted_label}")
    

    return pred
    

# Example usage
#image_path = "/path/to/your/image.jpg"
#image = Image.open(image_path).convert("RGB")
#star_positions = detect_star_positions(image)
#visualize_star_positions(image, star_positions)
"""
data_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder"
batch_size = 8
num_classes = 72
test_dataset = CustomDataset(root_dir=data_dir, transform=transform)

save_dir = "/media/student/B076126976123098/my_data/SiT/model_swin3/"
#mkdir $save_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dataset
train_dataset = CustomDataset(root_dir=data_dir, transform=transform)
test_dataset = CustomDataset(root_dir=data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Swin model and redefine classifier head
swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=num_classes ,ignore_mismatched_sizes=True,output_hidden_states=True,output_attentions=True,return_dict=True)
model = SwinWithStarPositions(swin_model, num_classes=num_classes)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training and testing
train_model(model, train_loader, optimizer, criterion, device)
test_model(model, test_loader, device)
"""





swin_model = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=72 ,ignore_mismatched_sizes=True,output_hidden_states=True,output_attentions=True,return_dict=True)
model = SwinWithStarPositions(swin_model, num_classes=72)
    
model.load_state_dict(torch.load("/media/student/B076126976123098/my_data/SiT/model_swin3/swin_epoch_80.pth", weights_only=True,map_location=torch.device('cpu')))


pth="/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder/N60/N60_3.png"

model.to("cuda" if torch.cuda.is_available() else "cpu")


# Perform the prediction
prediction = predict_single_image(model,pth)
print("Predicted logits:", prediction)           

