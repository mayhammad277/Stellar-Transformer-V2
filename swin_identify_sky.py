import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SwinForImageClassification, SwinConfig
import os
from PIL import Image
from tqdm import tqdm

import json
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
        if self.transform:
            img = self.transform(img)
        return img, label
save_dir = "/media/student/B076126976123098/my_data/SiT/model_swin/"
#!mkdir $save_dir
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=50):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Configuration
data_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder/"
batch_size = 1
num_classes = 72
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = CustomDataset(root_dir=data_dir, transform=transform)
test_dataset = CustomDataset(root_dir=data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Swin model and redefine classifier head
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-small-patch4-window7-224",
)
# Override the classifier for 2 classes
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training and testing
train_model(model, train_loader, optimizer, criterion, device)
test_model(model, test_loader, device)
