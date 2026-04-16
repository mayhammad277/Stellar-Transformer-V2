
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SwinForImageClassification
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report

class CustomAnnotationDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.annotation_df = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = self._get_class_to_idx()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.annotation_image_names = self.annotation_df['image_name'].tolist()
        self.labels = self.annotation_df['orientation_class'].tolist()
        self.all_image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def _get_class_to_idx(self):
        classes = sorted(self.annotation_df['orientation_class'].unique())
        return {str(cls): idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.annotation_df)

    def _find_matching_image(self, annotation_img_name):
        # Strip "img_" and replace with "stars_"
        potential_base_name = annotation_img_name.replace("img_", "stars-").replace(".jpg", "")
        print("potential_base_name",potential_base_name)
        # Look for files that start with the potential base name and are image files
        matching_files = [f for f in self.all_image_files if f.startswith(potential_base_name) and not f.endswith(".json")]

        if matching_files:
            # If multiple matches (e.g., different augmentations), you might need a more specific rule.
            # For now, we'll return the first one found. You might need to refine this.
            return matching_files[0]
        return None

    def __getitem__(self, idx):
        annotation_img_name = self.annotation_image_names[idx]
        label = self.labels[idx]
        matching_img_filename = self._find_matching_image(annotation_img_name)

        if matching_img_filename:
            img_path = os.path.join(self.img_dir, matching_img_filename)
            try:
                img = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"Error: Matching image not found at {img_path}")
                return None, -1
        else:
            print(f"Warning: No matching image found for {annotation_img_name}")
            return None, -1

        label_str = str(label)
        if label_str in self.class_to_idx:
            label_idx = self.class_to_idx[label_str]
        else:
            raise ValueError(f"Class '{label_str}' not found in class mapping.")

        if self.transform:
            img = self.transform(img)
        return img, label_idx
def train_model(model, dataloader, optimizer, criterion, device, num_epochs=10, save_dir=None):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if images is None:
                continue  # Skip if there was an error loading the image
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

        if save_dir:
            checkpoint_path = os.path.join(save_dir, f"swin_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                json.dump(loss_history, f)

def test_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (images, labels) in dataloader:
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
def predict_single_image(model, image_path, transform, device, class_names):
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor).logits
        probs = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_idx].item()
        predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name, confidence
# --- Usage in your training script (adjust paths accordingly) ---
annotation_file_path = "/home/student/star_tracker"  # Replace
image_directory = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/transformed_images"      # Replace
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed_model"
learning_rate=1e-5
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomAnnotationDataset(
    annotation_file=annotation_file_path,
    img_dir=image_directory,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)



# Determine the number of classes
num_classes = len(train_dataset.class_to_idx)
class_names = list(train_dataset.class_to_idx.keys())
print(f"Number of classes: {num_classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")
print(f"Class names: {class_names}")

# Load Swin model
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", ignore_mismatched_sizes=True)
model.classifier = nn.Linear(model.config.hidden_size, num_classes)
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
print("\n--- Training the model ---")
train_model(model, train_loader, optimizer, criterion, device, num_epochs=80, save_dir=save_dir)

# --- Evaluation (Optional - if you have a separate test annotation file and image directory) ---
# test_annotation_file_path = "/path/to/your/test_annotation.csv"
# test_image_directory = "/path/to/your/test_images"
#
# test_dataset = CustomAnnotationDataset(
#     annotation_file=test_annotation_file_path,
#     img_dir=test_image_directory,
#     transform=transform
# )
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#
# print("\n--- Evaluating the model ---")
# test_model(model, test_loader, device, class_names)

# --- Prediction on a single image (Example) ---
def predict_single_image(model, image_path, transform, device, class_names):
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor).logits
        probs = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class_idx].item()
        predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name, confidence

if __name__ == "__main__":
    # Example prediction
    sample_image_path = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/processed_images/stars-002.png" # Replace with an actual image path
    if os.path.exists(sample_image_path):
        loaded_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", ignore_mismatched_sizes=True)
        loaded_model.classifier = nn.Linear(loaded_model.config.hidden_size, num_classes)
        loaded_model.load_state_dict(torch.load(os.path.join(save_dir, f"swin_epoch_{num_epochs}.pth"), map_location=device))
        loaded_model.eval().to(device)

        prediction, confidence = predict_single_image(loaded_model, sample_image_path, transform, device, class_names)
        if prediction:
            print(f"\n--- Single Image Prediction ---")
            print(f"Predicted Class: {prediction}, Confidence: {confidence:.4f}")
    else:
        print(f"\n--- Skipping single image prediction as the sample image path is not valid ---")
