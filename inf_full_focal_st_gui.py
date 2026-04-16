import os
import random
import json
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QScrollArea, QGridLayout, QTextEdit, QFileDialog
)
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "~/envs/rspi5/lib/python3.8/site-packages/PyQt5/Qt/plugins/platforms"

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from PIL import Image
import torch
from torchvision import transforms

# Your model and functions
from inferene_full_foxal_steleruim import (
    SwinWithStarPositions,
    infer_image
)
from transformers import Swinv2ForImageClassification

BASE_IMAGE_DIR = "/home/bora3i/may_data/new_all_star_processed/images"
BASE_STAR_MAP_DIR = "/home/bora3i/may_data/new_all_star_processed/star_maps"
CHECKPOINT_PATH = "/home/bora3i/may_data/new_all_star_processed_model/swin_epoch_45.pth"
NUM_CLASSES = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_mapping = {i: f"Orientation_{i}" for i in range(NUM_CLASSES)}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load Model
swin_base = Swinv2ForImageClassification.from_pretrained(
    "microsoft/swinv2-tiny-patch4-window8-256",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    output_hidden_states=True
)
model = SwinWithStarPositions(swin_base, NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class StarTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛰️ Star Tracker - Fancy GUI")
        self.resize(1000, 600)

        self.image_list = []
        self.current_image = None

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Buttons
        self.start_btn = QPushButton("🚀 Start (Capture 20 Images)")
        self.align_btn = QPushButton("🧭 Align (Run Inference)")
        self.start_btn.clicked.connect(self.capture_images)
        self.align_btn.clicked.connect(self.run_inference)

        # Thumbnails Grid in ScrollArea
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_widget.setLayout(self.grid_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.grid_widget)

        # Large Image Display
        self.image_label = QLabel("Click a thumbnail to view")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Results
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setStyleSheet("font-family: Consolas; font-size: 14px;")

        # Assemble layouts
        left_layout.addWidget(self.start_btn)
        left_layout.addWidget(self.align_btn)
        left_layout.addWidget(self.scroll_area)

        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.result_box)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

    def capture_images(self):
        self.image_list = random.sample(os.listdir(BASE_IMAGE_DIR), 20)
        self.result_box.setText("20 images captured.")
        self.populate_thumbnails()

    def populate_thumbnails(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for i, img_name in enumerate(self.image_list):
            path = os.path.join(BASE_IMAGE_DIR, img_name)
            thumb_label = QLabel()
            thumb_label.setFixedSize(100, 100)
            thumb_label.setScaledContents(True)
            pixmap = QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb_label.setPixmap(pixmap)
            thumb_label.mousePressEvent = lambda e, p=path: self.show_large_image(p)
            self.grid_layout.addWidget(thumb_label, i // 5, i % 5)

    def show_large_image(self, img_path):
        image = QPixmap(img_path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(image)
        self.current_image = img_path

    def run_inference(self):
        if not self.image_list:
            self.result_box.setText("Please press Start first.")
            return

        results = []
        for img_name in self.image_list:
            img_base = os.path.splitext(img_name)[0]
            img_path = os.path.join(BASE_IMAGE_DIR, img_name)
            star_map_path = os.path.join(BASE_STAR_MAP_DIR, f"{img_base}_star_map.json")

            label, prob = infer_image(model, img_path, star_map_path, transform, DEVICE, class_mapping)
            try:
                with open(star_map_path) as f:
                    star_data = json.load(f)
                ra = star_data.get("RA", "N/A")
                dec = star_data.get("Declination", "N/A")
            except:
                ra, dec = "Unknown", "Unknown"
            results.append(f"{img_name}\n → Label: {label}\n → RA: {ra}, DEC: {dec}\n")

        self.result_box.setText("\n\n".join(results[:5]) + "\n...")

# --- Launch App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StarTrackerApp()
    window.show()
    sys.exit(app.exec_())

