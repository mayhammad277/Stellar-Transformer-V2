# Stellar-Transformer-V2
An advanced celestial attitude determination system using Swinv2 Transformers and Focal Loss. Includes automated star-map extraction from Stellarium, K-Means coordinate clustering, and a PyQt5 real-time orientation dashboard.





# 🌠 Stellar Transformer: AI-Driven Celestial Orientation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SwinV2](https://img.shields.io/badge/Model-SwinV2--Transformer-orange)](https://huggingface.co/docs/transformers/model_doc/swinv2)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

**Stellar Transformer** is a high-precision star tracking system designed to determine satellite attitude by analyzing celestial imagery. By combining **SwinV2 Transformer** backbones with custom **Star Heatmap branches**, this system achieves robust orientation classification even in sparse or noisy star fields.

---

## 📺 Dashboard Preview
<div align="center">
  <video src="PASTE_YOUR_GITHUB_VIDEO_LINK_HERE" width="100%" controls muted>
    Your browser does not support the video tag.
  </video>
  <p><i>Live inference: Predicting celestial orientation classes with real-time RA/Dec telemetry.</i></p>
</div>

---

## ✨ Key Features
* **Automated Data Gen:** `extract_stelruim.py` uses PyAutoGUI to capture high-res synthetic star maps directly from Stellarium.
* **Intelligent Labeling:** `cluster_ra_dec.py` uses K-Means clustering in 3D Cartesian space to group celestial coordinates into balanced orientation classes.
* **Hybrid Architecture:** A custom `SwinWithStarPositions` module that fuses global image features with localized star-centroid heatmaps.
* **Imbalance Handling:** Implementation of **Focal Loss** to prioritize learning from difficult or sparse star configurations.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/Stellar-Transformer-V2.git](https://github.com/yourusername/Stellar-Transformer-V2.git)
cd Stellar-Transformer-V2

# Set up the environment (using the provided yaml)
conda env create -f poet_2_env.yml
conda activate torch21

# Install additional dependencies
pip install transformers opencv-python PyQt5 pyautogui scikit-learn
```
🧠 Technical Architecture

- Coordinate Clustering. Instead of simple regression, we treat orientation as a classification problem. Celestial coordinates (RA/Dec) are mapped to 3D unit vectors and clustered to ensure maximum separation between states:$$x = \cos(Dec) \cos(RA), \quad y = \cos(Dec) \sin(RA), \quad z = \sin(Dec)$$
- Hybrid Feature Fusion:
The model doesn't just "see" an image; it processes a secondary channel of star centroids via a convolutional heatmap branch.

```Python

# Feature Fusion Logic from inf_full_focal_new_data3.py
swin_output_features = self.swin.config.hidden_size # 768 for tiny
heatmap_output_features = 64 * 4 * 4               # Convolved heatmap
star_features_size = 6                             # Metadata
combined_features = swin_output_features + heatmap_output_features + star_features_size

```


🚀 Training Deep Dive: Stellar Transformer V2. The training pipeline for this project is designed to handle the high precision required for celestial attitude determination. It moves beyond simple image classification by integrating spatial star-centroid data and addressing the inherent class imbalance of the night sky.

1. Data Synthesis & Labeling :
- Stellarium Extraction: We utilize extract_stelruim.py to capture high-fidelity synthetic star maps. This ensures the model is trained on a "perfect" ground truth before being adapted to noisy real-world sensors.
- Spherical K-Means Clustering: Because RA (0-360°) and Dec (-90° to +90°) are spherical coordinates, standard clustering fails at the poles. cluster_ra_dec.py converts these into 3D Cartesian vectors ($x, y, z$) before applying K-Means to ensure mathematically accurate orientation grouping.

2. Hybrid Architecture (Swin-Fusion):
The model, defined in train_full_focal_steleruim.py, uses a multi-stream fusion approach:

- Visual Backbone: A SwinV2-Tiny transformer extracts hierarchical features from the 256x256 star map.

- Heatmap Stream: A parallel Convolutional Neural Network (CNN) processes star-centroid heatmaps. This forces the model to attend to the specific geometry of star constellations rather than just global lighting.

- Metadata Injection: Six dimensions of numerical star metadata (centroids/magnitudes) are concatenated into the final dense layer.

3. Overcoming Class Imbalance:
The distribution of stars is not uniform across the sky (e.g., the Galactic Plane is much denser than the Galactic Poles). To prevent the model from becoming biased toward "busy" star fields,
we calculate Balanced Class Weights:

```Python

# From train_full_focal_steleruim.py
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.labels),
    y=train_dataset.labels
)
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

```

4. Execution:
```Python

python train_full_focal_steleruim.py

```

- Checkpoints: Saved automatically every epoch.

- Mixed Precision: Utilizes torch.cuda.amp (GradScaler) to reduce VRAM usage and accelerate training on NVIDIA GPUs.


Visualisations :




<img width="516" height="290" alt="Screenshot from 2026-04-16 04-30-36" src="https://github.com/user-attachments/assets/d54d1000-d164-4126-a511-ef26e6e7124b" />







<img width="513" height="289" alt="Screenshot from 2026-04-16 04-31-25" src="https://github.com/user-attachments/assets/b74e757f-99c9-4e8e-87b9-a1ccb8dd57d9" />









<img width="514" height="295" alt="Screenshot from 2026-04-16 04-31-45" src="https://github.com/user-attachments/assets/ff80994b-ba17-4fe2-81d7-1e0f0d16ad0a" />














⚖️ License
Distributed under the MIT License. See LICENSE for more information.
