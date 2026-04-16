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
