import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

def detect_star_positions(image):
    """
    Detect star positions in an image using color thresholding and blob detection.
    :param image: Input image (PIL or NumPy array).
    :return: List of star positions [(x1, y1), (x2, y2), ...].
    """
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
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            star_positions.append((cx, cy))
    
    return star_positions

def create_star_heatmap(image_shape, star_positions, sigma=5):
    """
    Create a heatmap for star positions.
    :param image_shape: Shape of the image (height, width).
    :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
    :param sigma: Standard deviation for Gaussian blobs.
    :return: Heatmap with Gaussian blobs at star positions.
    """
    heatmap = np.zeros(image_shape, dtype=np.float32)
    for (x, y) in star_positions:
        heatmap[int(y), int(x)] = 1.0  # Mark star positions
    heatmap = gaussian_filter(heatmap, sigma=sigma)  # Apply Gaussian blur
    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return heatmap

def encode_star_positions(star_positions, image_size):
    """
    Encode star positions as a normalized feature vector.
    :param star_positions: List of star positions [(x1, y1), (x2, y2), ...].
    :param image_size: Size of the image (width, height).
    :return: Normalized feature vector of star positions.
    """
    width, height = image_size
    normalized_positions = []
    for (x, y) in star_positions:
        normalized_x = x / width
        normalized_y = y / height
        normalized_positions.extend([normalized_x, normalized_y])
    
    # Pad with zeros if fewer than 3 stars are detected
    while len(normalized_positions) < 6:  # 3 stars * 2 coordinates each
        normalized_positions.append(0.0)
    
    return torch.tensor(normalized_positions, dtype=torch.float32)

def process_single_image(image_path, output_dir):
    """
    Process a single image to generate heatmap and star features.
    :param image_path: Path to the input image.
    :param output_dir: Directory to save the heatmap and star features.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_size = image.size  # (width, height)
    
    # Detect star positions
    star_positions = detect_star_positions(image)
    print(f"Detected star positions: {star_positions}")
    
    # Generate heatmap
    heatmap = create_star_heatmap((256, 256), star_positions, sigma=5)
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Encode star features
    star_features = encode_star_positions(star_positions, image_size)
    print(f"Star features: {star_features}")
    
    # Save heatmap and star features
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "heatmap.pt")
    star_features_path = os.path.join(output_dir, "star_features.pt")
    
    torch.save(heatmap, heatmap_path)
    torch.save(star_features, star_features_path)
    print(f"Heatmap saved to: {heatmap_path}")
    print(f"Star features saved to: {star_features_path}")

# Example usage
if __name__ == "__main__":
    # Path to the input image
    image_path = "/path/to/your/test_image.jpg"
    
    # Directory to save the heatmap and star features
    output_dir = "/path/to/save/output"
    
    # Process the single image
    process_single_image(image_path, output_dir)
