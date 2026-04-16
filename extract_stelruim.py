import os
import time
import pyautogui
import subprocess
from PIL import Image
from datetime import datetime

# Configuration
SAVE_DIR = os.path.expanduser("/media/student/B076126976123098/my_data/SiT/dataset_sky/stellarium_extracted_images")
STELLARIUM_TITLE = "Stellarium"  # Adjust if your window has a different name (check using wmctrl -l)
NUM_CAPTURES = 10
CAPTURE_INTERVAL = 2  # seconds

def ensure_save_directory():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created save directory: {SAVE_DIR}")

def activate_stellarium_window(window_title="Stellarium"):
    try:
        print(f"Activating window with title: {window_title}")
        subprocess.run(["wmctrl", "-a", window_title], check=True)
        time.sleep(2)  # Give time for the window to activate
        print("Window activated.")
    except subprocess.CalledProcessError:
        print(f"[ERROR] Could not activate Stellarium window. Is it open?")
        exit(1)

def setup_stellarium_for_capture():
    activate_stellarium_window(STELLARIUM_TITLE)
    print("Configuring Stellarium interface...")
    pyautogui.press('f11')  # Fullscreen
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'shift', 'i')  # Hide interface
    time.sleep(1)
    pyautogui.hotkey('ctrl', 'm')  # Turn off menu bars
    time.sleep(1)

def capture_stellarium_images(num_images=10, interval=2):
    for i in range(num_images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"stellarium_{timestamp}.png")
        print(f"Capturing image {i+1}/{num_images} -> {filename}")
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        time.sleep(interval)

def main():
    ensure_save_directory()
    setup_stellarium_for_capture()
    print("Starting image capture...")
    capture_stellarium_images(NUM_CAPTURES, CAPTURE_INTERVAL)
    print("Done.")

if __name__ == "__main__":
    main()

