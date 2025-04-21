import os
import cv2
import numpy as np

def load_images(image_dir, img_size=(256, 256)):
    """Load and preprocess images from a directory."""
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)

def prepare_dataset(high_res_dir, low_res_dir):
    """Prepare highRes and lowRes image pairs."""
    high_res_images = load_images(high_res_dir)
    low_res_images = load_images(low_res_dir, img_size=(64, 64))  # Downscale to 64x64
    return low_res_images, high_res_images