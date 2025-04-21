import cv2
import numpy as np
import tensorflow as tf
from models.generator import build_generator

def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    lr_image = cv2.resize(image, (64, 64))
    lr_image = lr_image.astype('float32') / 255.0
    lr_image = np.expand_dims(lr_image, axis=0)
    return lr_image, image

def save_image(image, output_path):
    """Save an image."""
    image = (image * 255).astype('uint8')
    cv2.imwrite(output_path, image)

def evaluate_model(generator, lr_image):
    """Generate a high-resolution image."""
    hr_image = generator.predict(lr_image)
    hr_image = np.squeeze(hr_image, axis=0)
    hr_image = np.clip(hr_image, 0, 1)
    return hr_image

def main():
    input_image_path = "data/lowRes/test_image.jpg"
    output_image_path = "output/highResImg.jpg"
    generator_weights_path = "models/generator_weights.h5"

    generator = build_generator()
    generator.load_weights(generator_weights_path)

    lr_image, original_image = load_image(input_image_path)
    hr_image = evaluate_model(generator, lr_image)
    save_image(hr_image, output_image_path)
    print(f"High-resolution image saved to {output_image_path}")

if __name__ == "__main__":
    main()
