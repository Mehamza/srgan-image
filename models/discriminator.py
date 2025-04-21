import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator():
    """Build the SRGAN discriminator model."""
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model