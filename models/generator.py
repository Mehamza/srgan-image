import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    """Build the SRGAN generator model."""
    model = tf.keras.Sequential([
        # Initial convolutional layer
        layers.Conv2D(64, kernel_size=9, strides=1, padding='same', activation='relu'),
        
        # Residual blocks
        *[ResidualBlock(64) for _ in range(16)],
        
        # Upsampling blocks
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.UpSampling2D(size=2),
        
        # Final convolutional layer
        layers.Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')
    ])
    return model

class ResidualBlock(layers.Layer):
    """Residual block for the generator."""
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return inputs + x