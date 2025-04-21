import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.losses import perceptual_loss, adversarial_loss, content_loss

class SRGAN:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def train_step(self, lr_images, hr_images):
        """Perform one training step for SRGAN."""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate high-resolution images
            generated_hr = self.generator(lr_images, training=True)

            # Discriminator loss
            real_output = self.discriminator(hr_images, training=True)
            fake_output = self.discriminator(generated_hr, training=True)
            disc_loss = adversarial_loss(tf.ones_like(real_output), real_output) + \
                        adversarial_loss(tf.zeros_like(fake_output), fake_output)

            # Generator loss
            gen_loss = perceptual_loss(hr_images, generated_hr) + \
                       content_loss(hr_images, generated_hr)

        # Update generator and discriminator weights
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def save_weights(self, generator_path, discriminator_path):
        """Save the weights of the generator and discriminator."""
        self.generator.save_weights(generator_path)
        self.discriminator.save_weights(discriminator_path)

    def load_weights(self, generator_path, discriminator_path):
        """Load the weights of the generator and discriminator."""
        self.generator.load_weights(generator_path)
        self.discriminator.load_weights(discriminator_path)