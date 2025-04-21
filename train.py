import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator
from utils.data_loader import prepare_dataset
from utils.losses import perceptual_loss, adversarial_loss, content_loss

# Load dataset
low_res_images, high_res_images = prepare_dataset('data/highRes', 'data/lowRes')

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
for epoch in range(100):
    for lr, hr in zip(low_res_images, high_res_images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate high-res image
            generated_hr = generator(tf.expand_dims(lr, axis=0), training=True)
            
            # Discriminator loss
            real_output = discriminator(tf.expand_dims(hr, axis=0), training=True)
            fake_output = discriminator(generated_hr, training=True)
            disc_loss = adversarial_loss(tf.ones_like(real_output), real_output) + \
                        adversarial_loss(tf.zeros_like(fake_output), fake_output)
            
            # Generator loss
            gen_loss = perceptual_loss(tf.expand_dims(hr, axis=0), generated_hr) + \
                       content_loss(tf.expand_dims(hr, axis=0), generated_hr)
            
        # Update weights
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")
