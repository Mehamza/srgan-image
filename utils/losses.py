import tensorflow as tf

def perceptual_loss(y_true, y_pred):
    """Perceptual loss using VGG19 feature maps."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    vgg.trainable = False
    loss_model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))

def adversarial_loss(y_true, y_pred):
    """Adversarial loss for the generator."""
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred), logits=y_pred))

def content_loss(y_true, y_pred):
    """Content loss (L1 or L2)."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))