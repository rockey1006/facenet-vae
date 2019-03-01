import os
import numpy as np
import tensorflow as tf
from .models.dfc_vae import Vae

IMAGE_MEAN = np.array([134.10714722, 102.52040863, 87.15436554])
IMAGE_STDDEV = np.sqrt(np.array([3941.30175781, 2856.94287109, 2519.35791016]))
LATENT_VAR_SIZE = 100

def normalize(images):
    return (images - IMAGE_MEAN) / IMAGE_STDDEV

def unnormalize(normalized):
    return normalized * IMAGE_STDDEV + IMAGE_MEAN

class VAE():
    """High-level wrapper for DFC-VAE"""
    def __init__(self, checkpoint_path):
        vae = Vae(100)
        image_size = vae.get_image_size()

        images = tf.placeholder(tf.float32, shape=(None, image_size,
                                                   image_size, 3),
                                name='input')

        # Normalize
        images_norm = normalize(images)
        images_norm_resize = tf.image.resize_images(images_norm,
                                                    (image_size, image_size))

        # Encoder
        mean, log_variance = vae.encoder(images_norm_resize, is_training=False)

        # Sampling
        epsilon = tf.random_normal((tf.shape(mean)[0], LATENT_VAR_SIZE))
        std = tf.exp(log_variance / 2)
        z = mean + epsilon * std

        # Decoder
        reconstructed_norm = vae.decoder(z, is_training=False)
        reconstructed = unnormalize(reconstructed_norm)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Restore weights from checkpoint
        # TODO: Remove print statement
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        print(f"Restoring VAE checkpoint: {checkpoint_path}")
        saver.restore(sess, checkpoint_path)

        # Keep references to important tensors
        self.vae = vae
        self.sess = sess
        self.input = images
        self.z = z
        self.reconstructed = reconstructed

    def get_code(self, images):
        """Encode images into codes"""
        return self.sess.run(self.z, feed_dict={self.input: images})

    def get_reconstruction_from_code(self, codes):
        """Decode codes into images"""
        return self.sess.run(self.reconstructed,
                             feed_dict={self.z: codes})

    def get_reconstruction_from_image(self, images):
        return self.sess.run(self.reconstructed,feed_dict={self.input: images})
