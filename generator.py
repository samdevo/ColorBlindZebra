from json import decoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        conv2d = tf.keras.layers.Conv2D
        batch = tf.keras.layers.BatchNormalization
        lrelu = tf.keras.layers.LeakyReLU

        self.encoders = tf.keras.Sequential([
            conv2d(filters=64, kernel_size=4, strides=2, padding='same'),
            batch(),
            conv2d(filters=128, kernel_size=4, strides=2, padding='same'),
            batch(), lrelu(),
            conv2d(filters=256, kernel_size=4, strides=2, padding='same'),
            batch(), lrelu(),
            conv2d(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), lrelu(),
            conv2d(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), lrelu(),
            conv2d(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), lrelu(),
            conv2d(filters=512, kernel_size=4, strides=2, padding='same'), #1x512
            batch(), lrelu(),
        ])

        conv2dtranspose = tf.keras.layers.Conv2DTranspose
        relu = tf.keras.layers.ReLU

        self.decoders = tf.keras.Sequential([
            conv2dtranspose(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=512, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=256, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=128, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=64, kernel_size=4, strides=2, padding='same'),
            batch(), relu(),
            conv2dtranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='tanh')
        ])

    def call(self, input):
        encoded = self.encoders(input)
        return self.decoders(encoded)

