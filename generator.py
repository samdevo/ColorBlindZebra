from json import decoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        size = 128
        self.encoders = [
            Encoder_Block(64, 4, batchnorm=False, input_shape=(size,size,3)),
            Encoder_Block(128,4),
            Encoder_Block(256,4),
            Encoder_Block(512,4),
            Encoder_Block(512,4),
            Encoder_Block(512,4),
            #Encoder_Block(512,4),
            Encoder_Block(512,4), #1x512
        ]

        self.decoders = [
            #Decoder_Block(512,4),
            Decoder_Block(512,4),
            Decoder_Block(512,4),
            Decoder_Block(512,4),
            Decoder_Block(256,4),
            Decoder_Block(128,4),
            Decoder_Block(64,4),
            tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
        ]

    def call(self, input):
        encoder_outs = []
        for i,layer in enumerate(self.encoders):
            if i == 0:
                encoder_outs.append(layer(input))
            else:
                encoder_outs.append(layer(encoder_outs[-1]))
        decoder_out = None
        for i,layer in enumerate(self.decoders):
            if i == 0:
                decoder_out = layer(encoder_outs[-1])
            else:
                corresponding_encoded = encoder_outs[len(self.decoders) - i - 1]
                concat = tf.keras.layers.Concatenate()([decoder_out, corresponding_encoded])
                decoder_out = layer(concat)
        pil_img = tf.keras.preprocessing.image.array_to_img(input[0].numpy())
        pil_img.show()
        pil_img = tf.keras.preprocessing.image.array_to_img(decoder_out[0].numpy())
        pil_img.show()
        return decoder_out


class Encoder_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, strides=2, batchnorm=True, input_shape=None):
        super(Encoder_Block, self).__init__()
        if input_shape is not None: 
            self.conv_layer = tf.keras.layers.Conv2D(
                num_filters, 
                kernel_size, 
                strides=strides, 
                padding='same', 
                input_shape=input_shape
            )
        else:
            self.conv_layer = tf.keras.layers.Conv2D(
                num_filters, 
                kernel_size, 
                strides=strides, 
                padding='same'
            )
        self.batch_norm = batchnorm
        if batchnorm:
            self.batch_norm = tf.keras.layers.BatchNormalization()
        self.leaky = tf.keras.layers.LeakyReLU()
    
    def call(self, inputs):
        conv_out = self.conv_layer(inputs)
        if self.batch_norm:
            batch_norm_out = self.batch_norm(conv_out)
        else:
            batch_norm_out = conv_out
        leaky_out = self.leaky(batch_norm_out)

        return leaky_out


class Decoder_Block(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size=3, strides=2):
        super(Decoder_Block, self).__init__()

        self.conv_t_layer = tf.keras.layers.Conv2DTranspose(
            num_filters, 
            kernel_size, 
            strides=strides, 
            padding='same'
        )

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        conv_out = self.conv_t_layer(inputs)
        batch_norm_out = self.batch_norm(conv_out)
        relu_out = self.relu(batch_norm_out)

        return relu_out



