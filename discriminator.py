import tensorflow as tf

from generator import Encoder_Block

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.encoders = [
            Encoder_Block(64, 4, batchnorm=False, input_shape=(256,256,3)), # 128x128
            Encoder_Block(128, 4), #64x64
            Encoder_Block(256, 4), #32x32
            Encoder_Block(512, 4, strides=1), #32x32
        ]
        # 30x30 after 1 zero padding
        self.final_conv = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid')

        self.optimizer = tf.keras.optimizers.Adam()

    def call(self, input, target):
        encoded = tf.keras.layers.Concatenate()([input, target])

        for layer in self.encoders:
            encoded = layer(encoded)

        padded = tf.keras.layers.ZeroPadding2D()(encoded)

        final = self.final_conv(padded)

        return final

    def loss_function(self, d_real, d_fake):

        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(d_real), d_real)

        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(d_fake), d_fake)

        return tf.reduce_mean(real_loss + fake_loss)









    
        

