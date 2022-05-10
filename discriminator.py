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
        self.final_conv = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=tf.random_normal_initializer(0., 0.02))

        self.optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    def call(self, input, target):
        encoded = tf.concat([input, target], axis=-1)
        # print(encoded.shape)
        # print("sending to layers...")
        # exit(0)

        for layer in self.encoders[:-1]:
            encoded = layer(encoded)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(encoded) 

        encoded = self.encoders[-1](zero_pad1)

        padded = tf.keras.layers.ZeroPadding2D()(encoded)

        final = self.final_conv(padded)

        return final

    def loss_function(self, d_real, d_fake):

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = loss_fn(tf.ones_like(d_real), d_real)

        fake_loss = loss_fn(tf.zeros_like(d_fake), d_fake)
        print("")
        print(f"real_loss: {str(tf.reduce_mean(real_loss).numpy())}")
        print(f"fake_loss: {str(tf.reduce_mean(fake_loss).numpy())}")

        return tf.reduce_mean(real_loss + fake_loss)









    
        

