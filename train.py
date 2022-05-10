import tensorflow as tf
import os
import tensorflow_io as tfio
from preprocessSam import norm_imgs, denorm_imgs
# from skimage.color import rgb2lab, lab2rgb


checkpoint_directory = "/tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

def train_epoch(generator, discriminator, dataset, new=True):
    # checkpoint = tf.train.Checkpoint(
    #     gen_optimizer=generator.optimizer,
    #     disc_optimizer=discriminator.optimizer,
    #     )
    # if not new:

    progbar = tf.keras.utils.Progbar(len(list(dataset)))

    for i, color_images in enumerate(dataset):
        color_images = norm_imgs(color_images)
        # print(color_images)
        if i % 10 == 0:
            l_with_ab = tf.concat([color_images[...,:1], generator(color_images)], axis=-1)
            # print(decoder_out[0,100:110,100:110,0])
            # print(decoder_out[0,100:110,100:110,1])
            # print(input[0,100:110,100:110,1])
            # print(input[0,100:110,100:110,2])
            # # print(image[100:110,100:110,0])
            # # print(image[100:110,100:110,1])
            # # print(image[100:110,100:110,2])
            # # print(l_with_ab.shape)
            
            pil_img = tf.keras.preprocessing.image.array_to_img(denorm_imgs(color_images[0]).numpy())
            pil_img.show()
            pil_img = tf.keras.preprocessing.image.array_to_img(denorm_imgs(l_with_ab[0]).numpy())
            pil_img.show()
        with tf.GradientTape(persistent=True) as tape:
            fake_images = generator(color_images)
            # color_images[:,:,:,0] = color_images[:,:,:,0] / 50. - 1
            # color_images[:,:,:,1] = (color_images[:,:,:,1] + 86.185) / 184.439
            # color_images[:,:,:,2] = (color_images[:,:,:,2] + 107.863) / 202.345

           
            d_fake = discriminator(color_images[:,:,:,:1], fake_images)
            # print(fake_images.shape)
            # print(color_images[:,:,:,1:].shape)

            gen_loss = generator.loss_function(fake_images, d_fake, color_images[:,:,:,1:])

            print(gen_loss)

            d_real = discriminator(color_images[...,:1], color_images[:,:,:,1:])

            d_loss = discriminator.loss_function(d_real, d_fake)

        gen_grads = tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = tape.gradient(d_loss, discriminator.trainable_variables)

        generator.optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        

        progbar.update(i, [("gen_loss", gen_loss), ("disc_loss", d_loss)])
