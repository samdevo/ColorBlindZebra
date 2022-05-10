import tensorflow as tf
import os
import tensorflow_io as tfio
from preprocessing import norm_imgs, denorm_imgs
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
        if i % 25 == 0:
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
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(color_images)
            # color_images[:,:,:,0] = color_images[:,:,:,0] / 50. - 1
            # color_images[:,:,:,1] = (color_images[:,:,:,1] + 86.185) / 184.439
            # color_images[:,:,:,2] = (color_images[:,:,:,2] + 107.863) / 202.345

           
            d_fake = discriminator(color_images[...,:1], fake_images)
            # print(fake_images.shape)
            # print(color_images[:,:,:,1:].shape)

            gen_loss, gen_mae_loss, gen_d_loss = generator.loss_function(fake_images, d_fake, color_images[:,:,:,1:])
            if i < 20: gen_d_loss = 0
            gen_loss = gen_mae_loss * 10000 + gen_d_loss * 5 * i

            # gen_loss = gen_mae_loss + 

            if i >= 20: 
                d_real = discriminator(color_images[...,:1], color_images[:,:,:,1:])
                d_loss = discriminator.loss_function(d_real, d_fake) 
            else: d_loss = 0

            # print("d_loss: " + str(d_loss))

        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        if i >= 20: disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        generator.optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        if i >= 20: discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        

        progbar.update(i, [("gen_mae_loss", gen_mae_loss), ("gen_d_loss", gen_d_loss), ("disc_loss", d_loss)])
