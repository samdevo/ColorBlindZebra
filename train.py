import tensorflow as tf
import os
import tensorflow_io as tfio


checkpoint_directory = "/tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

def train_epoch(generator, discriminator, dataset, new=True):
    # checkpoint = tf.train.Checkpoint(
    #     gen_optimizer=generator.optimizer,
    #     disc_optimizer=discriminator.optimizer,
    #     )
    # if not new:

    progbar = tf.keras.utils.Progbar(len(list(dataset)))

    for i, (bw_images, color_images) in enumerate(dataset):
        # color_images = color_images.numpy()
        # # print(color_images[0])
        # # color_images[:,:,:,0] = color_images[:,:,:,0] / 50. - 1
        # # color_images[:,:,:,1] = (color_images[:,:,:,1] + 86.185) / 184.439
        # # color_images[:,:,:,2] = (color_images[:,:,:,2] + 107.863) / 202.345
        # print(color_images[:,:,:,0].min())
        # print(color_images[:,:,:,1].min())
        # print(color_images[:,:,:,2].min())
        # print(color_images[:,:,:,0].max())
        # print(color_images[:,:,:,1].max())
        # print(color_images[:,:,:,2].max())
        
        # print("____________________")
        # continue
        # l_with_ab = tf.concat([color_images[:,:,:,:1], generator(color_images)], axis=-1).numpy()
        # print(l_with_ab[0])
        # l_with_ab[:,:,:,0] = (l_with_ab[:,:,:,0] + 1) * 50
        # l_with_ab[:,:,:,1] = (l_with_ab[:,:,:,1] * 184.439) - 86.185
        # l_with_ab[:,:,:,2] = (l_with_ab[:,:,:,2] * 202.345) - 107.863
        # print(l_with_ab[0])
        # exit(0)
        # pil_img = tf.keras.preprocessing.image.array_to_img(tfio.experimental.color.lab_to_rgb(input[0]))
        # pil_img.show()
        # generator step
        with tf.GradientTape(persistent=True) as tape:
            fake_images = generator(color_images)
            # color_images[:,:,:,0] = color_images[:,:,:,0] / 50. - 1
            # color_images[:,:,:,1] = (color_images[:,:,:,1] + 86.185) / 184.439
            # color_images[:,:,:,2] = (color_images[:,:,:,2] + 107.863) / 202.345

           
            d_fake = discriminator(color_images[:,:,:,:1], fake_images)
            print(fake_images.shape)
            print(color_images[:,:,:,1:].shape)

            gen_loss = generator.loss_function(fake_images, d_fake, color_images[:,:,:,1:])

            print(gen_loss)

            d_real = discriminator(bw_images, color_images[:,:,:,1:])

            d_loss = discriminator.loss_function(d_real, d_fake)

        gen_grads = tape.gradient(gen_loss, generator.trainable_variables)
        disc_grads = tape.gradient(d_loss, discriminator.trainable_variables)

        generator.optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        

        progbar.update(i, [("gen_loss", gen_loss), ("disc_loss", d_loss)])
