import tensorflow as tf
import os


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
        # generator step
        with tf.GradientTape(persistent=True) as tape:
            fake_images = generator(bw_images)

            d_fake = discriminator(bw_images, fake_images)
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
