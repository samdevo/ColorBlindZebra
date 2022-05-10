import tensorflow as tf
import os
import tensorflow_io as tfio
from preprocessing import norm_imgs, denorm_imgs
# from skimage.color import rgb2lab, lab2rgb


# checkpoint_directory = "/tmp/training_checkpoints"
# checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

def train_epoch(generator, discriminator, dataset, epoch):
    
    progbar = tf.keras.utils.Progbar(len(list(dataset)))

    for i, color_images in enumerate(dataset):
        # convert rgb images to lab (can't do this in preprocessing b/c can't do these operations to tensors
        color_images = norm_imgs(color_images)
        # see how training's going
        if i % 25 == 0: save_example(epoch, i, color_images, generator(color_images))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(color_images)
            d_fake = discriminator(color_images[...,:1], fake_images)
            gen_loss, gen_mae_loss, gen_d_loss = generator.loss_function(fake_images, d_fake, color_images[...,1:])
            
            # wait for generator to train a bit before we start training the discriminator
            if i >= 0: 
                d_real = discriminator(color_images[...,:1], color_images[...,1:])
                d_loss = discriminator.loss_function(d_real, d_fake) 
            else: d_loss = 0


        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        if i >= 0: disc_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

        generator.optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        if i >= 0: discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        
        progbar.update(i, [("gen_mae_loss", gen_mae_loss), ("gen_d_loss", gen_d_loss), ("disc_loss", d_loss)])


def save_example(epoch, i, color_images, generated):
    bw = tf.concat([color_images[...,:1], tf.zeros_like(color_images[...,1:])], axis=-1)
    l_with_ab = tf.concat([color_images[...,:1], generated], axis=-1)
    just_colors = tf.concat([tf.zeros_like(color_images[...,:1]), generated], axis=-1)
    real_colors = tf.concat([tf.zeros_like(color_images[...,:1]), color_images[...,1:]], axis=-1)
    pil_img = tf.keras.preprocessing.image.array_to_img(denorm_imgs(tf.concat([bw[0], color_images[0], l_with_ab[0], just_colors[0], real_colors[0]], axis=1)).numpy())
    pil_img.save(f"gen_imgs/epoch{str(epoch)}-batch{str(i)}.png")
