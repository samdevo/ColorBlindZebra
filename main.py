import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

import matplotlib.pyplot as plt

from generator import Generator
from preprocess import get_data, make_BW_images
from preprocessSam import get_data as get_data_sam
import datetime


if __name__ == '__main__':
    print("helluh")
    gen_model = Generator()
    gen_model.compile(
        optimizer='adam',
        loss='MAE',
        metrics=['MAE'],
        run_eagerly=True
    )

    # images = get_data('cifar-10-python.tar.gz')
    # images_BW = make_BW_images('cifar-10-python.tar.gz')
    dataset = get_data_sam(64)
    # bw = tf.image.rgb_to_grayscale(images)

    # i = 0
    # print(dataset)
    # for batch_images, batch_bw in dataset:
    #     print(batch_images)
    #     img = batch_images[0]
    #     bw_img = batch_bw[0]
    #     pil_img = tf.keras.preprocessing.image.array_to_img(img)
    #     pil_img.show()
    #     pil_img = tf.keras.preprocessing.image.array_to_img(bw_img)
    #     pil_img.show()
    #     i += 1
    #     if i > 3: break
    
    # x = tf.ones((1,256,256,3))
    # gen_model(x)
    # gen_model.summary()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True)
    gen_model.fit(dataset, epochs=1,callbacks=[tensorboard_callback])

    tf.keras.utils.plot_model(gen_model, show_shapes=True, dpi=64)
