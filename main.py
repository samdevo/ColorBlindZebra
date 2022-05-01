import tensorflow as tf

import matplotlib.pyplot as plt

from generator import Generator
from preprocess import get_data, make_BW_images


if __name__ == '__main__':
    print("helluh")
    gen_model = Generator()
    gen_model.compile(
        optimizer='adam',
        loss='MAE',
        metrics=['MAE']
    )

    images = get_data('cifar-10-python.tar.gz')
    images_BW = make_BW_images('cifar-10-python.tar.gz')
    
    # x = tf.ones((1,256,256,3))
    # gen_model(x)
    # gen_model.summary()
    gen_model.fit(images_BW, images, epochs=1)

    tf.keras.utils.plot_model(gen_model, show_shapes=True, dpi=64)
