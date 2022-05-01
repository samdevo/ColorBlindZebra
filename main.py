import tensorflow as tf

import matplotlib.pyplot as plt

import generator

if __name__ == '__main__':
    print("helluh")
    gen_model = generator.Generator()
    tf.keras.utils.plot_model(gen_model, show_shapes=True, dpi=64)
