import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import tensorflow as tf
import matplotlib.pyplot as plt
from generator import Generator
from preprocess import get_data as get_data_sam


if __name__ == '__main__':
    gen_model = Generator()
    gen_model.compile(
        optimizer='adam',
        loss='MAE',
        metrics=['MAE'],
        run_eagerly=True
    )

    dataset = get_data_sam(64)
    gen_model.fit(dataset, epochs=1)

