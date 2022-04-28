import gzip
import numpy as np
import pickle
from cv2 import cv2

#pip install opencv-python
#Link to dataset: https://www.cs.toronto.edu/~kriz/cifar.html

def get_data(inputs_file_path):
    """
Each of the batch files contains a dictionary with the following elements:
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
The first 1024 entries contain the red channel values, the next 1024 the green,
 and the final 1024 the blue. The image is stored in row-major order, so that 
the first 32 entries of the array are the red channel values of the first row of the image.
(NOT USING): labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates 
the label of the ith image in the array data.
    """
    
    with open(inputs_file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[0]

def make_BW_images(inputs_file_path):

    "Returning black and white version of colorized dataset -- a 10000x3072 numpy array of uint8s"

    colorized_images = get_data(inputs_file_path)
    bw_images = []
    for photo in colorized_images:
        bw_images.append(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    bw_images = np.array(bw_images)
    return bw_images
