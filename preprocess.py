import gzip
import numpy as np
import pickle

#Link to dataset: https://www.cs.toronto.edu/~kriz/cifar.html

# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. 
# The first 1024 entries contain the red channel values, the next 1024 the green,
#  and the final 1024 the blue. The image is stored in row-major order, so that 
# the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates 
# the label of the ith image in the array data.

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(inputs_file_path, num_examples):
    """
    Takes in an inputs file path unzips both files, 
    normalizes the inputs, and returns NumPy array of inputs. 
    
    Read the data of the file into a buffer and use 
    np.frombuffer to turn the data into a NumPy array. Keep in mind that 
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data. 
    
    
    :param inputs_file_path: file path for inputs, e.g. 'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer.
    :return: NumPy array of inputs (float32)
    """
    
    # TODO: Load inputs and labels
    # TODO: Normalize inputs

    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestreamInputs:
        bytestreamInputs.read(16)
        inputsArray = np.frombuffer(bytestreamInputs.read(), dtype = np.uint8)
        np.reshape(inputsArray, (num_examples, 32*32))
        inputsArray = inputsArray/255
        # f.close()
    return np.array(inputsArray, dtype= np.float32)