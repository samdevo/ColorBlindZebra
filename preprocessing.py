import tensorflow as tf
import tensorflow_datasets as tfds
from skimage.color import rgb2lab, lab2rgb

# converts -1 to 1 LAB-space image to RGB
def denorm_imgs(images):
    images = images.numpy()
    images[...,0] = (images[...,0] + 1) * 50
    images[...,1:] = images[...,1:] * 128
    images = lab2rgb(images)
    return tf.convert_to_tensor(images)

# converts 0-1 rgb image to -1 to 1 LAB-space image
def norm_imgs(images):
    lab = rgb2lab(images.numpy())
    lab[...,0] = lab[...,0] / 50. - 1
    lab[...,1:] = lab[...,1:] / 128
    return tf.convert_to_tensor(lab)

# crops image to dimensions and rescales to 0-1
def prepare_image(file):
    image = tf.cast(file["image"], tf.float32)
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    image = normalization_layer(image)
    return image

     
# prepares and batches dataset
def get_data():
    
    mnist_builder = tfds.builder("tf_flowers")
    mnist_builder.download_and_prepare()
    dataset = mnist_builder.as_dataset()["train"]
    
    assert isinstance(dataset, tf.data.Dataset)

    dataset = dataset.map(prepare_image)

    return dataset

