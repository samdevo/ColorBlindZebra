import tensorflow as tf
import tensorflow_datasets as tfds
from rich.traceback import install

install()




def prepare_image(file):        
    print(file)
    image = file["image"]
    # Turns inputs in [0,255] to be in [-1,1]
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

    image = tf.image.resize_with_crop_or_pad(image, 128, 128)
    image = normalization_layer(image)
    image_bw = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

    return image_bw, image

     

def get_data(batch_size):
    # ds = tfds.load('imagenet_v2', split='test', shuffle_files=True)
    # mnist_builder = tfds.builder("stanford_dogs")
    mnist_builder = tfds.builder("stl10")
    mnist_info = mnist_builder.info
    mnist_builder.download_and_prepare()
    dataset = mnist_builder.as_dataset()["train"] # change to unlabelled for big dataset

    assert isinstance(dataset, tf.data.Dataset)

    dataset = dataset.shuffle(1024)

    dataset = dataset.map(prepare_image).batch(batch_size)

    return dataset

# ds = get_data(128)
# i = 0
# for batch_images, batch_labels in ds:
#     img = batch_images[0]
#     pil_img = tf.keras.preprocessing.image.array_to_img(img)
    # pil_img.show()
#     i += 1
#     if i > 10: break




