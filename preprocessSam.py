import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

# a_channel = (a_channel + 86.185) / 184.439
# b_channel = (b_channel + 107.863) / 202.345


def prepare_image(file, rgb=False):
    image = tf.cast(file["image"], tf.float32)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)

    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    image = normalization_layer(image)

    if rgb:
        image_bw = tf.image.rgb_to_grayscale(image)
    else:
        image = tfio.experimental.color.rgb_to_lab(image)
    
    


    # print(image[100:110,100:110,0])
    # print(image[100:110,100:110,1])
    # print(image[100:110,100:110,2])

    
    # print(image[100:110,100:110,0])
    # print(image[100:110,100:110,1])
    # print(image[100:110,100:110,2])

    # exit(0)
    
        
        # print(image_bw.shape)
        # exit(0)
    image_bw = image[:,:,:1]
    return image_bw, image

     

def get_data(batch_size):
    # exit(0)
    # ds = tfds.load('imagenet_v2', split='test', shuffle_files=True)
    # mnist_builder = tfds.builder("stanford_dogs")
    mnist_builder = tfds.builder("cassava")
    mnist_builder.download_and_prepare()
    # print("here")
    dataset = mnist_builder.as_dataset()["train"]

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




