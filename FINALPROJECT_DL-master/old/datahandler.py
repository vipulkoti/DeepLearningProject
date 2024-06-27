# standard imports
import tensorflow as tf
import os, re, tqdm, pathlib, einops
import numpy as np
import matplotlib.pyplot as plt
import string

# custom parts
from caption_encoder import CaptionEmbedder


def load_8k_data(plot=False):
    image_dir_path = "./archive/Flickr8k_Dataset/"
    caption_file_path = "./archive/Flickr8k_text/Flickr8k.token.txt"
    train_img_file_path = "./archive/Flickr8k_text/Flickr_8k.trainImages.txt"
    test_img_file_path = "./archive/Flickr8k_text/Flickr_8k.testImages.txt"

    images = os.listdir(image_dir_path)
    captions_text = open(caption_file_path, "r").read()

    # print("Total Images in Dataset - {}\n".format(len(images)))
    # print("Sample Caption Details - \n{}\n".format(captions_text[:696]))

    captions_dict = dict()
    for line in captions_text.split("\n"):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split("#")[0]
            image_desc = " ".join(tokens[1:])
            if image_id not in captions_dict:
                captions_dict[image_id] = list()
            captions_dict[image_id].append(image_desc)

    train_images = (
        pathlib.Path(train_img_file_path).read_text().splitlines()
    )  # saves the image names into a list

    train_captions = [
        ((image_dir_path + file_name), captions_dict[file_name])
        for file_name in train_images
    ]  # saves the image path and the captions for that image into a list

    test_images = (
        pathlib.Path(test_img_file_path).read_text().splitlines()
    )  # saves the image names into a list

    test_captions = [
        ((image_dir_path + file_name), captions_dict[file_name])
        for file_name in test_images
    ]  # saves the image path and the captions for that image into a list

    train_dataset = tf.data.experimental.from_list(
        train_captions
    )  # saves the image path and the captions for that image into a list
    test_dataset = tf.data.experimental.from_list(
        test_captions
    )  # saves the image path and the captions for that image into a list

    for img_path, captions in train_dataset.take(1):
        numpy_images = img_path.numpy()
        numpy_labels = captions.numpy()

    if plot:
        pic = numpy_images.decode()
        x = plt.imread(pic)
        plt.imshow(x)
        plt.show()
        print(numpy_labels)

    return train_dataset, test_dataset


def load_mobilenet_model():
    IMAGE_SHAPE = (224, 224, 3)
    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True
    )
    mobilenet.trainable = False
    # tack on the image shape to the model for convenience
    mobilenet.image_shape = IMAGE_SHAPE

    return mobilenet


def imageLoadResize(img_path, image_model):
    IMAGE_SHAPE = image_model.image_shape
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    return img


def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f"[{re.escape(string.punctuation)}]", "")
    s = tf.strings.join(["[START]", s, "[END]"], separator=" ")
    return s


def create_tokenizer(vocabulary_size=5000, train_dataset=None, test_tokenizer=False):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size, standardize=standardize, ragged=True
    )
    tokenizer.adapt(train_dataset.map(lambda fp, txt: txt).unbatch().batch(1024))

    word_to_index = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary()
    )

    index_to_word = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
    )

    if test_tokenizer:
        t = tokenizer([["a cat is sitting on a dog"], ["a robot is dancing"]])
        w = index_to_word(t)
        print(tf.strings.reduce_join(w, separator=" ", axis=-1).numpy())

    return tokenizer


def match_shapes(images, captions):
    caption_shape = einops.parse_shape(captions, "b c")
    captions = einops.rearrange(captions, "b c -> (b c)")
    images = einops.repeat(images, "b ... -> (b c) ...", c=caption_shape["c"])
    return images, captions


if __name__ == "__main__":
    image_dir_path = "../archive/Flickr8k_Dataset/"
    caption_file_path = "../archive/Flickr8k_text/Flickr8k.token.txt"
    train_img_file_path = "../archive/Flickr8k_text/Flickr_8k.trainImages.txt"
    test_img_file_path = "../archive/Flickr8k_text/Flickr_8k.testImages.txt"
    # test data loading
    train_dataset, test_dataset = load_8k_data(plot=False)

    print(train_dataset)
    print(test_dataset)

    # print the first object in both datasets
    for img_path, captions in train_dataset.take(1):
        numpy_images = img_path.numpy()
        numpy_labels = captions.numpy()

    # print(numpy_images)
    # print(numpy_labels)

    # print the first object in both datasets
    for img_path, captions in test_dataset.take(1):
        numpy_images = img_path.numpy()
        numpy_labels = captions.numpy()

    # print(numpy_images)
    # print(numpy_labels)

    # test image loading
    image_model = load_mobilenet_model()
    # print the model
    print(image_model)

    # print image path
    print("image path: ", numpy_images[0])

    # test image loading
    img = imageLoadResize(numpy_images, image_model)
    print(img.shape)

    # test tokenizer
    tokenizer = create_tokenizer(train_dataset=train_dataset, test_tokenizer=True)
