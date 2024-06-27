# standard imports
import tensorflow as tf
import os, re, pathlib, einops
import numpy as np
import matplotlib.pyplot as plt
import string, pickle
from tqdm import tqdm

# from transformers import AutoImageProcessor, TFViTModel
# from datasets import load_dataset


def load_8k_data(plot=False, img_path="./flickr8k", text_path="./flickr8k"):
    # Paths from project folder
    #
    image_dir_path = f"{img_path}/Flicker8k_Dataset/"
    caption_file_path = f"{text_path}/Flickr8k.token.txt"
    train_img_file_path = f"{text_path}/Flickr_8k.trainImages.txt"
    test_img_file_path = f"{text_path}/Flickr_8k.testImages.txt"

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

    # pic = numpy_images.decode()
    # x = plt.imread(pic)
    # plt.imshow(x)
    # plt.show()
    # print(numpy_labels)

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


def load_resnet50_model():
    IMAGE_SHAPE = (224, 224, 3)
    res50 = tf.keras.applications.ResNet50(input_shape=IMAGE_SHAPE, include_top=False)
    res50.trainable = False
    # tack on the image shape to the model for convenience
    res50.image_shape = IMAGE_SHAPE

    return res50


def load_resnet101v2_model():
    IMAGE_SHAPE = (224, 224, 3)
    res101 = tf.keras.applications.ResNet101V2(
        input_shape=IMAGE_SHAPE, include_top=False
    )
    res101.trainable = False
    # tack on the image shape to the model for convenience
    res101.image_shape = IMAGE_SHAPE

    return res101


# def load_vit_model():
#     IMAGE_SHAPE = (1,1,1)
#     model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


def imageLoadResize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])

    return img


def standardize(s):
    s = tf.strings.lower(s)
    s = tf.strings.regex_replace(s, f"[{re.escape(string.punctuation)}]", "")
    s = tf.strings.join(["[START]", s, "[END]"], separator=" ")
    return s


def create_custom_tokenizer(train_dataset, vocabulary_size=5000):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size, standardize=standardize, ragged=True
    )
    tokenizer.adapt(train_dataset.map(lambda fp, txt: txt).unbatch().batch(1024))
    return tokenizer


def word_to_index(tokenizer, words):
    out = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary()
    )
    return out(words)


def index_to_word(tokenizer, indices):
    out = tf.keras.layers.StringLookup(
        invert=True, mask_token="", vocabulary=tokenizer.get_vocabulary()
    )
    return out(indices)


def match_shapes(images, captions):
    caption_shape = einops.parse_shape(captions, "b c")
    captions = einops.rearrange(captions, "b c -> (b c)")
    images = einops.repeat(images, "b ... -> (b c) ...", c=caption_shape["c"])
    return images, captions


def prepare_txt(image, captions):
    tokens = tokenizer(captions)
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return (image, input_tokens), label_tokens


def prepare_dataset(ds, batch_size=32, shuffle_buffer=1000):
    ds = (
        ds.shuffle(10000)
        .map(lambda path, caption: (imageLoadResize(path), caption))
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size)
    )

    def to_tensor(inputs, labels):
        (images, in_token), out_token = inputs, labels
        return (images, in_token.to_tensor()), out_token.to_tensor()

    return (
        ds.map(match_shapes, tf.data.AUTOTUNE)
        .unbatch()
        .shuffle(shuffle_buffer)
        .batch(batch_size)
        .map(prepare_txt, tf.data.AUTOTUNE)
        .map(to_tensor, tf.data.AUTOTUNE)
    )


def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):
    def custom_reader_func(datasets):
        datasets = datasets.shuffle(1000)
        return datasets.interleave(lambda x: x, cycle_length=cycle_length)

    ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

    def drop_index(i, x):
        return x

    ds = (
        ds.map(drop_index, tf.data.AUTOTUNE)
        .shuffle(shuffle)
        .padded_batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


def extract_features(ds, save_path, image_model, shards=10, batch_size=32):
    # Load the images and make batches.
    ds = (
        ds.map(lambda path, caption: (imageLoadResize(path), caption))
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size)
    )

    # Run the feature extractor on each batch
    # Don't do this in a .map, because tf.data runs on the CPU.
    def gen():
        for images, captions in tqdm(ds):
            feature_maps = image_model(images)

            feature_maps, captions = match_shapes(feature_maps, captions)
            yield feature_maps, captions

    # Wrap the generator in a new tf.data.Dataset.
    new_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=image_model.output_shape),
            tf.TensorSpec(shape=(None,), dtype=tf.string),
        ),
    )

    # Apply the tokenization
    new_ds = new_ds.map(prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(1000)

    # Save the dataset into shard files.
    def shard_func(i, item):
        return i % shards

    new_ds.enumerate().save(save_path, shard_func=shard_func)


def prep_data(
    dataset_name="flickr8k",
    embedding_model_name="mobilenet",
    tokenizer_name="custom",
):
    print(f"Loding {dataset_name} dataset...")
    # load data into memory
    if dataset_name == "flickr8k":
        train_dataset, test_dataset = load_8k_data(plot=False)
        print("Loaded flickr8k dataset")
    else:
        raise ValueError("Invalid dataset name")

    print(f"Loading {embedding_model_name} embedding model...")
    # load embedding model
    if embedding_model_name == "mobilenet":
        image_model = load_mobilenet_model()
        print("Loaded MobileNet embedding model")
    elif embedding_model_name == "resnet50":
        image_model = load_resnet50_model()
        print("Loaded MobileNet embedding model")
    elif embedding_model_name == "resnet101":
        image_model = load_resnet101v2_model()
        print("Loaded MobileNet embedding model")
    else:
        raise ValueError("Invalid embedding model name")
    global IMAGE_SHAPE
    IMAGE_SHAPE = image_model.image_shape

    print(f"Creating {tokenizer_name} tokenizer...")
    # create tokenizer
    global tokenizer
    if tokenizer_name == "custom":
        tokenizer = create_custom_tokenizer(train_dataset)
        print("Created custom tokenizer")
    else:
        raise ValueError("Invalid tokenizer name")

    # print("Preparing datasets")
    # # prepare dataset
    # new_train_dataset_raw = prepare_dataset(train_dataset)
    # new_test_dataset_raw = prepare_dataset(test_dataset)

    # print(new_train_dataset_raw.element_spec)
    # print(new_test_dataset_raw.element_spec)

    # print("Prepared datasets")

    # train file name
    train_path = f"train_{dataset_name}_{embedding_model_name}_{tokenizer_name}"
    # test file name
    test_path = f"test_{dataset_name}_{embedding_model_name}_{tokenizer_name}"

    # save datasets
    # check if datasets already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Datasets already exist, done")
    else:
        print("Extracting features and saving datasets...")
        extract_features(train_dataset, train_path, image_model)
        extract_features(test_dataset, test_path, image_model)
        print("Saved datasets")

    print("getting raw data for initializations later")
    train_raw = prepare_dataset(train_dataset)
    test_raw = prepare_dataset(test_dataset)

    return (train_path, test_path, image_model, tokenizer, train_raw, test_raw)


if __name__ == "__main__":
    prep_data(
        dataset_name="flickr8k",
        embedding_model_name="mobilenet",
        tokenizer_name="custom",
    )
