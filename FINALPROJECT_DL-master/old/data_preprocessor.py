# Filename:
#   data_preprocessor.py
# Description:
#   Functionality for preprocessing text and image dataset & splitting into TRAIN-VAL-TEST 
#

from enum import Enum
import string

CAPTION_FILEPATH = './archive/Flickr8k_text/Flickr8k.token.txt'
TRAIN_IMG_FILEPATH = './archive/Flickr8k_text/Flickr_8k.trainImages.txt'
VAL_IMG_FILEPATH = './archive/Flickr8k_text/Flickr_8k.devImages.txt'
TEST_IMG_FILEPATH = './archive/Flickr8k_text/Flickr_8k.testImages.txt'

class DataSplit(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

# 1. Read captions from text file
# 2. Do basic sentence cleanup
# 3. Return caption text as dictionary {image_id:[caption, caption, ...]}
#
def get_preprocessed_captions():
    # Create dictionary
    #
    raw_captions = open(CAPTION_FILEPATH,'r').read()
    captions = dict()
    for line in raw_captions.split('\n'):
        tokens = line.split()
        if len(line) > 2:
          image_id = tokens[0].split('.')[0]
          image_desc = ' '.join(tokens[1:])
          if image_id not in captions:
              captions[image_id] = list()
          captions[image_id].append(image_desc)

    # Perform basic caption cleanup
    #
    table = str.maketrans('', '', string.punctuation)
    for key, caption_list in captions.items():
        for i in range(len(caption_list)):
            desc = caption_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            caption_list[i] =  ' '.join(desc).strip()
    
    return captions

# Get dataset TRAIN-VAL-TEST split as dictionary {TRAIN: [img_id, ...], VAL: [img_id, ...], TEST: [img_id, ...]}
#
def generate_dataset_splits():
    # Create split set dictionaries
    #
    splits = {DataSplit.TRAIN:[], DataSplit.VAL:[], DataSplit.TEST:[]}
    for split in splits.keys():
        filepath = ""
        if split == DataSplit.TRAIN:
            filepath = TRAIN_IMG_FILEPATH
        elif split == DataSplit.VAL:
            filepath = VAL_IMG_FILEPATH
        else:
            filepath = TEST_IMG_FILEPATH
        with open(filepath,'r') as f:
            img_names = f.read()
        for img_name in img_names.split('\n'):
            if len(img_name) < 2: continue
            image_id = img_name.split('.')[0]
            splits[split].append(image_id)
        print(f"{split} Dataset size: {len(splits[split])}")
    print("\n")
    return splits