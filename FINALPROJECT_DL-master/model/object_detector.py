# Filename:
#   object_detector.py
# Description:
#   Functionality for encoding images (i.e. concatenation of CNN output and image extraction output)
#   NOTE: Incomplete: could not get integration working successfully 
# Citations:
#   With reference to: https://www.tensorflow.org/hub/tutorials/object_detection
#
import tensorflow as tf
import tensorflow.keras.layers as tf_kl
import tensorflow_hub as tf_hub
import numpy as np

# Download the SSD object detector MobileNetv2 backing pre-trained on Imagenet
#
def download_od():
    pretrained_model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    od = tf_hub.load(pretrained_model_url).signatures["default"]
    return od

def get_obj_detector_encoding(od, image):
    objdet = od(image)
    objdet_vec = np.zeros(2048)
    for idx in range(100):
        jdx = idx*6
        objdet_vec[jdx] = objdet["detection_classes"][:,idx]
        objdet_vec[jdx+1:jdx+5] = objdet["detection_boxes"][:,idx,:]
        objdet_vec[jdx+5] = objdet["detection_scores"][:,idx]
    return objdet_vec