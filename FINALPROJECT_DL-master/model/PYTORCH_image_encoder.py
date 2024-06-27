# Filename:
#   image_encoder.py
# Description:
#   Functionality for encoding images with CNN
#

from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn

HIDDEN_SIZE = 512

# Initialize and return image encoder model
# Citation: https://pytorch.org/vision/stable/models.html tutorial
# Citation: https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648 help deleting CNN classifier layers
#
def init_encoder():
    weights = Inception_V3_Weights.DEFAULT
    full_cnn = inception_v3(weights=weights)
    retained_layer_list = list(full_cnn.children())[:-2]
    cnn_encoder = torch.nn.Sequential(
        *retained_layer_list
    )
    return cnn_encoder


# TODO: Encode tokenized caption to internal representation
def encode_tokenized_caption(tok_caption):
    pass

if __name__ == "__main__":
    init_encoder()