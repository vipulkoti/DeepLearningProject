# Filename:
#   caption_encoder.py
# Description:
#   Functionality for encoding plaintext captions with position
# Citations:
#   Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning

import tensorflow as tf
import tensorflow.keras.layers as tf_kl

# Add tokenized caption vectors to positional encoding vector of same size
#
class CaptionEmbedder(tf_kl.Layer):
    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def __init__(self, output_dim, vocab_len, max_caption_len):
        super().__init__()
        self.layer_tok_vec = tf_kl.Embedding(
            input_dim=vocab_len,
            output_dim=output_dim,
            embeddings_initializer='uniform',
            mask_zero=True
        )
        self.layer_pos_vec = tf_kl.Embedding(
            input_dim=max_caption_len,
            output_dim=output_dim,
            embeddings_initializer='uniform'
        )
        self.layer_add_op = tf_kl.Add()

    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def call(self, caption):
        # Generate tokenized caption vector and positional encoding vector
        #
        tok_vec = self.tok_vec(caption)
        pos_vec = tf.expand_dims(self.pos_vec(caption, axis=0))

        # Add vectors and return result
        #
        result = self.layer_add_op((tok_vec, pos_vec))
        return result

if __name__ == "__main__":
    print("Test create caption embedding layer")
    print(CaptionEmbedder(10,50,4))
