# Filename:
#   caption_decoder.py
# Description:
#   Functionality for decoding text/image feature vectors into captions
# Citations:
#   Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning

import tensorflow as tf
import tensorflow.keras.layers as tf_kl


# Implementation of transformer causal self attention
# Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
#
class SubLayer_SelfAttention(tf_kl.Layer):
    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def __init__(self, multihead_params, norm_params):
        super().__init__()
        self.layer_multihead_att = tf_kl.MultiHeadAttention(**multihead_params)
        self.layer_add_op = tf_kl.Add()
        self.layer_norm = tf_kl.LayerNormalization(**norm_params)

    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def call(self, input):
        # Generate multiheaded self attention
        #
        multihead_att = self.layer_multihead_att(
            query=input, value=input, use_causal_mask=True
        )

        # Add self attention to input and then normalize
        #
        result = self.layer_norm(self.layer_add_op((input, multihead_att)))
        return result


# Implementation of transformer cross attention
#
class SubLayer_CrossAttention(tf_kl.Layer):
    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def __init__(self, multihead_params, norm_params):
        super().__init__()
        self.layer_multihead_att = tf_kl.MultiHeadAttention(**multihead_params)
        self.layer_add_op = tf_kl.Add()
        self.layer_norm = tf_kl.LayerNormalization(**norm_params)

    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def call(self, input, cross_input):
        # Generate multiheaded cross attention
        #
        attn, attn_scores = self.layer_multihead_att(
            query=input,
            value=cross_input,
            use_causal_mask=False,
            return_attention_scores=True,
        )

        self.last_attention_scores = attn_scores

        # Add cross attention to input and then normalize
        #
        result = self.layer_norm(self.layer_add_op((input, attn)))
        return result


# Implementation of transformer feed forward
#
class SubLayer_FeedForward(tf_kl.Layer):
    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def __init__(self, units, dropout_rate):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf_kl.Dense(units, activation="relu"),
                tf_kl.Dense(units),
                tf_kl.Dropout(dropout_rate),
            ]
        )

        self.layer_norm = tf_kl.LayerNormalization()

    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def call(self, input):
        # Generate feed forward
        #
        x = input + self.seq(input)
        return self.layer_norm(x)


# Combine Self attention, cross attention, and feed forward layer -- implement full transformer decoder
#
class CaptionTransformerDecoder(tf_kl.Layer):
    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def __init__(
        self,
        units,
        num_heads=1,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.self_attention = SubLayer_SelfAttention(
            multihead_params={
                "num_heads": num_heads,
                "key_dim": units,
                "dropout": dropout_rate,
            },
            norm_params={"epsilon": 1e-6},
        )

        self.cross_attention = SubLayer_CrossAttention(
            multihead_params={
                "num_heads": num_heads,
                "key_dim": units,
                "dropout": dropout_rate,
            },
            norm_params={"epsilon": 1e-6},
        )

        self.feed_forward = SubLayer_FeedForward(units, dropout_rate)

    # Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
    #
    def call(self, inputs, training=False):
        in_seq, out_seq = inputs

        # first we do self attention with just the output sequence (which is the caption)
        out_seq = self.self_attention(out_seq)

        # then we do cross attention with the output sequence and the input sequence (which is the image features)
        out_seq = self.cross_attention(out_seq, in_seq)

        # save the attention scores for later (we can use this to visualize the attention weights)
        self.last_attention_scores = self.cross_attention.last_attention_scores

        # finally we do feed forward on the output sequence (which is the caption)
        out_seq = self.feed_forward(out_seq)

        return out_seq


if __name__ == "__main__":
    # test the model
    #
    caption_transformer_decoder = CaptionTransformerDecoder(512, 1)

    # generate fake data to test the model
    #
    image_features = tf.random.uniform((1, 64, 512))
    caption = tf.random.uniform((1, 16, 512))

    # run the model
    #
    caption = caption_transformer_decoder((image_features, caption))

    # print the output shape
    #
    print(caption.shape)

    # print the attention scores
    #
    print(caption_transformer_decoder.last_attention_scores.shape)
    print(caption_transformer_decoder.last_attention_scores)
