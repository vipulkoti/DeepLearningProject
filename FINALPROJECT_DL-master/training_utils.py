# Filename:
#   training_utils.py
# Description:
#   Functionality for training and evaluating our models
# Citations:
#   Heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning

import tensorflow as tf

# With heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
#
def loss_func(y_true, y_pred):
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)

    # Ensure that START, END, UNK, EMPTY tokens are ignored in loss calculation
    #
    ignoremask_elements = (y_true != 0) & (ce < 1e8)
    mask = tf.cast(ignoremask_elements, ce.dtype)

    # Element-wise applying ignore mask and normalizing by caption sizes
    #
    loss = ce * mask
    print(tf.reduce_sum(loss)/tf.reduce_sum(mask))
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

# With heavy implementation reference: https://www.tensorflow.org/tutorials/text/image_captioning
#
def top1_accuracy_func(y_true, y_pred):
    # Ensure that START, END, UNK, EMPTY tokens are ignored in loss calculation
    #
    ignoremask_elements = tf.cast((y_true != 0), tf.float32)

    # Cross check best guesses against actuals; normalize correct guesses by total guesses
    #
    best_guesses = tf.argmax(y_pred, axis=-1)
    print(best_guesses)
    correct_guesses = tf.cast(int(y_true) == int(best_guesses), tf.float32)
    return tf.reduce_sum(ignoremask_elements*correct_guesses)/tf.reduce_sum(ignoremask_elements)

# So close to getting this to work, but I was unable
# I will try to come back to this
#
def _disfunctional_top5_accuracy_func(y_true, y_pred):
    # Ensure that START, END, UNK, EMPTY tokens are ignored in loss calculation
    #
    ignoremask_elements = tf.cast((y_true != 0), tf.float32)

    # Cross check best guesses against actuals; normalize correct guesses by total guesses
    #
    best_guesses = tf.math.top_k(y_pred, k=1).indices
    correct_guesses = tf.cast(int(y_true == int(best_guesses), tf.float32))
    total_accuracy = tf.reduce_sum(ignoremask_elements*correct_guesses)/tf.reduce_sum(ignoremask_elements)
    
    return total_accuracy

def BLEU_score(y_true, y_pred):
    pass
