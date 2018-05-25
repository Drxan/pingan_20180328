#encoding:utf-8
"""
@project : Evaluation
@file : losses
@author : Drxan
@create_time : 18-3-30 下午9:58
"""
import tensorflow as tf
import keras.backend as K


def l2_normalize(x, axis):
    """Normalizes a tensor wrt the L2 norm alongside the specified axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform normalization.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis %= len(x.get_shape())
    return tf.nn.l2_normalize(x, dim=axis)


def cosine_proximity(y_true, y_pred):
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)

def softmax(x, axis=-1):
    """Softmax of a tensor.

    # Arguments
        x: A tensor or variable.
        axis: The dimension softmax would be performed on.
            The default is -1 which indicates the last dimension.

    # Returns
        A tensor.
    """
    return tf.nn.softmax(x, dim=axis)


cosine = cosine_proximity
soft_max = softmax