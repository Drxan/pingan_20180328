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


cosine = cosine_proximity
