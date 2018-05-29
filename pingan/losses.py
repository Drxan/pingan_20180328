#encoding:utf-8
"""
@project : Evaluation
@file : losses
@author : Drxan
@create_time : 18-3-30 下午9:58
"""
import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd


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


def ndcg(y_true, y_pred, group_size=8):
    a=np.sort(y_true, axis=-1)
    print('a type',type(a))
    sample_num = tf.size(y_pred)
    print(sample_num)
    ranks = np.arange(1, group_size + 1)
    # idcg
    y_true_sort = np.sort(y_true)[::-1]
    idcg = y_true[0] + np.sum(y_true[1:] / np.log2(ranks[1:]))
    # dcg
    #dcg = y_true_new[0] + np.sum(y_true_new[1:] / np.log2(ranks[1:]))
    # ndcg
    #ndcg_score = dcg/idcg
    return idcg


cosine = cosine_proximity
ndcg = ndcg