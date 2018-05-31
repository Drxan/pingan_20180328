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


def ndcg(y_true, y_pred, batch_size=32, group_size=16):
    y_true_sort, y_true_idx = tf.nn.top_k(y_true, k=group_size)
    y_pred_sort, y_pred_idx = tf.nn.top_k(y_pred, k=group_size)
    # 对y_true按照y_pred排序
    shape_y_pred = tf.shape(y_pred)
    auxiliary_indices = tf.meshgrid(
        *[tf.range(d) for d in (tf.unstack(shape_y_pred[:(y_pred.get_shape().ndims - 1)]) + [group_size])], indexing='ij')
    sort_y_true = tf.gather_nd(y_true, tf.stack(auxiliary_indices[:-1] + [y_pred_idx], axis=-1))

    rank = (list(np.arange(1, group_size+1))) * batch_size
    ranks = tf.constant(rank, dtype=tf.float32, shape=(batch_size, group_size))

    idcg = tf.reduce_sum(tf.div(y_true_sort, tf.log(np.e + ranks)), -1)
    dcg = tf.reduce_sum(tf.div(sort_y_true, tf.log(np.e + ranks)), -1)
    ndcg_score = tf.div(dcg, idcg)
    return ndcg_score


cosine = cosine_proximity
ndcg = ndcg