# coding: utf-8
import tensorflow as tf


def sparse_softmax_cross_entropy_with_prior(labels, logits, priors, tau=1.0):
    """带先验分布的稀疏交叉熵: Long-Tail Learning via Logit Adjustment. priors: shape is [num_classes], 类别的先验概率分布"""
    log_priors = tf.math.log(priors)
    if len(log_priors.shape.as_list()) == 1:
        log_priors = tf.expand_dims(log_priors, 0)
    # print(log_priors.shape)
    # print(logits.shape)
    # print(labels.shape)
    logits += tau * log_priors
    return tf.losses.sparse_softmax_cross_entropy(labels, logits)

