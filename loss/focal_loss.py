# coding: utf-8
"""Implements Focal loss."""
#  ____  __    ___   __   __      __     __   ____  ____
# (  __)/  \  / __) / _\ (  )    (  )   /  \ / ___)/ ___)
#  ) _)(  O )( (__ /    \/ (_/\  / (_/\(  O )\___ \\___ \
# (__)  \__/  \___)\_/\_/\____/  \____/ \__/ (____/(____/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def sigmoid_focal_loss_with_logits(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args
        y_true: true targets tensor (labels).
        y_pred: predictions tensor (logits).
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    Raises:
        ValueError: If the shape of `sample_weight` is invalid or value of
          `gamma` is less than zero
    """
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the binary cross_entropy
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # convert the predictions into probabilities
    y_pred = tf.nn.sigmoid(y_pred)

    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = 1
    modulating_factor = 1

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)
        modulating_factor = tf.pow((1 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_mean(alpha_factor * modulating_factor * bce, axis=-1, keepdims=True)


def softmax_focal_loss_with_logits(logits, labels, alpha=None, sample_weights=None, gamma=2.0, epsilon=1.e-7):
    """
    Args:
        logits:  [batch_size, num_class]
        labels: [batch_size]  not one-hot !!!
        alpha: [num_class] 一般为其他类的样本比例，样本越多的类对应的alpha越小
    Returns:
        -alpha*(1-y)^r * log(y)
    它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
    logits softmax之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

    怎么把alpha的权重加上去？
    通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的
    """
    labels = tf.cast(labels, dtype=tf.int32)
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * num_class]
    batch_size, num_class = get_shape_list(logits)
    # (N,) > (N,), 但是数值变换了，变成了每个label在 N * num_class 中的位置
    labels_shift = tf.range(0, batch_size) * num_class + labels
    # (N * num_class,) > (N,)
    prob = tf.gather(softmax, labels_shift)  # 属于当前类的概率
    # 预防预测概率值为0的情况; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    weights = tf.pow(tf.subtract(1., prob), gamma)

    if alpha is not None:
        alpha = tf.constant(alpha, dtype=tf.float32)  # (num_class, 1)
        # (num_class ,1) > (N,)
        alpha_choice = tf.gather(alpha, labels)
        weights = tf.multiply(alpha_choice, weights)

    if sample_weights is not None:
        weights = tf.multiply(weights, sample_weights)
    return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights=weights)

def get_shape_list(tensor):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
