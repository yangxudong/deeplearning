# coding: utf-8
import tensorflow as tf


def triplet_center_loss(features, labels, num_classes, margin, alpha=0.5):
    """获取triplet center loss及center的更新op
    paper: Triplet-Center Loss for Multi-View 3D Object Retrieval
    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
                features should be l2 normalized.
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
        margin: the margin of triplet loss
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers_update_op: op, 用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
        for example:
        ```
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = ...
        ```
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, trainable=False,
                              initializer=tf.contrib.layers.xavier_initializer())

    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    square_a = tf.reduce_sum(tf.square(features), axis=1, keepdims=True)
    square_b = tf.reduce_sum(tf.square(tf.transpose(centers)), axis=0, keepdims=True)
    # shape (batch_size, num_classes)
    distances = square_a + square_b - 2.0 * tf.matmul(features, centers, transpose_b=True)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(0.5 * distances, 0.0)

    labels = tf.cast(labels, tf.int32)
    anchor_positive_dist = tf.batch_gather(distances, tf.expand_dims(labels, -1))  # (batch_size, 1)

    max_distances = tf.reduce_max(distances, axis=1, keepdims=True)
    mask = tf.one_hot(labels, num_classes) * max_distances
    anchor_negative_dist = tf.reduce_min(distances + mask, axis=1, keepdims=True)

    # 计算loss
    triplet_loss = tf.maximum(anchor_positive_dist - anchor_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)
    return triplet_loss, centers_update_op
