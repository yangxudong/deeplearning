#!/usr/bin/env bash
# coding: utf-8

import tensorflow as tf


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op

    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32, trainable=False,
                              initializer=tf.contrib.layers.xavier_initializer())
    # initializer=tf.constant_initializer(0))
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    # labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.losses.mean_squared_error(features, centers_batch)

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
    return loss, centers, centers_update_op


# from facenet
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def AM_logits_compute(embeddings, label_batch, args, nrof_classes):
    '''
    loss head proposed in paper:<Additive Margin Softmax for Face Verification>
    link: https://arxiv.org/abs/1801.05599
    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    '''
    m = 0.35
    s = 30

    with tf.name_scope('AM_logits'):
        kernel = tf.get_variable(name='kernel', dtype=tf.float32, shape=[args.embedding_size, nrof_classes],
                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
        cos_theta = tf.matmul(embeddings, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1)  # for numerical steady
        phi = cos_theta - m
        label_onehot = tf.one_hot(label_batch, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot, 1), phi, cos_theta)

        return adjust_theta


if __name__ == '__main__':
    b = tf.constant([[1, 1], [2, 2], [3, 3]], dtype=tf.float32)
    with tf.Session() as sess:
        print(sess.run(tf.nn.l2_loss(b)))
