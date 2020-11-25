# coding=utf-8
import tensorflow as tf
import numpy as np


def circle_loss(pair_wise_cosine_matrix, pred_true_mask,
                pred_neg_mask,
                margin=0.25,
                gamma=64):
    """
    @:param pair_wise_cosine_matrix 所有样本对的相似度矩阵
    @:param pred_true_mask 正样本对的mask矩阵
    @:param pred_neg_mask 负样本对的mask矩阵
    https://github.com/zhen8838/Circle-Loss/blob/master/circle_loss.py
    """
    O_p = 1 + margin
    O_n = -margin

    Delta_p = 1 - margin
    Delta_n = margin

    ap = tf.nn.relu(-tf.stop_gradient(pair_wise_cosine_matrix * pred_true_mask) + 1 + margin)
    an = tf.nn.relu(tf.stop_gradient(pair_wise_cosine_matrix * pred_neg_mask) + margin)

    logit_p = -ap * (pair_wise_cosine_matrix - Delta_p) * gamma * pred_true_mask
    logit_n = an * (pair_wise_cosine_matrix - Delta_n) * gamma * pred_neg_mask

    logit_p = logit_p - (1 - pred_true_mask) * 1e12
    logit_n = logit_n - (1 - pred_neg_mask) * 1e12

    joint_neg_loss = tf.reduce_logsumexp(logit_n, axis=-1)
    joint_pos_loss = tf.reduce_logsumexp(logit_p, axis=-1)
    logits = tf.nn.softplus(joint_neg_loss + joint_pos_loss)
    return logits


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
  Args:
    labels: tf.int32 `Tensor` with shape [batch_size]
  Returns:
    mask: tf.bool `Tensor` with shape [batch_size, batch_size]
  """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
  Args:
    labels: tf.int32 `Tensor` with shape [batch_size]
  Returns:
    mask: tf.bool `Tensor` with shape [batch_size, batch_size]
  """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


input_tensor = tf.convert_to_tensor(np.random.random((10, 16)).astype(np.float32))
input_tensor = tf.nn.l2_normalize(input_tensor, axis=-1)
labels = tf.convert_to_tensor([1, 0, 2, 2, 1, 1, 4, 0, 4, 1])

# [10, 10]
pair_wise_cosine_matrix = tf.matmul(input_tensor, tf.transpose(input_tensor))

positive_mask = _get_anchor_positive_triplet_mask(labels)
negative_mask = _get_anchor_negative_triplet_mask(labels)

positive_mask = tf.cast(positive_mask, tf.float32)
negative_mask = tf.cast(negative_mask, tf.float32)

loss = circle_loss(pair_wise_cosine_matrix, positive_mask,
                   negative_mask,
                   margin=0.25,
                   gamma=64)
sess = tf.Session()
print(sess.run([positive_mask, negative_mask, loss]))
