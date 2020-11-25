import tensorflow as tf


def f1_reweight_loss(logits, labels, beta2):
    """from paper: Adaptive Scaling for Sparse Detection in Information Extraction"""
    m = tf.count_nonzero(labels, dtype=tf.float32)
    batch_size = tf.shape(labels)[0]
    n = tf.cast(batch_size, tf.float32) - m
    probs = tf.nn.softmax(logits)  # [batch_size, num_classes]
    batch_idx = tf.range(batch_size)
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx, tf.cast(labels, tf.int32)]], 1)  # [batch_size, 2]
    golden_probs = tf.gather_nd(probs, label_with_idx)  # [batch_size]
    zeros = tf.zeros_like(golden_probs)
    is_negative = tf.equal(labels, 0)
    p1 = tf.reduce_sum(tf.where(is_negative, zeros, golden_probs))  # TP
    p2 = tf.reduce_sum(tf.where(is_negative, golden_probs, zeros))  # TN
    neg_weights = p1 / ((beta2 * m) + n - p2 + 1e-8)
    ones = tf.ones_like(golden_probs)
    weights = tf.where(is_negative, ones * neg_weights, ones)
    return tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)


def f1_reweight_loss_v2(logits, labels, beta2):
    probs = tf.nn.softmax(logits)  # [batch_size, num_classes]
    labels = tf.cast(labels, tf.int32)
    negative_idx = tf.where(tf.equal(labels, 0), tf.ones_like(labels, dtype=tf.float32), tf.zeros_like(labels, dtype=tf.float32))
    positive_idx = 1.0 - negative_idx

    batch_idx = tf.range(tf.shape(probs)[0])
    label_with_idx = tf.concat([tf.expand_dims(t, 1) for t in [batch_idx, labels]], 1)
    golden_prob = tf.gather_nd(probs, label_with_idx)
    m = tf.reduce_sum(positive_idx)
    n = tf.reduce_sum(negative_idx)
    p1 = tf.reduce_sum(positive_idx * golden_prob)
    p2 = tf.reduce_sum(negative_idx * golden_prob)
    neg_weight = p1 / ((beta2 * m) + n - p2 + 1e-8)
    all_one = tf.ones(tf.shape(golden_prob))
    loss_weight = all_one * positive_idx + all_one * neg_weight * negative_idx

    loss = - loss_weight * tf.log(golden_prob + 1e-8)
    return loss


def f1_reweight_sigmoid_cross_entropy(logits, labels, beta_square, label_smoothing=0, weights=None):
    probs = tf.nn.sigmoid(logits)
    if len(labels.shape.as_list()) == 1:
        labels = tf.expand_dims(labels, -1)
    labels = tf.to_float(labels)
    batch_size = tf.shape(labels)[0]
    batch_size_float = tf.to_float(batch_size)
    num_pos = tf.reduce_sum(labels, axis=0)
    num_neg = batch_size_float - num_pos
    tp = tf.reduce_sum(probs, axis=0)
    tn = batch_size_float - tp
    neg_weight = tp / (beta_square * num_pos + num_neg - tn + 1e-8)
    neg_weight_tile = tf.tile(tf.expand_dims(neg_weight, 0), [batch_size, 1])
    final_weights = tf.where(tf.equal(labels, 1.0), tf.ones_like(labels), neg_weight_tile)
    if weights is not None:
        if len(weights.shape.as_list()) == 1:
            weights = tf.expand_dims(weights, -1)
        final_weights *= weights
    return tf.losses.sigmoid_cross_entropy(labels, logits, final_weights, label_smoothing=label_smoothing)
