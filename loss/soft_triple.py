import tensorflow as tf


def soft_triple_loss(labels, embeddings, num_classes, num_centers=2, lamb=20.0, gamma=0.1, delta=0.01, tau=0.2):
    """
    paper: SoftTriple Loss: Deep Metric Learning Without Triplet Sampling
    ref: https://medium.com/@sebastianpinedaarango/implementation-of-softtriple-loss-dfff803bab7f
    """
    embedding_size = embeddings.shape[-1]
    centers = tf.get_variable("soft_triple_centers", shape=[num_classes * num_centers, embedding_size],
                              dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    norm_centers = tf.math.l2_normalize(centers, axis=-1)
    norm_embeddings = tf.math.l2_normalize(embeddings, axis=-1)
    inner_logits = tf.matmul(norm_embeddings, norm_centers, transpose_b=True)
    inner_logits = tf.reshape(inner_logits, [-1, num_centers, num_classes])
    inner_softmax = tf.math.softmax(inner_logits / gamma, axis=1)
    sim_i_c = tf.reduce_sum(tf.multiply(inner_softmax, inner_logits), axis=1)

    one_hot_label = tf.one_hot(tf.cast(labels, tf.int32), num_classes, dtype=tf.float32)
    logits = lamb * (sim_i_c - delta * one_hot_label)

    loss = tf.losses.softmax_cross_entropy(one_hot_label, logits)

    if tau > 0.0 and num_centers > 1:  # do regularize, make adaptive number of centers
        sim_centers = tf.matmul(norm_centers, norm_centers, transpose_b=True)
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        dist_centers = tf.maximum(2.0 - 2.0 * sim_centers, 0.0)

        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(dist_centers, 0.0))
        dist_centers = dist_centers + mask * 1e-16
        dist_centers = tf.sqrt(dist_centers)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        dist_centers = dist_centers * (1.0 - mask)  # shape: (C*K, C*K)

        checkerboard = tf.range(num_classes, dtype=tf.int32)
        checkerboard = tf.one_hot(checkerboard, depth=num_classes, dtype=tf.float32)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=0)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=1)  # shape: (C*K, C*K)

        dist_centers = tf.multiply(dist_centers, checkerboard)
        mask = tf.ones_like(dist_centers, dtype=tf.float32) - tf.eye(num_classes * num_centers, dtype=tf.float32)
        dist_centers = tf.multiply(dist_centers, mask)
        reg_numerator = tau * tf.reduce_sum(dist_centers) / 2.0
        reg_denominator = num_classes * num_centers * (num_centers - 1.0)
        loss_reg = reg_numerator / reg_denominator
        loss += loss_reg
    return loss, logits


def soft_triple(gt, embeddings, dim_features, num_class, num_centers=2, p_lambda=20.0, p_tau=0.2, p_gamma=0.1,
                p_delta=0.01, with_reg=True):
    """
    paper: SoftTriple Loss: Deep Metric Learning Without Triplet Sampling
    code from: https://github.com/geonm/tf_SoftTriple_loss
    """
    large_centers = tf.get_variable(name='feature_extractor/large_centers',
                                    shape=[num_class * num_centers, dim_features],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                    trainable=True)
    large_centers = tf.nn.l2_normalize(large_centers, axis=-1)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    large_logits = tf.matmul(embeddings, large_centers, transpose_b=True)  # [batch_size, num_class * num_centers]

    batch_size = tf.shape(large_logits)[0]

    rs_large_logits = tf.reshape(large_logits, [batch_size, num_centers, num_class])

    exp_rs_large_logits = tf.exp((1.0 / p_gamma) * rs_large_logits)

    sum_rs_large_logits = tf.reduce_sum(exp_rs_large_logits, axis=1, keepdims=True)

    coeff_large_logits = exp_rs_large_logits / sum_rs_large_logits

    rs_large_logits = tf.multiply(rs_large_logits, coeff_large_logits)

    logits = tf.reduce_sum(rs_large_logits, axis=1, keepdims=False)

    # get labels_map
    gt = tf.reshape(gt, [-1])  # e.g., [0, 7, 3, 22, 39, ...]

    gt_int = tf.cast(gt, tf.int32)

    labels_map = tf.one_hot(gt_int, depth=num_class, dtype=tf.float32)

    # subtract p_delta
    delta_map = p_delta * labels_map

    logits_delta = logits - delta_map
    scaled_logits_delta = p_lambda * (logits_delta)

    # get xentropy loss
    loss_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits_delta, labels=labels_map)
    loss_xentropy = tf.reduce_mean(loss_xentropy, name='loss_xentropy')

    # get regularizer terms
    loss_reg = 0.0
    if with_reg:
        # get R function
        # large_centers = [num_class * num_centers, dim_features]
        sim_large_centers = tf.abs(tf.matmul(large_centers, large_centers,
                                             transpose_b=True))  # [num_class * num_centers, num_class * num_centers]

        # check error
        # sim_large_centers = tf.where(sim_large_centers > 1.0, tf.ones_like(sim_large_centers, dtype=tf.float32), sim_large_centers)

        dist_large_centers = tf.sqrt(tf.abs(2.0 - 2.0 * sim_large_centers) + 1e-10)
        checkerboard = tf.range(num_class, dtype=tf.int32)
        checkerboard = tf.one_hot(checkerboard, depth=num_class, dtype=tf.float32)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=0)
        checkerboard = tf.keras.backend.repeat_elements(checkerboard, num_centers, axis=1)

        dist_large_centers = tf.multiply(dist_large_centers, checkerboard)

        mask = tf.ones_like(dist_large_centers, dtype=tf.float32) - tf.eye(num_class * num_centers, dtype=tf.float32)

        dist_large_centers = p_tau * tf.multiply(dist_large_centers, mask)

        reg_numer = tf.reduce_sum(dist_large_centers) / 2.0

        reg_denumer = num_class * num_centers * (num_centers - 1.0)

        loss_reg = reg_numer / reg_denumer

    # l2 reg loss
    # reg_embeddings = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings), 1))
    # reg_centers = tf.reduce_mean(tf.reduce_sum(tf.square(large_centers), 1))
    # loss_l2_reg = tf.multiply(0.25 * 0.002, reg_embeddings + reg_centers, name='loss_l2_reg')

    total_loss = loss_xentropy + loss_reg

    return total_loss
