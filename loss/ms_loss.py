import tensorflow as tf
import modeling


def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=False, embed_normed=True):
    """
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    """
    # make sure embedding should be l2-normalized
    if not embed_normed:
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    embed_shape = modeling.get_shape_list(embeddings)
    batch_size = embed_shape[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(batch_size, dtype=tf.float32)
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)
    return loss


def recall_at_k(labels, embeddings, k, embed_normed=True):
    # make sure embedding should be l2-normalized
    if not embed_normed:
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    # batch_size = tf.size(labels)
    embed_shape = modeling.get_shape_list(embeddings)
    batch_size = embed_shape[0]

    sim_mat = tf.matmul(embeddings, embeddings, transpose_b=True)
    sim_mat = sim_mat - tf.eye(batch_size) * 2.0

    labels = tf.expand_dims(labels, -1)
    mask = tf.equal(labels, tf.transpose(labels))  # shape: (batch_size, batch_size)
    eye = tf.eye(batch_size, dtype=tf.bool)
    mask = tf.logical_and(mask, tf.logical_not(eye))
    mask_pos = tf.where(mask, sim_mat, -tf.ones_like(sim_mat))  # shape: (batch_size, batch_size)

    if isinstance(k, int):
        _, pos_top_k_idx = tf.nn.top_k(mask_pos, k)  # shape: (batch_size, k)
        return tf.metrics.recall_at_k(labels=tf.to_int64(pos_top_k_idx), predictions=sim_mat, k=k)
    if any((isinstance(k, list), isinstance(k, tuple), isinstance(k, set))):
        metrics = {}
        for kk in k:
            if k < 1:
                continue
            _, pos_top_k_idx = tf.nn.top_k(mask_pos, kk)
            metrics["recall@"+str(kk)] = tf.metrics.recall_at_k(labels=tf.to_int64(pos_top_k_idx), predictions=sim_mat, k=kk)
        return metrics
    raise ValueError("k should be a `int` or a list/tuple/set of int.")


def get_matrix_mask_indices(matrix, num_rows=None):
    if num_rows is None:
        num_rows = modeling.get_shape_list(matrix)[0]
    indices = tf.where(matrix)
    num_indices = tf.shape(indices)[0]
    elem_per_row = tf.bincount(tf.cast(indices[:, 0], tf.int32), minlength=num_rows)
    max_elem_per_row = tf.reduce_max(elem_per_row)
    row_start = tf.concat([[0], tf.cumsum(elem_per_row[:-1])], axis=0)
    r = tf.range(max_elem_per_row)
    idx = tf.expand_dims(row_start, 1) + r
    idx = tf.minimum(idx, num_indices - 1)
    result = tf.gather(indices[:, 1], idx)
    # replace invalid elements with -1
    result = tf.where(tf.expand_dims(elem_per_row, 1) > r, result, -tf.ones_like(result))
    max_index_per_row = tf.reduce_max(result, axis=1, keepdims=True)
    max_index_per_row = tf.tile(max_index_per_row, [1, max_elem_per_row])
    result = tf.where(result >= 0, result, max_index_per_row)
    return result


def average_precision_at_k(labels, embeddings, k, embed_normed=True):
    # make sure embedding should be l2-normalized
    if not embed_normed:
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    # batch_size = tf.size(labels)
    embed_shape = modeling.get_shape_list(embeddings)
    batch_size = embed_shape[0]

    sim_mat = tf.matmul(embeddings, embeddings, transpose_b=True)
    sim_mat = sim_mat - tf.eye(batch_size) * 2.0

    labels = tf.expand_dims(labels, -1)
    mask = tf.equal(labels, tf.transpose(labels))  # shape: (batch_size, batch_size)
    label_indices = get_matrix_mask_indices(mask)
    if isinstance(k, int):
        return tf.metrics.average_precision_at_k(label_indices, sim_mat, k)
    if any((isinstance(k, list), isinstance(k, tuple), isinstance(k, set))):
        metrics = {}
        for kk in k:
            if k < 1:
                continue
            metrics["MAP@"+str(kk)] = tf.metrics.average_precision_at_k(label_indices, sim_mat, kk)
        return metrics
    raise ValueError("k should be a `int` or a list/tuple/set of int.")


def knn(labels, embeddings, k, embed_normed=True):
    # make sure embedding should be l2-normalized
    if not embed_normed:
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    embed_shape = modeling.get_shape_list(embeddings)
    batch_size = embed_shape[0]
    sim_mat = tf.matmul(embeddings, embeddings, transpose_b=True)
    sim_mat = sim_mat - tf.eye(batch_size) * 2.0

    _, top_k_idx = tf.nn.top_k(sim_mat, k)
    top_k_labels = tf.squeeze(tf.gather(labels, top_k_idx))

    def knn_vote(v):
        nearest_k_y, idx, votes = tf.unique_with_counts(v)
        majority_idx = tf.argmax(votes)
        predict_res = tf.gather(nearest_k_y, majority_idx)
        return predict_res

    majority = tf.map_fn(knn_vote, top_k_labels)
    return majority


def knn_metrics(labels, embeddings, k, embed_normed=True):
    knn_result = knn(labels, embeddings, k, embed_normed)
    accuracy = tf.metrics.accuracy(labels, knn_result)

    is_black = tf.where(labels < -100000, tf.ones_like(labels), tf.zeros_like(labels))
    predictions = tf.where(tf.equal(labels, knn_result), is_black, 1 - is_black)
    precision = tf.metrics.precision(is_black, predictions)
    recall = tf.metrics.recall(is_black, predictions)
    return {
        "knn_accuracy@" + str(k): accuracy,
        "knn_precision@" + str(k): precision,
        "knn_recall@" + str(k): recall
    }
