import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def hierarchical_triplet_loss(features, labels, embeddings, min_pos_id=0, beta=0.5, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        features: features of the batch
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim). Embeddings should be l2 normalized.
        beta: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    # pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    pairwise_dist = pairwise_distance(embeddings, squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    margin = _get_dynamic_margin(features, beta)
    print("margin", margin.shape)
    mask = _get_triplet_mask(labels, min_pos_id)
    mask = tf.to_float(mask)
    triplet_loss, fraction_positive_triplets = batch_all_triplet_loss_v2(
        anchor_positive_dist, anchor_negative_dist, mask, margin)
    tf.summary.scalar("loss/fraction_positive_triplets", fraction_positive_triplets)

    # mask_lvl2 = _get_triplet_mask(features["parent_id"])
    # mask_lvl2 = tf.to_float(mask_lvl2) - mask  # only include the instances of the same parent but has different label
    # margin_lvl2 = _get_dynamic_margin_v2(features)
    # triplet_loss2, fraction_positive_triplets2 = batch_all_triplet_loss_v2(
    #     anchor_positive_dist, anchor_negative_dist, mask_lvl2, margin_lvl2)
    # tf.summary.scalar("loss/fraction_positive_triplets2", fraction_positive_triplets)
    # return triplet_loss + triplet_loss2
    return triplet_loss


def batch_all_triplet_loss_v2(anchor_positive_dist, anchor_negative_dist, mask, margin):
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-8))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def pairwise_distance(feature, squared=False, normalized=True):
    """from the source code of `tf.contrib.losses.metric_learning.triplet_semihard_loss`
    Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
      normalized: Boolean, whether or not input feature has be l2 normalized.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    if normalized:
        pairwise_distances_squared = 2.0 * (1.0 - math_ops.matmul(feature, array_ops.transpose(feature)))
    else:
        pairwise_distances_squared = math_ops.add(
            math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
            math_ops.reduce_sum(math_ops.square(array_ops.transpose(feature)), axis=[0], keepdims=True))\
            - 2.0 * math_ops.matmul(feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        # Get the mask where the zero distances are at.
        error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared +
            math_ops.cast(error_mask, dtypes.float32) * 1e-16)
        # Undo conditionally adding 1e-16.
        pairwise_distances = math_ops.multiply(
            pairwise_distances,
            math_ops.cast(math_ops.logical_not(error_mask), dtypes.float32))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
        array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    # dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels, minPosId):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))  # (batch, batch)
    positive = tf.greater_equal(labels, minPosId)
    positive_matric = tf.logical_and(tf.expand_dims(positive, 0), tf.expand_dims(positive, 1))
    positive_equal = tf.logical_and(label_equal, positive_matric)

    # i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_j = tf.expand_dims(positive_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


# def _get_dynamic_margin_v2(features, beta1=0.3, beta2=0.49):
def _get_dynamic_margin_v2(features, beta1=0.1, beta2=0.2):
    """ return beta + d(labels[j], labels[k]) of shape (1, batch_size, batch_size) """
    grand_parent = features["grand_parent"]
    grand_parents_equal = tf.equal(tf.expand_dims(grand_parent, 0), tf.expand_dims(grand_parent, 1))
    ones = tf.ones_like(grand_parents_equal, dtype=tf.float32)
    margin = tf.where(grand_parents_equal, ones * beta1, ones * beta2)
    return tf.expand_dims(margin, 0)


def _get_dynamic_margin(features, beta=0.5, beta1=0.35, beta2=0.15):
    """ return beta + d(labels[j], labels[k]) of shape (1, batch_size, batch_size) """
    parents = features["parent_id"]
    grand_parent = features["grand_parent"]
    parents_equal = tf.equal(tf.expand_dims(parents, 0), tf.expand_dims(parents, 1))
    grand_parents_equal = tf.equal(tf.expand_dims(grand_parent, 0), tf.expand_dims(grand_parent, 1))
    print("parents_equal", parents_equal.shape)
    print("grand_parents_equal", grand_parents_equal.shape)
    ones = tf.ones_like(parents_equal, dtype=tf.float32)
    zeros = tf.zeros_like(parents_equal, dtype=tf.float32)
    part_margin = tf.where(parents_equal, zeros, ones * beta1)
    margin = tf.where(grand_parents_equal, part_margin, ones * beta2) + beta
    return tf.expand_dims(margin, 0)


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-8))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


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


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


if __name__ == '__main__':
    a = tf.constant([[0.1, 0.3, 0.5],
                     [0.01, 0.34, 0.64],
                     [0.02, 0.34, 0.64],
                     [0.4, 0.44, 0.1]])
    norm_a = tf.nn.l2_normalize(a, axis=1)
    # distance0 = pairwise_distance(norm_a, False, False)
    distance1 = pairwise_distance(norm_a)
    # distance2 = pairwise_distance(a, False, False)
    distance3 = _pairwise_distances(a)
    distance4 = _pairwise_distances(norm_a)
    with tf.Session() as sess:
        # print(sess.run(norm_a))
        # print(sess.run(tf.reduce_sum(tf.square(norm_a), axis=1)))
        # print(sess.run(distance0))
        print(sess.run(distance1))
        # print(sess.run(distance2))
        print(sess.run(distance3))
        print(sess.run(distance4))
