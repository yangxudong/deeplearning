import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers
import random
import collections
import re


def textcnn(x, num_filters=64, filter_sizes=(3, 4, 5), scope_name="textcnn", regularizer=None, reuse=False):
    # x: None * step_dim * embed_dim
    pooled_outputs = []
    for filter_size in filter_sizes:
        scope_name_i = scope_name + "_" + str(filter_size)
        with tf.variable_scope(scope_name_i, reuse=reuse):
            initializer = tf.contrib.layers.variance_scaling_initializer()
            # conv shape: (batch_size, seq_len - filter_size + 1, num_filters)
            conv = tf.layers.conv1d(x, filters=num_filters, kernel_size=filter_size, activation=tf.nn.relu,
                                    name="conv_layer", reuse=reuse, kernel_initializer=initializer,
                                    kernel_regularizer=regularizer)
            pool = tf.reduce_max(conv, axis=1)  # max pooling, shape: (batch_size, num_filters)
        pooled_outputs.append(pool)
    pool_flat = tf.concat(pooled_outputs, 1)  # shape: (batch_size, num_filters * len(filter_sizes))
    return pool_flat


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        if "word_embeddings" in name or "char_embeddings" in name:
            continue
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def scaffold(init_ckp):
    # model_dirs = tf.train.match_filenames_once(init_checkpoint)
    # init_ckp = model_dirs[-1]
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_ckp)
    init_op = tf.train.init_from_checkpoint(init_ckp, assignment_map)
    return tf.train.Scaffold(init_op)


class SemanticModel(tf.estimator.Estimator):
    """An estimator for tensorflow DSSM model"""

    def __init__(self,
                 params,
                 model_dir=None,
                 optimizer='Adagrad',
                 config=None
                 ):
        if not optimizer: optimizer = 'Adagrad'
        self.optimizer = optimizers.get_optimizer_instance(optimizer, params["learning_rate"])

        def _model_fn(features, labels, mode, params):
            """ If mode is `ModeKeys.PREDICT`, `labels=None` will be passed.
                features: query features
                labels: positive doc features
            """
            anchor_id = features.pop("anchor_item")
            anchor_vec = self._get_matching_features(features, mode, params, "anchor")
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'anchor_id': anchor_id,
                    'anchor_vec': anchor_vec,
                    'anchor_cate': features["anchor_cate"],
                    'anchor_commodity': features["anchor_commodity"]
                }
                export_outputs = {
                    'prediction': tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            higher_vec = self._get_matching_features(features, mode, params, "higher")
            lower_vec = self._get_matching_features(features, mode, params, "lower")
            higher_score = tf.keras.backend.batch_dot(anchor_vec, higher_vec, axes=(1, 1))
            lower_score = tf.keras.backend.batch_dot(anchor_vec, lower_vec, axes=(1, 1))

            vectors = []
            # sampling negative samples
            batch_size = params["batch_size"]
            shifts = random.sample(range(1, batch_size), params["num_negative_samples"])
            for rand in shifts:
                neg_vec = tf.concat([tf.slice(lower_vec, [rand, 0], [batch_size - rand, -1]),
                                     tf.slice(higher_vec, [0, 0], [rand, -1])], 0)
                vectors.append(neg_vec)

            # all_vectors's shape: (batch_size, num_negative_samples, vec_dim)
            neg_vectors = tf.stack(vectors, axis=1)
            # sim_scores's shape: (batch_size, num_negative_samples)
            neg_scores = tf.keras.backend.batch_dot(anchor_vec, neg_vectors, axes=(1, 2))  # cosine similarities

            max_neg_score = tf.reduce_max(neg_scores, axis=1, keep_dims=True)
            lower_hit = tf.greater(lower_score, max_neg_score)
            higher_hit = tf.logical_and(tf.greater(higher_score, lower_score), tf.greater(higher_score, max_neg_score))

            t = params["t"]
            new_higher_score = higher_score - params["margin"]
            cond1 = tf.greater_equal(new_higher_score - neg_scores, params["negative_margin"])
            cond2 = tf.greater_equal(lower_score - neg_scores, params["negative_margin"])
            cond = tf.logical_and(cond1, cond2)
            mask = tf.where(cond, tf.zeros_like(cond, tf.float32), tf.ones_like(cond, tf.float32))  # I_k
            new_neg_scores = mask * (neg_scores * t + t - 1) + (1 - mask) * neg_scores
            new_lower_score = tf.where(tf.greater_equal(new_higher_score, lower_score), lower_score,
                                       lower_score * t + t - 1)
            logits = tf.concat([new_higher_score, new_lower_score, new_neg_scores], axis=1)
            if 1.0 != params["smooth"]:
                logits *= params["smooth"]
            labels = tf.zeros([batch_size], dtype=tf.int32)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

            if params["l2_scale"] > 0:
                l2_loss = tf.losses.get_regularization_loss()
                tf.summary.scalar('loss/l2_loss', l2_loss)
                loss += l2_loss

            global_step = tf.train.get_or_create_global_step()
            if mode == tf.estimator.ModeKeys.EVAL:
                labels = tf.ones_like(higher_score, tf.bool)
                higher_hit_ratio = tf.metrics.accuracy(labels=labels, predictions=higher_hit)
                lower_hit_ratio = tf.metrics.accuracy(labels=labels, predictions=lower_hit)
                metrics = {"accuracy/higher": higher_hit_ratio, "accuracy/lower": lower_hit_ratio}
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            assert mode == tf.estimator.ModeKeys.TRAIN
            one = tf.ones_like(higher_score, tf.float32)
            zero = tf.zeros_like(higher_score, tf.float32)
            tf.summary.scalar('accuracy/higher', tf.reduce_mean(tf.where(higher_hit, one, zero)))
            tf.summary.scalar('accuracy/lower', tf.reduce_mean(tf.where(lower_hit, one, zero)))
            tf.summary.scalar('similarity/higher', tf.reduce_mean(higher_score))
            tf.summary.scalar('similarity/lower', tf.reduce_mean(lower_score))
            neg_similarity = tf.reduce_mean(tf.reduce_max(neg_scores, axis=1))
            tf.summary.scalar('similarity/negative', neg_similarity)

            # Create training op.
            if params['use_batch_norm']:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = self.optimizer.minimize(loss, global_step=global_step)
            else:
                train_op = self.optimizer.minimize(loss, global_step=global_step)

            train_scaffold = scaffold(params['init_checkpoint']) if params['warm_start'] else None
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=train_scaffold)

        super(SemanticModel, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config, params=params)

    def _get_matching_features(self, features, mode, params, prefix):
        xavier_initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(scale=params["l2_scale"]) if params["l2_scale"] > 0 else None
        with tf.variable_scope("seq_embeddings/", initializer=xavier_initializer, reuse=tf.AUTO_REUSE):
            initializer = params["word_initializer"] if "word_initializer" in params else xavier_initializer
            word_emb_matrix = tf.get_variable(name="word_embeddings", dtype=tf.float32, initializer=initializer,
                                              shape=[params["word_vocab_size"], params["word_embedding_size"]])
            char_initializer = params["char_initializer"] if "char_initializer" in params else xavier_initializer
            char_emb_matrix = tf.get_variable(name="char_embeddings", dtype=tf.float32, initializer=char_initializer,
                                              shape=[params["char_vocab_size"], params["char_embedding_size"]])
            tag_emb_matrix = tf.get_variable(name="tag_embeddings", dtype=tf.float32,
                                             shape=[params["tag_vocab_size"], params["tag_embedding_size"]])

        with tf.name_scope("word_sequence_encode"):
            word_seq = features[prefix + "_title_word"]
            tag_seq = features[prefix + "_title_word_tag"]
            weight_seq = features[prefix + "_title_word_weight"]
            word_emb = tf.nn.embedding_lookup(word_emb_matrix, word_seq)  # (batch_size, seq_len, embedding_size)
            tag_emb = tf.nn.embedding_lookup(tag_emb_matrix, tag_seq)
            word_info = tf.concat([word_emb, tag_emb, tf.expand_dims(weight_seq, -1)], axis=-1, name="word_info")
            word_seq_enc = self._encode(word_info, params, "word", regularizer)

        with tf.name_scope("char_sequence_encode"):
            char_seq = features[prefix + "_title_char"]
            char_info = tf.nn.embedding_lookup(char_emb_matrix, char_seq)
            char_seq_enc = self._encode(char_info, params, "char", regularizer)

        encoding = tf.concat([word_seq_enc, char_seq_enc], axis=-1, name="encoding")
        if params["use_feature"]:
            feature_columns = params[prefix + "_feature_columns"]
            side = tf.feature_column.input_layer(features, feature_columns)
            encoding = tf.concat([encoding, side], axis=-1, name="input_feature")
        net = self._add_fc_layers(encoding, mode, params)
        return tf.nn.l2_normalize(net, dim=-1, name=prefix + "_vector")

    def _encode(self, x, params, granularity, regularizer=None, reuse=tf.AUTO_REUSE):
        num_filters = params["word_cnn_num_filters"] if granularity == "word" else params["char_cnn_num_filters"]
        filter_sizes = params["word_cnn_filter_sizes"] if granularity == "word" else params["char_cnn_filter_sizes"]
        return textcnn(x, num_filters, filter_sizes=filter_sizes, scope_name=granularity + "cnn",
                       regularizer=regularizer, reuse=reuse)

    def _add_fc_layers(self, net, mode, params, regularizer=None):
        use_batch_norm = params['use_batch_norm']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        activation_dict = dict(relu=tf.nn.relu, tanh=tf.nn.tanh, sigmoid=tf.nn.sigmoid, elu=tf.nn.elu,
                               softplus=tf.nn.softplus, linear=tf.keras.activations.linear)
        # Build the hidden layers, sized according to the 'hidden_units' param.
        num_layers = len(params["hidden_units"])
        for i in range(num_layers):
            name = "hidden_layer_{}".format(i)
            units = params["hidden_units"][i]
            act_fn = activation_dict[params["activations"][i]]
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                if params["activations"][i] == 'relu':
                    initializer = tf.contrib.layers.variance_scaling_initializer()
                else:
                    initializer = tf.contrib.layers.xavier_initializer()
                if use_batch_norm and i < num_layers - 1:
                    x = tf.layers.dense(net, units=units, use_bias=False, name='fc', kernel_regularizer=regularizer,
                                        kernel_initializer=initializer, reuse=tf.AUTO_REUSE)
                    net = act_fn(tf.layers.batch_normalization(x, training=is_training, name='bn', reuse=tf.AUTO_REUSE))
                else:
                    if 'dropout_rate' in params and params["dropout_rate"] > 0.0:
                        net = tf.layers.dropout(net, params["dropout_rate"], training=is_training)
                    net = tf.layers.dense(net, units=units, activation=act_fn, name="dense",
                                          kernel_regularizer=regularizer,
                                          kernel_initializer=initializer, reuse=tf.AUTO_REUSE)
        return net
