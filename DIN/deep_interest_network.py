import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

class DIN(tf.estimator.Estimator):
  def __init__(self,
    params,
    model_dir=None,
    optimizer='Adagrad',
    config=None,
    warm_start_from=None,
  ):
    """ an implement of <<Deep Interest Network for Click-Through Rate Prediction>>"""
    if not optimizer: optimizer = 'Adagrad'
    self.optimizer = optimizers.get_optimizer_instance(optimizer, params["learning_rate"])

    def _model_fn(features, labels, mode, params):
      common = tf.feature_column.input_layer(features, params['feature_columns'])
      pid_vocab_size = params["vocab_size"]["product"]
      behaviorPids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorPids"]), pid_vocab_size)
      productId = tf.string_to_hash_bucket_fast(tf.as_string(features["productId"]), pid_vocab_size)
      good_emb, gid_emb = self.attention_layer(behaviorPids, productId, "product")

      bid_vocab_size = params["vocab_size"]["brand"]
      behaviorBids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorBids"]), bid_vocab_size)
      brandId = tf.string_to_hash_bucket_fast(tf.as_string(features["brandId"]), bid_vocab_size)
      brand_emb, bid_emb = self.attention_layer(behaviorBids, brandId, "brand")

      sid_vocab_size = params["vocab_size"]["seller"]
      behaviorSids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorSids"]), sid_vocab_size)
      sellerId = tf.string_to_hash_bucket_fast(tf.as_string(features["sellerId"]), sid_vocab_size)
      seller_emb, sid_emb = self.attention_layer(behaviorSids, sellerId, "seller")

      cid_vocab_size = params["vocab_size"]["cate"]
      behaviorCids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorCids"]), cid_vocab_size)
      cateId = tf.string_to_hash_bucket_fast(tf.as_string(features["cateId"]), cid_vocab_size)
      cate_emb, cid_emb = self.attention_layer(behaviorCids, cateId, "cate")

      c1id_vocab_size = params["vocab_size"]["cate1"]
      behaviorC1ids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorC1ids"]), c1id_vocab_size)
      cate1Id = tf.string_to_hash_bucket_fast(tf.as_string(features["cate1Id"]), c1id_vocab_size)
      cate1_emb, c1id_emb = self.attention_layer(behaviorC1ids, cate1Id, "cate1")

      net = tf.concat([common, good_emb, cate1_emb, cate_emb, brand_emb, seller_emb,
                       gid_emb, c1id_emb, cid_emb, bid_emb, sid_emb], axis=1)
      for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
          net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
      my_head = tf.contrib.estimator.binary_classification_head(thresholds=[0.5])
      logits = tf.layers.dense(net, units=my_head.logits_dimension)
      return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: self.optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )

    super(DIN, self).__init__(
      model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

  def attention_layer(self, seq_ids, tid, id_type):
    with tf.variable_scope("attention_" + id_type):
      embedding_size = self._params["embedding_size"][id_type]
      embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                   shape=[self._params["vocab_size"][id_type], embedding_size])
      seq_emb = tf.nn.embedding_lookup(embeddings, seq_ids)  # shape(batch_size, max_seq_len, embedding_size)
      tid_emb = tf.nn.embedding_lookup(embeddings, tid)  # shape(batch_size, embedding_size)
      max_seq_len = tf.shape(seq_ids)[1] # padded_dim
      u_emb = tf.reshape(seq_emb, shape=[-1, embedding_size])
      a_emb = tf.reshape(tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, embedding_size])
      net = tf.concat([u_emb, u_emb - a_emb, a_emb], axis=1)
      for units in self._params['attention_hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
      att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
      att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")
      wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
      #masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
      masks = tf.expand_dims(tf.cast(seq_ids >= 0, tf.float32), axis=-1)
      att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
      return att_emb, tid_emb
