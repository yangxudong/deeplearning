import tensorflow as tf

def get_behavior_embedding(params, features):
  # this function can be implemented with tf.contrib.feature_column.sequence_input_layer
  with tf.variable_scope("behavior_embedding"):
    pid_vocab_size = params["pid_vocab_size"]
    behaviorPids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorPids"]), pid_vocab_size)
    productId = tf.string_to_hash_bucket_fast(tf.as_string(features["productId"]), pid_vocab_size)
    pid_embeddings = tf.get_variable(name="pid_embeddings", dtype=tf.float32,
                                     shape=[pid_vocab_size, params["pid_embedding_size"]])
    pids = tf.nn.embedding_lookup(pid_embeddings, behaviorPids)  # shape(batch_size, max_seq_len, embedding_size)
    pid = tf.nn.embedding_lookup(pid_embeddings, productId)

    bid_vocab_size = params["bid_vocab_size"]
    behaviorBids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorBids"]), bid_vocab_size)
    brandId = tf.string_to_hash_bucket_fast(tf.as_string(features["brandId"]), bid_vocab_size)
    bid_embeddings = tf.get_variable(name="bid_embeddings", dtype=tf.float32,
                                     shape=[bid_vocab_size, params["bid_embedding_size"]])
    bids = tf.nn.embedding_lookup(bid_embeddings, behaviorBids)  # shape(batch_size, max_seq_len, embedding_size)
    bid = tf.nn.embedding_lookup(bid_embeddings, brandId)

    sid_vocab_size = params["sid_vocab_size"]
    behaviorSids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorSids"]), sid_vocab_size)
    sellerId = tf.string_to_hash_bucket_fast(tf.as_string(features["sellerId"]), sid_vocab_size)
    sid_embeddings = tf.get_variable(name="sid_embeddings", dtype=tf.float32,
                                     shape=[sid_vocab_size, params["sid_embedding_size"]])
    sids = tf.nn.embedding_lookup(sid_embeddings, behaviorSids)  # shape(batch_size, max_seq_len, embedding_size)
    sid = tf.nn.embedding_lookup(sid_embeddings, sellerId)

    cid_vocab_size = params["cid_vocab_size"]
    behaviorCids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorCids"]), cid_vocab_size)
    cateId = tf.string_to_hash_bucket_fast(tf.as_string(features["cateId"]), bid_vocab_size)
    cid_embeddings = tf.get_variable(name="cid_embeddings", dtype=tf.float32,
                                     shape=[cid_vocab_size, params["cid_embedding_size"]])
    cids = tf.nn.embedding_lookup(cid_embeddings, behaviorCids)  # shape(batch_size, max_seq_len, embedding_size)
    cid = tf.nn.embedding_lookup(cid_embeddings, cateId)

    c1id_vocab_size = params["c1id_vocab_size"]
    behaviorC1ids = tf.string_to_hash_bucket_fast(tf.as_string(features["behaviorC1ids"]), c1id_vocab_size)
    cate1Id = tf.string_to_hash_bucket_fast(tf.as_string(features["cate1Id"]), c1id_vocab_size)
    c1id_embeddings = tf.get_variable(name="c1id_embeddings", dtype=tf.float32,
                                      shape=[c1id_vocab_size, params["c1id_embedding_size"]])
    c1ids = tf.nn.embedding_lookup(c1id_embeddings, behaviorC1ids)  # shape(batch_size, max_seq_len, embedding_size)
    c1id = tf.nn.embedding_lookup(c1id_embeddings, cate1Id)

    item_emb = tf.concat([pid, bid, sid, cid, c1id], -1)  # shape(batch_size, embedding_size)

    behavior_hour = tf.one_hot(features["behaviorTimeBucket"], 8, dtype=tf.float32)  # shape: (batch_size, seq_len, 8)
    behavior_type = tf.one_hot(features["behaviorTypes"], 5, dtype=tf.float32)
    behavior_weekend = tf.expand_dims(tf.to_float(features["behaviorTimeIsWeekend"]), -1)
    behavior_weight = tf.expand_dims(features["behaviorTimeWeight"], -1)
    behavior_scenario = tf.expand_dims(tf.to_float(features["behaviorScenarios"]), -1)
    behvr_columns = [pids, bids, sids, cids, c1ids,
                     behavior_hour, behavior_weekend, behavior_weight, behavior_type, behavior_scenario]
    behvr_emb = tf.concat(behvr_columns, -1)
    return behvr_emb, item_emb


def attention(inputs, context, params, masks):
  with tf.variable_scope("attention_layer"):
    shape = inputs.shape.as_list()
    max_seq_len = shape[1]  # padded_dim
    embedding_size = shape[2]
    seq_emb = tf.reshape(inputs, shape=[-1, embedding_size])  # shape(batch_size * max_seq_len, embedding_size)
    ctx_len = context.shape.as_list()[1]
    ctx_emb = tf.reshape(tf.tile(context, [1, max_seq_len]), shape=[-1, ctx_len])
    print("seq_emb shape:", seq_emb.shape)
    print("ctx_emb shape:", ctx_emb.shape)
    net = tf.concat([seq_emb, ctx_emb], axis=1)
    print("attention input shape:", net.shape)
    print("attention_hidden_units:", params['attention_hidden_units'])
    for units in params['attention_hidden_units']:
      net = tf.layers.dense(net, units=int(units), activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)
    att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")  # shape(batch_size, max_seq_len, 1)
    wgt_emb = tf.multiply(inputs, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
    # masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.expand_dims(masks, axis=-1)
    att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
    return att_emb


def dupn_logit_fn(features, mode, params):
  behavior_embedding_collection = tf.get_collection("behavior_embedding")
  if behavior_embedding_collection:
    behvr_emb, item_emb = behavior_embedding_collection
  else:
    behvr_emb, item_emb = get_behavior_embedding(params, features)
    tf.add_to_collection("behavior_embedding", behvr_emb)
    tf.add_to_collection("behavior_embedding", item_emb)
  print("behvr_emb shape:", behvr_emb.shape)
  print("item_emb shape:", item_emb.shape)

  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=params["num_units"])
  #initial_state = lstm_cell.zero_state(params["batch_size"], tf.float32) # this will cause failure when serving
  outputs, state = tf.nn.dynamic_rnn(lstm_cell, behvr_emb, dtype=tf.float32)
  print("lstm output shape:", outputs.shape)

  masks = tf.cast(features["behaviorPids"] >= 0, tf.float32)
  user = tf.feature_column.input_layer(features, params["user_feature_columns"])
  context = tf.concat([user, item_emb], -1)
  print("attention context shape:", context.shape)
  sequence = attention(outputs, context, params, masks)
  print("sequence embedding shape:", sequence.shape)

  other = tf.feature_column.input_layer(features, params["other_feature_columns"])
  net = tf.concat([sequence, item_emb, other], -1)
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=int(units), activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Compute logits
  logits = tf.layers.dense(net, 1, activation=None)
  return logits
