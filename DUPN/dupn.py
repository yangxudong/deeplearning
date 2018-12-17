#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.python.estimator.canned import optimizers
import os
import json
# for python 2.x
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_string("attention_hidden_units", "128,64", "Comma-separated list of number of units in each hidden layer of the attention NN")
flags.DEFINE_string("optimizer", "Adagrad", "optimizer method")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("num_units", 128, "state size of LSTM cell")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 4, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")

FLAGS = flags.FLAGS


def create_user_feature_columns():
  gender = fc.indicator_column(fc.categorical_column_with_identity("gender", num_buckets=3, default_value=0))
  age_class = fc.indicator_column(fc.categorical_column_with_identity("age_class", num_buckets=7, default_value=0))
  has_baby = fc.indicator_column(fc.categorical_column_with_identity("has_baby", num_buckets=2, default_value=0))
  baby_gender = fc.indicator_column(fc.categorical_column_with_identity("baby_gender", num_buckets=3, default_value=0))
  baby_age = fc.indicator_column(fc.categorical_column_with_identity("baby_age", num_buckets=7, default_value=0))
  grade = fc.indicator_column(fc.categorical_column_with_identity("grade", num_buckets=7, default_value=0))
  rfm_type = fc.indicator_column(fc.categorical_column_with_identity("bi_rfm_type", num_buckets=12, default_value=0))
  cate1_price_prefer = fc.indicator_column(fc.categorical_column_with_identity("cate1_price_prefer", num_buckets=6, default_value=0))
  cate2_price_prefer = fc.indicator_column(fc.categorical_column_with_identity("cate2_price_prefer", num_buckets=6, default_value=0))
  cate3_price_prefer = fc.indicator_column(fc.categorical_column_with_identity("cate3_price_prefer", num_buckets=6, default_value=0))
  city_id = fc.categorical_column_with_hash_bucket("city", 700)
  city = fc.shared_embedding_columns([city_id], 16)
  cols = [gender, age_class, has_baby, baby_gender, baby_age, grade, rfm_type, cate1_price_prefer, cate2_price_prefer, cate3_price_prefer]
  return cols + city

def create_feature_columns():
  c2id = fc.categorical_column_with_hash_bucket("cate2Id", 5000, dtype=tf.int64)
  modified_time = fc.numeric_column("modified_time", default_value=0.0)
  modified_time_sqrt = fc.numeric_column("modified_time_sqrt", default_value=0.0)
  modified_time_square = fc.numeric_column("modified_time_square", default_value=0.0)
  props_sex = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("props_sex", ["男", "女", "通用", "情侣"], default_value=0))
  brand_grade = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("brand_grade", ["A类品牌", "B类品牌", "C类品牌", "D类品牌"], default_value=0))
  shipment_rate = fc.numeric_column("shipment_rate", default_value=0.0)
  shipping_rate = fc.numeric_column("shipping_rate", default_value=0.0)
  ipv_ntile = fc.bucketized_column(fc.numeric_column("ipv_ntile", dtype=tf.int64, default_value=99), boundaries=[1, 2, 3, 4, 5, 10, 20, 50, 80])
  pay_ntile = fc.bucketized_column(fc.numeric_column("pay_ntile", dtype=tf.int64, default_value=99), boundaries=[1, 2, 3, 4, 5, 10, 20, 50, 80])
  price = fc.numeric_column("price_norm", default_value=0.0)
  ctr_1d = fc.numeric_column("ctr_1d", default_value=0.0)
  cvr_1d = fc.numeric_column("cvr_1d", default_value=0.0)
  uv_cvr_1d = fc.numeric_column("uv_cvr_1d", default_value=0.0)
  ctr_1w = fc.numeric_column("ctr_1w", default_value=0.0)
  cvr_1w = fc.numeric_column("cvr_1w", default_value=0.0)
  uv_cvr_1w = fc.numeric_column("uv_cvr_1w", default_value=0.0)
  ctr_2w = fc.numeric_column("ctr_2w", default_value=0.0)
  cvr_2w = fc.numeric_column("cvr_2w", default_value=0.0)
  uv_cvr_2w = fc.numeric_column("uv_cvr_2w", default_value=0.0)
  ctr_1m = fc.numeric_column("ctr_1m", default_value=0.0)
  cvr_1m = fc.numeric_column("cvr_1m", default_value=0.0)
  uv_cvr_1m = fc.numeric_column("uv_cvr_1m", default_value=0.0)
  pay_qty_1d = fc.numeric_column("pay_qty_1d", default_value=0.0)
  pay_qty_1w = fc.numeric_column("pay_qty_1w", default_value=0.0)
  pay_qty_2w = fc.numeric_column("pay_qty_2w", default_value=0.0)
  pay_qty_1m = fc.numeric_column("pay_qty_1m", default_value=0.0)
  cat2_pay_qty = fc.numeric_column("cat2_pay_qty_1d", default_value=0.0)
  cat1_pay_qty = fc.numeric_column("cat1_pay_qty_1d", default_value=0.0)
  brd_pay_qty = fc.numeric_column("brd_pay_qty_1d", default_value=0.0)
  slr_pay_qty_1d = fc.numeric_column("slr_pay_qty_1d", default_value=0.0)
  slr_pay_qty_1w = fc.numeric_column("slr_pay_qty_1w", default_value=0.0)
  slr_pay_qty_2w = fc.numeric_column("slr_pay_qty_2w", default_value=0.0)
  slr_pay_qty_1m = fc.numeric_column("slr_pay_qty_1m", default_value=0.0)
  slr_brd_pay_qty_1d = fc.numeric_column("slr_brd_pay_qty_1d", default_value=0.0)
  slr_brd_pay_qty_1w = fc.numeric_column("slr_brd_pay_qty_1w", default_value=0.0)
  slr_brd_pay_qty_2w = fc.numeric_column("slr_brd_pay_qty_2w", default_value=0.0)
  slr_brd_pay_qty_1m = fc.numeric_column("slr_brd_pay_qty_1m", default_value=0.0)
  weighted_ipv = fc.numeric_column("weighted_ipv", default_value=0.0)
  cat1_weighted_ipv = fc.numeric_column("cat1_weighted_ipv", default_value=0.0)
  cate_weighted_ipv = fc.numeric_column("cate_weighted_ipv", default_value=0.0)
  slr_weighted_ipv = fc.numeric_column("slr_weighted_ipv", default_value=0.0)
  brd_weighted_ipv = fc.numeric_column("brd_weighted_ipv", default_value=0.0)
  cms_scale = fc.numeric_column("cms_scale", default_value=0.0)
  cms_scale_sqrt = fc.numeric_column("cms_scale_sqrt", default_value=0.0)

  # context feature
  matchScore = fc.numeric_column("matchScore", default_value=0.0)
  popScore = fc.numeric_column("popScore", default_value=0.0)
  brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0)
  cate2Prefer = fc.numeric_column("cate2Prefer", default_value=0.0)
  catePrefer = fc.numeric_column("catePrefer", default_value=0.0)
  sellerPrefer = fc.numeric_column("sellerPrefer", default_value=0.0)
  matchType = fc.indicator_column(fc.categorical_column_with_identity("matchType", 9, default_value=0))
  position = fc.bucketized_column(fc.numeric_column("position", dtype=tf.int64, default_value=301),
    boundaries=[1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 20, 30, 40, 50, 80, 100, 150, 200, 300])
  triggerNum = fc.indicator_column(fc.categorical_column_with_identity("triggerNum", 41, default_value=40))
  triggerRank = fc.indicator_column(fc.categorical_column_with_identity("triggerRank", 41, default_value=40))
  sceneType = fc.indicator_column(fc.categorical_column_with_identity("type", 2, default_value=0))
  hour = fc.indicator_column(fc.categorical_column_with_identity("hour", 24, default_value=0))
  phoneBrandId = fc.categorical_column_with_hash_bucket("phoneBrand", 1000)
  phoneBrand = fc.shared_embedding_columns([phoneBrandId], 20)
  phoneResolutionId = fc.categorical_column_with_hash_bucket("phoneResolution", 500)
  phoneResolution = fc.shared_embedding_columns([phoneResolutionId], 10)
  phoneOs = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0))
  tab = fc.indicator_column(fc.categorical_column_with_vocabulary_list("tab",
        ["ALL", "TongZhuang", "XieBao", "MuYing", "NvZhuang", "MeiZhuang", "JuJia", "MeiShi"], default_value=0))

  c2id_embed = fc.shared_embedding_columns([c2id], 16, shared_embedding_collection_name="c2id")
  feature_columns = [matchScore, matchType, position, triggerNum, triggerRank, sceneType, hour,
    phoneOs, tab, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer,
    price, props_sex, brand_grade, modified_time, modified_time_sqrt, modified_time_square,
    shipment_rate, shipping_rate, ipv_ntile, pay_ntile, uv_cvr_1d, uv_cvr_1w, uv_cvr_2w, uv_cvr_1m,
    ctr_1d, ctr_1w, ctr_2w, ctr_1m, cvr_1d, cvr_1w, cvr_2w, cvr_1m,
    pay_qty_1d, pay_qty_1w, pay_qty_2w, pay_qty_1m, cat2_pay_qty, cat1_pay_qty, brd_pay_qty,
    slr_pay_qty_1d, slr_pay_qty_1w, slr_pay_qty_2w, slr_pay_qty_1m,
    slr_brd_pay_qty_1d, slr_brd_pay_qty_1w, slr_brd_pay_qty_2w, slr_brd_pay_qty_1m,
    weighted_ipv, cat1_weighted_ipv, cate_weighted_ipv, slr_weighted_ipv, brd_weighted_ipv,
    cms_scale, cms_scale_sqrt]
  feature_columns += c2id_embed
  feature_columns += phoneResolution
  feature_columns += phoneBrand
  return feature_columns


def parse_exmp(serial_exmp, feature_spec):
  share = fc.numeric_column("share", default_value=0, dtype=tf.int64)
  userType = fc.numeric_column("user_type", default_value=0, dtype=tf.int64)
  fea_columns = [share, userType]
  feature_spec.update(tf.feature_column.make_parse_example_spec(fea_columns))
  feats = tf.parse_single_example(serial_exmp, features=feature_spec)
  feats["modified_time_sqrt"] = tf.sqrt(feats["modified_time"])
  feats["modified_time_square"] = tf.square(feats["modified_time"])
  feats["cms_scale_sqrt"] = tf.sqrt(feats["cms_scale"])
  share = feats.pop('share')
  return feats, share


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size, num_parallel_readers):
  #dataset = tf.data.TFRecordDataset(filenames)
  files = tf.data.Dataset.list_files(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
  dataset = dataset.map(lambda x: parse_exmp(x, feature_spec), num_parallel_calls=8)
  dataset = dataset.filter(lambda x, y: tf.not_equal(tf.unstack(x["user_type"])[0], 0)) # only keep seller user samples
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat().batch(batch_size).prefetch(1)
  #print(dataset.output_types)
  #print(dataset.output_shapes)
  # Return the read end of the pipeline.
  return dataset


def eval_input_fn(filename, feature_spec, batch_size):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(lambda x: parse_exmp(x, feature_spec), num_parallel_calls=8)
  dataset = dataset.filter(lambda x, y: tf.not_equal(tf.unstack(x["user_type"])[0], 0)) # only keep seller user samples
  # Shuffle, repeat, and batch the examples.
  #dataset = dataset.batch(batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # Return the read end of the pipeline.
  return dataset


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

    item_emb = tf.concat([pid, bid, sid, cid, c1id], -1) # shape(batch_size, embedding_size)
    behvr_emb = tf.concat([pids, bids, sids, cids, c1ids], -1)  # shape(batch_size, max_seq_len, embedding_size)

    behavior_hour = tf.one_hot(features["behaviorTimeBucket"], 8, dtype=tf.float32)  # shape: (batch_size, seq_len, 8)
    behavior_type = tf.one_hot(features["behaviorTypes"], 5, dtype=tf.float32)
    behavior_weekend = tf.expand_dims(tf.to_float(features["behaviorTimeIsWeekend"]), -1)
    behavior_weight = tf.expand_dims(features["behaviorTimeWeight"], -1)
    behavior_scenario = tf.expand_dims(tf.to_float(features["behaviorScenarios"]), -1)
    property_emb = tf.concat([behavior_hour, behavior_weekend, behavior_weight, behavior_type, behavior_scenario], -1)
    return behvr_emb, property_emb, item_emb


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


def dupn_model_fn(features, labels, mode, params):
  behvr_emb, property_emb, item_emb = get_behavior_embedding(params, features)
  print("behvr_emb shape:", behvr_emb.shape)
  print("property_emb shape:", property_emb.shape)
  print("item_emb shape:", item_emb.shape)

  inputs = tf.concat([behvr_emb, property_emb], -1)
  print("lstm inputs shape:", inputs.shape)
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=params["num_units"])
  #initial_state = lstm_cell.zero_state(params["batch_size"], tf.float32)
  outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
  print("lstm output shape:", outputs.shape)

  masks = tf.cast(features["behaviorPids"] >= 0, tf.float32)
  user = fc.input_layer(features, params["user_feature_columns"])
  context = tf.concat([user, item_emb], -1)
  print("attention context shape:", context.shape)
  sequence = attention(outputs, context, params, masks)
  print("sequence embedding shape:", sequence.shape)

  other = fc.input_layer(features, params["other_feature_columns"])
  net = tf.concat([sequence, item_emb, other], -1)
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=int(units), activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Compute logits
  logits = tf.layers.dense(net, 1, activation=None)

  optimizer = optimizers.get_optimizer_instance(params["optimizer"], params["learning_rate"])
  my_head = tf.contrib.estimator.binary_classification_head(thresholds=[0.5])
  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
  )



def set_tfconfig_environ():
  if "TF_CLUSTER_DEF" in os.environ:
    cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]

    tf_config = dict()
    worker_num = len(cluster["worker"])
    if task_type == "ps":
      tf_config["task"] = {"index": task_index, "type": task_type}
      FLAGS.job_name = "ps"
      FLAGS.task_index = task_index
    else:
      if task_index == 0:
        tf_config["task"] = {"index": 0, "type": "chief"}
      else:
        tf_config["task"] = {"index": task_index - 1, "type": task_type}
      FLAGS.job_name = "worker"
      FLAGS.task_index = task_index

    if worker_num == 1:
      cluster["chief"] = cluster["worker"]
      del cluster["worker"]
    else:
      cluster["chief"] = [cluster["worker"][0]]
      del cluster["worker"][0]

    tf_config["cluster"] = cluster
    os.environ["TF_CONFIG"] = json.dumps(tf_config)
    print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

  if "INPUT_FILE_LIST" in os.environ:
    INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    if INPUT_PATH:
      print("input path:", INPUT_PATH)
      FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
      FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
    else:  # for ps
      print("load input path failed.")
      FLAGS.train_data = None
      FLAGS.eval_data = None


def parse_argument():
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  os.environ["TF_ROLE"] = FLAGS.job_name
  os.environ["TF_INDEX"] = str(FLAGS.task_index)

  # Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  cluster = {"worker": worker_spec, "ps": ps_spec}
  os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def main(unused_argv):
  set_tfconfig_environ()
  user_feature_columns = create_user_feature_columns()
  other_feature_columns = create_feature_columns()
  classifier = tf.estimator.Estimator(
    model_fn=dupn_model_fn,
    params={
      'user_feature_columns': user_feature_columns,
      'other_feature_columns': other_feature_columns,
      'optimizer': FLAGS.optimizer,
      'hidden_units': FLAGS.hidden_units.split(','),
      'attention_hidden_units': FLAGS.attention_hidden_units.split(','),
      'learning_rate': FLAGS.learning_rate,
      'dropout_rate': FLAGS.dropout_rate,
      'num_units': FLAGS.num_units,
      'batch_size': FLAGS.batch_size,
      'pid_vocab_size': 150000,
      'pid_embedding_size': 48,
      'bid_vocab_size': 10000,
      'bid_embedding_size': 16,
      'sid_vocab_size': 10000,
      'sid_embedding_size': 16,
      'cid_vocab_size': 10000,
      'cid_embedding_size': 16,
      'c1id_vocab_size': 100,
      'c1id_embedding_size': 8,
    },
    config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  )
  batch_size = FLAGS.batch_size
  print("train steps:", FLAGS.train_steps, "batch_size:", batch_size)
  if isinstance(FLAGS.train_data, str) and os.path.isdir(FLAGS.train_data):
    train_files = [FLAGS.train_data + '/' + x for x in os.listdir(FLAGS.train_data)] if os.path.isdir(
      FLAGS.train_data) else FLAGS.train_data
  else:
    train_files = FLAGS.train_data
  if isinstance(FLAGS.eval_data, str) and os.path.isdir(FLAGS.eval_data):
    eval_files = [FLAGS.eval_data + '/' + x for x in os.listdir(FLAGS.eval_data)] if os.path.isdir(
      FLAGS.eval_data) else FLAGS.eval_data
  else:
    eval_files = FLAGS.eval_data
  shuffle_buffer_size = FLAGS.shuffle_buffer_size
  print("train_data:", train_files)
  print("eval_data:", eval_files)
  print("shuffle_buffer_size:", shuffle_buffer_size)

  feature_spec = tf.feature_column.make_parse_example_spec(user_feature_columns + other_feature_columns)
  other_feature_spec = {
    "behaviorBids": tf.FixedLenFeature([60], tf.int64),
    "behaviorCids": tf.FixedLenFeature([60], tf.int64),
    "behaviorC1ids": tf.FixedLenFeature([60], tf.int64),
    "behaviorSids": tf.FixedLenFeature([60], tf.int64),
    "behaviorPids": tf.FixedLenFeature([60], tf.int64),
    "behaviorTypes": tf.FixedLenFeature([60], tf.int64),
    "behaviorTimeBucket": tf.FixedLenFeature([60], tf.int64),
    "behaviorTimeIsWeekend": tf.FixedLenFeature([60], tf.int64),
    "behaviorTimeWeight": tf.FixedLenFeature([60], tf.float32),
    "behaviorScenarios": tf.FixedLenFeature([60], tf.int64),
    "productId": tf.FixedLenFeature([], tf.int64),
    "sellerId": tf.FixedLenFeature([], tf.int64),
    "brandId": tf.FixedLenFeature([], tf.int64),
    "cate1Id": tf.FixedLenFeature([], tf.int64),
    "cateId": tf.FixedLenFeature([], tf.int64)
  }
  feature_spec.update(other_feature_spec)
  #print(feature_spec)

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input_fn(train_files, feature_spec, batch_size, shuffle_buffer_size, FLAGS.num_parallel_readers),
    max_steps=FLAGS.train_steps
  )
  input_fn_for_eval = lambda: eval_input_fn(eval_files, feature_spec, batch_size)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300, steps=None)

  print("before train and evaluate")
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  print("after train and evaluate")

  # Evaluate accuracy.
  results = classifier.evaluate(input_fn=input_fn_for_eval)
  for key in sorted(results): print('%s: %s' % (key, results[key]))
  print("after evaluate")

  if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
  print("quit main")


if __name__ == "__main__":
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  if FLAGS.run_on_cluster:
    parse_argument()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
