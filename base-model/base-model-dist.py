from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as fc
# for python 2.x
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_string("input_format", "tfrecord", "the format(tfrecord/text) of the training & evaluation data.")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                                    "If you don't use GPU, please set it to '0'")
flags.DEFINE_string("hidden_units", "512,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 10000, "Save checkpoints every this many steps")
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


class DebugHook(tf.train.SessionRunHook):
  def begin(self):
    # You can add ops to the graph here.
    print('Before starting the session.')

  def after_create_session(self, session, coord):
    # When this is called, the graph is finalized and ops can no longer be added to the graph.
    print('Session created.')

  def before_run(self, run_context):
    # print('Before calling session.run().')
    return None  # SessionRunArgs(self.your_tensor)

  def after_run(self, run_context, run_values):
    # print('Done running one step. The value of my tensor: %s', run_values.results)
    # if you-need-to-stop-loop:
    #  run_context.request_stop()
    pass

  def end(self, session):
    print('Done with the session.')


def create_feature_columns():
  # user feature
  bids = fc.categorical_column_with_hash_bucket("behaviorBids", 10240, dtype=tf.int64)
  c1ids = fc.categorical_column_with_hash_bucket("behaviorC1ids", 100, dtype=tf.int64)
  cids = fc.categorical_column_with_hash_bucket("behaviorCids", 10240, dtype=tf.int64)
  sids = fc.categorical_column_with_hash_bucket("behaviorSids", 10240, dtype=tf.int64)
  pids = fc.categorical_column_with_hash_bucket("behaviorPids", 1000000, dtype=tf.int64)

  # item feature
  pid = fc.categorical_column_with_hash_bucket("productId", 1000000, dtype=tf.int64)
  sid = fc.categorical_column_with_hash_bucket("sellerId", 10240, dtype=tf.int64)
  bid = fc.categorical_column_with_hash_bucket("brandId", 10240, dtype=tf.int64)
  c1id = fc.categorical_column_with_hash_bucket("cate1Id", 100, dtype=tf.int64)
  cid = fc.categorical_column_with_hash_bucket("cateId", 10240, dtype=tf.int64)

  # context feature
  matchScore = fc.numeric_column("matchScore", default_value=0.0)
  popScore = fc.numeric_column("popScore", default_value=0.0)
  brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0)
  cate2Prefer = fc.numeric_column("cate2Prefer", default_value=0.0)
  catePrefer = fc.numeric_column("catePrefer", default_value=0.0)
  sellerPrefer = fc.numeric_column("sellerPrefer", default_value=0.0)
  matchType = fc.indicator_column(fc.categorical_column_with_identity("matchType", 9, default_value=0))
  postition = fc.indicator_column(fc.categorical_column_with_identity("position", 201, default_value=200))
  triggerNum = fc.indicator_column(fc.categorical_column_with_identity("triggerNum", 51, default_value=50))
  triggerRank = fc.indicator_column(fc.categorical_column_with_identity("triggerRank", 51, default_value=50))
  sceneType = fc.indicator_column(fc.categorical_column_with_identity("type", 2, default_value=0))
  hour = fc.indicator_column(fc.categorical_column_with_identity("hour", 24, default_value=0))
  phoneBrand = fc.indicator_column(fc.categorical_column_with_hash_bucket("phoneBrand", 1000))
  phoneResolution = fc.indicator_column(fc.categorical_column_with_hash_bucket("phoneResolution", 500))
  phoneOs = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0))
  tab = fc.indicator_column(fc.categorical_column_with_vocabulary_list("tab",
                                                                       ["ALL", "TongZhuang", "XieBao", "MuYing",
                                                                        "NvZhuang", "MeiZhuang", "JuJia", "MeiShi"],
                                                                       default_value=0))

  pid_embed = fc.shared_embedding_columns([pids, pid], 32, combiner='sum', shared_embedding_collection_name="pid")
  bid_embed = fc.shared_embedding_columns([bids, bid], 16, combiner='sum', shared_embedding_collection_name="bid")
  cid_embed = fc.shared_embedding_columns([cids, cid], 16, combiner='sum', shared_embedding_collection_name="cid")
  c1id_embed = fc.shared_embedding_columns([c1ids, c1id], 8, combiner='sum', shared_embedding_collection_name="c1id")
  sid_embed = fc.shared_embedding_columns([sids, sid], 16, combiner='sum', shared_embedding_collection_name="sid")
  columns = [matchScore, matchType, postition, triggerNum, triggerRank, sceneType, hour, phoneBrand, phoneResolution,
             phoneOs, tab, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer]
  columns += pid_embed
  columns += sid_embed
  columns += bid_embed
  columns += cid_embed
  columns += c1id_embed
  print("feature columns:", columns)
  return columns


def parse_exmp(serial_exmp):
  features = {
    "click": tf.FixedLenFeature([], tf.int64),
    "behaviorBids": tf.FixedLenFeature([20], tf.int64),
    "behaviorCids": tf.FixedLenFeature([20], tf.int64),
    "behaviorC1ids": tf.FixedLenFeature([10], tf.int64),
    "behaviorSids": tf.FixedLenFeature([20], tf.int64),
    "behaviorPids": tf.FixedLenFeature([20], tf.int64),
    "productId": tf.FixedLenFeature([], tf.int64),
    "sellerId": tf.FixedLenFeature([], tf.int64),
    "brandId": tf.FixedLenFeature([], tf.int64),
    "cate1Id": tf.FixedLenFeature([], tf.int64),
    "cateId": tf.FixedLenFeature([], tf.int64),
    "tab": tf.FixedLenFeature([], tf.string),
    "matchType": tf.FixedLenFeature([], tf.int64),
    "position": tf.FixedLenFeature([], tf.int64),
    "type": tf.FixedLenFeature([], tf.int64),
    "triggerNum": tf.FixedLenFeature([], tf.int64),
    "triggerRank": tf.FixedLenFeature([], tf.int64),
    "hour": tf.FixedLenFeature([], tf.int64),
    "matchScore": tf.FixedLenFeature([], tf.float32),
    "popScore": tf.FixedLenFeature([], tf.float32),
    "brandPrefer": tf.FixedLenFeature([], tf.float32),
    "sellerPrefer": tf.FixedLenFeature([], tf.float32),
    "catePrefer": tf.FixedLenFeature([], tf.float32),
    "cate2Prefer": tf.FixedLenFeature([], tf.float32),
    "phoneBrand": tf.FixedLenFeature([], tf.string),
    "phoneOs": tf.FixedLenFeature([], tf.string),
    "phoneResolution": tf.FixedLenFeature([], tf.string)
  }
  feats = tf.parse_single_example(serial_exmp, features=features)
  labels = feats.pop('click')
  return feats, labels


def filter_func(line):
  fields = line.decode().split("\t")
  if len(fields) < 8:
    return False
  for field in fields:
    if not field:
      return False
  return True


def parse_line(line):
  _COLUMNS = ["sellerId", "brandId", "cate1Id", "cateId", "tab"]
  _INT_COLUMNS = ["click", "productId", "matchType", "position", "type", "triggerNum", "triggerRank", "hour"]
  _FLOAT_COLUMNS = ["matchScore", "popScore", "brandPrefer", "sellerPrefer", "catePrefer", "cate2Prefer"]
  _STRING_COLUMNS = ["phoneResolution", "phoneBrand", "phoneOs"]
  _SEQ_COLUMNS = ["behaviorC1ids", "behaviorBids", "behaviorCids", "behaviorSids", "behaviorPids"]

  def get_content(record):
    import datetime
    fields = record.decode().split("\t")
    if len(fields) < 8:
      raise ValueError("invalid record %s" % record)
    for field in fields:
      if not field:
        raise ValueError("invalid record %s" % record)
    fea = json.loads(fields[1])
    if fea["time"]:
      dt = datetime.datetime.fromtimestamp(fea["time"])
      fea["hour"] = dt.hour
    else:
      fea["hour"] = 0
    seq_len = 10
    for x in _SEQ_COLUMNS:
      sequence = fea.setdefault(x, [])
      n = len(sequence)
      if n < seq_len:
        sequence.extend([-1] * (seq_len - n))
      elif n > seq_len:
        fea[x] = sequence[:seq_len]
      seq_len = 20

    elems = [np.int64(fields[2]), np.int64(fields[3]), np.int64(fields[4]), np.int64(fields[6]), fields[7]]
    elems += [np.int64(fea.get(x, 0)) for x in _INT_COLUMNS]
    elems += [np.float32(fea.get(x, 0.0)) for x in _FLOAT_COLUMNS]
    elems += [fea.get(x, "") for x in _STRING_COLUMNS]
    elems += [np.int64(fea[x]) for x in _SEQ_COLUMNS]
    return elems

  out_type = [tf.int64] * 4 + [tf.string] + [tf.int64] * len(_INT_COLUMNS) + [tf.float32] * len(_FLOAT_COLUMNS) + [
    tf.string] * len(_STRING_COLUMNS) + [tf.int64] * len(_SEQ_COLUMNS)
  result = tf.py_func(get_content, [line], out_type)
  n = len(result) - len(_SEQ_COLUMNS)
  for i in range(n):
    result[i].set_shape([])
  result[n].set_shape([10])
  for i in range(n + 1, len(result)):
    result[i].set_shape([20])
  columns = _COLUMNS + _INT_COLUMNS + _FLOAT_COLUMNS + _STRING_COLUMNS + _SEQ_COLUMNS
  features = dict(zip(columns, result))
  labels = features.pop('click')
  return features, labels


def my_input_fn(filenames, batch_size, shuffle_buffer_size):
  print("start my_input_fn")
  dataset = tf.data.TextLineDataset(filenames)
  dataset = dataset.filter(lambda x: tf.py_func(filter_func, [x], tf.bool, False))
  dataset = dataset.map(parse_line, num_parallel_calls=8)
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat().batch(batch_size)
  print(dataset.output_types)
  print(dataset.output_shapes)
  # Return the read end of the pipeline.
  return dataset


def train_input_fn(filenames, batch_size, shuffle_buffer_size):
  #dataset = tf.data.TFRecordDataset(filenames)
  files = tf.data.Dataset.list_files(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  #dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_exmp, batch_size=batch_size))
  #dataset = dataset.repeat().prefetch(1)
  dataset = dataset.map(parse_exmp, num_parallel_calls=8)
  dataset = dataset.repeat().batch(batch_size).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  # Return the read end of the pipeline.
  return dataset


def eval_input_fn(filename, batch_size):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(parse_exmp, num_parallel_calls=8)
  # Shuffle, repeat, and batch the examples.
  dataset = dataset.batch(batch_size)
  # Return the read end of the pipeline.
  return dataset


def my_model(features, labels, mode, params):
  net = fc.input_layer(features, params['feature_columns'])
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
  my_head = tf.contrib.estimator.binary_classification_head(thresholds=[0.5])
  # Compute logits (1 per class).
  logits = tf.layers.dense(net, my_head.logits_dimension, activation=None, name="my_model_output_logits")
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

  def _train_op_fn(loss):
    return optimizer.minimize(loss, global_step=tf.train.get_global_step())

  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=_train_op_fn
  )


def main(unused_argv):
  set_tfconfig_environ()
  my_feature_columns = create_feature_columns()
  classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
      'feature_columns': my_feature_columns,
      # Two hidden layers of 10 nodes each.
      'hidden_units': FLAGS.hidden_units.split(','),
      'learning_rate': FLAGS.learning_rate,
      # The model must choose between 2 classes.
      'n_classes': 2
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
  hook = DebugHook()
  if FLAGS.input_format == "tfrecord":
    print("input_format: tfrecord")
    input_fn_for_training = lambda: train_input_fn(train_files, batch_size, shuffle_buffer_size)
    input_fn_for_eval = lambda: eval_input_fn(eval_files, batch_size)
  else:
    print("input_format: text")
    input_fn_for_training = lambda: my_input_fn(train_files, batch_size, shuffle_buffer_size)
    input_fn_for_eval = lambda: my_input_fn(eval_files, batch_size, 0)

  train_spec = tf.estimator.TrainSpec(
    input_fn=input_fn_for_training,
    max_steps=FLAGS.train_steps,
    hooks=[hook]
  )
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

  print("before train and evaluate")
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  print("after train and evaluate")

  # Evaluate accuracy.
  results = classifier.evaluate(input_fn=input_fn_for_eval)
  for key in sorted(results): print('%s: %s' % (key, results[key]))
  print("after evaluate")

  if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    print(feature_spec)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
  print("quit main")


if __name__ == "__main__":
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  if FLAGS.run_on_cluster: parse_argument()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
