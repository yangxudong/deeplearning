from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import tensorflow as tf
from tensorflow import feature_column as fc
# for python 2.x
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
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
my_feature_columns = []


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


def create_feature_columns():
  # user feature
  bids = fc.categorical_column_with_hash_bucket("behaviorBids", 10240, dtype=tf.int64)
  c1ids = fc.categorical_column_with_hash_bucket("behaviorC1ids", 100, dtype=tf.int64)
  cids = fc.categorical_column_with_hash_bucket("behaviorCids", 10240, dtype=tf.int64)
  sids = fc.categorical_column_with_hash_bucket("behaviorSids", 10240, dtype=tf.int64)
  pids = fc.categorical_column_with_hash_bucket("behaviorPids", 1000000, dtype=tf.int64)
  bids_weighted = fc.weighted_categorical_column(bids, "bidWeights")
  c1ids_weighted = fc.weighted_categorical_column(c1ids, "c1idWeights")
  cids_weighted = fc.weighted_categorical_column(cids, "cidWeights")
  sids_weighted = fc.weighted_categorical_column(sids, "sidWeights")
  pids_weighted = fc.weighted_categorical_column(pids, "pidWeights")

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
        ["ALL", "TongZhuang", "XieBao", "MuYing", "NvZhuang", "MeiZhuang", "JuJia", "MeiShi"], default_value=0))

  pid_embed = fc.shared_embedding_columns([pids_weighted, pid], 64, combiner='sum', shared_embedding_collection_name="pid")
  bid_embed = fc.shared_embedding_columns([bids_weighted, bid], 32, combiner='sum', shared_embedding_collection_name="bid")
  cid_embed = fc.shared_embedding_columns([cids_weighted, cid], 32, combiner='sum', shared_embedding_collection_name="cid")
  c1id_embed = fc.shared_embedding_columns([c1ids_weighted, c1id], 10, combiner='sum', shared_embedding_collection_name="c1id")
  sid_embed = fc.shared_embedding_columns([sids_weighted, sid], 32, combiner='sum', shared_embedding_collection_name="sid")
  global my_feature_columns
  my_feature_columns = [matchScore, matchType, postition, triggerNum, triggerRank, sceneType, hour, phoneBrand, phoneResolution,
             phoneOs, tab, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer]
  my_feature_columns += pid_embed
  my_feature_columns += sid_embed
  my_feature_columns += bid_embed
  my_feature_columns += cid_embed
  my_feature_columns += c1id_embed
  print("feature columns:", my_feature_columns)
  return my_feature_columns


def parse_exmp(serial_exmp):
  click = fc.numeric_column("click", default_value=0, dtype=tf.int64)
  fea_columns = [click]
  fea_columns += my_feature_columns
  feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
  feats = tf.parse_single_example(serial_exmp, features=feature_spec)
  labels = feats.pop('click')
  return feats, labels


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
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
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
  create_feature_columns()
  classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
      'feature_columns': my_feature_columns,
      'hidden_units': FLAGS.hidden_units.split(','),
      'learning_rate': FLAGS.learning_rate,
      'dropout_rate': FLAGS.dropout_rate
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

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input_fn(train_files, batch_size, shuffle_buffer_size),
    max_steps=FLAGS.train_steps
  )
  input_fn_for_eval = lambda: eval_input_fn(eval_files, batch_size)
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
