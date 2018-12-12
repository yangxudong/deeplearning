#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import sys
# for python 2.x
#reload(sys)
#sys.setdefaultencoding("utf-8")
sys.path.append(os.getcwd())
from .deep_interest_network import *
from .bb_input_fn import *

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
flags.DEFINE_string("hidden_units", "512,256,128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_string("attention_hidden_units", "32,16",
                    "Comma-separated list of number of units in each hidden layer of the attention layer")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 10, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("optimizer", "RMSProp", "optimizer")
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
flags.DEFINE_boolean("use_batch_norm", True, "Whether to use batch normalization in deep network")

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


def main(unused_argv):
  set_tfconfig_environ()
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
  print("train_data:", train_files)
  print("eval_data:", eval_files)
  print("train steps:", FLAGS.train_steps, "batch_size:", FLAGS.batch_size)
  print("shuffle_buffer_size:", FLAGS.shuffle_buffer_size)
  feature_columns = create_feature_columns()
  estimator = DIN(
    params={
      'feature_columns': feature_columns,
      'hidden_units': FLAGS.hidden_units.split(','),
      'learning_rate': FLAGS.learning_rate,
      'attention_hidden_units': FLAGS.attention_hidden_units.split(','),
      'vocab_size': {
        "product": 1000000,
        "cate1": 100,
        "cate": 10240,
        "brand": 10240,
        "seller": 10240
      },
      'embedding_size': {
        "product": 64,
        "cate1": 10,
        "cate": 32,
        "brand": 32,
        "seller": 32
      },
      'dropout_rate': FLAGS.dropout_rate
    },
    optimizer='RMSProp',
    config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  )
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input_fn(train_files, FLAGS.batch_size, FLAGS.shuffle_buffer_size, FLAGS.num_parallel_readers),
    max_steps=FLAGS.train_steps
  )
  input_fn_for_eval = lambda: eval_input_fn(eval_files, FLAGS.batch_size)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300, steps=None)

  print("before train and evaluate")
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  print("after train and evaluate")

  # Evaluate accuracy.
  results = estimator.evaluate(input_fn=input_fn_for_eval)
  for key in sorted(results): print('%s: %s' % (key, results[key]))
  print("after evaluate")

  if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    other_feature_spec = {
      "behaviorBids": tf.FixedLenFeature([20], tf.int64),
      "behaviorCids": tf.FixedLenFeature([20], tf.int64),
      "behaviorC1ids": tf.FixedLenFeature([10], tf.int64),
      "behaviorSids": tf.FixedLenFeature([20], tf.int64),
      "behaviorPids": tf.FixedLenFeature([20], tf.int64),
      "productId": tf.FixedLenFeature([], tf.int64),
      "sellerId": tf.FixedLenFeature([], tf.int64),
      "brandId": tf.FixedLenFeature([], tf.int64),
      "cate1Id": tf.FixedLenFeature([], tf.int64),
      "cateId": tf.FixedLenFeature([], tf.int64)
    }
    feature_spec.update(other_feature_spec)
    print(feature_spec)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
  print("quit main")


if __name__ == "__main__":
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  if FLAGS.run_on_cluster: parse_argument()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
