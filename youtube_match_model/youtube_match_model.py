#-*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.python.estimator.canned import optimizers
# for python 2.x
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("output_item_vector", "./item_vector_output/nce_weights.ckpt", "Path to the trained item vector.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_integer("n_classes", 150000, "The number of possible classes/labels")
flags.DEFINE_integer("num_sampled", 500, "The number of negative classes to randomly sample per batch.")
flags.DEFINE_string("hidden_units", "128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("last_hidden_units", "64", "last hidden layer of the NN, equal to user vector")
flags.DEFINE_string("eval_top_n", "20,50,100,200,300,500", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("top_k", 20, "predict the top k results")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.0, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("optimizer", "Adagrad", "the name of optimizer")
flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")
flags.DEFINE_boolean("use_batch_norm", False, "Whether to use batch normalization for hidden layers")
flags.DEFINE_boolean("predict", False, "Whether to predict")

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
  bids = fc.categorical_column_with_hash_bucket("behaviorBids", 10000, dtype=tf.int64)
  c1ids = fc.categorical_column_with_hash_bucket("behaviorC1ids", 100, dtype=tf.int64)
  cids = fc.categorical_column_with_hash_bucket("behaviorCids", 10000, dtype=tf.int64)
  sids = fc.categorical_column_with_hash_bucket("behaviorSids", 10000, dtype=tf.int64)
  pids = fc.categorical_column_with_hash_bucket("behaviorPids", 500000, dtype=tf.int64)
  bids_weighted = fc.weighted_categorical_column(bids, "bidWeights")
  c1ids_weighted = fc.weighted_categorical_column(c1ids, "c1idWeights")
  cids_weighted = fc.weighted_categorical_column(cids, "cidWeights")
  sids_weighted = fc.weighted_categorical_column(sids, "sidWeights")
  pids_weighted = fc.weighted_categorical_column(pids, "pidWeights")
  pid_embed = fc.embedding_column(pids_weighted, 64)
  bid_embed = fc.embedding_column(bids_weighted, 32)
  cid_embed = fc.embedding_column(cids_weighted, 48)
  c1id_embed = fc.embedding_column(c1ids_weighted, 10)
  sid_embed = fc.embedding_column(sids_weighted, 32)
  phoneBrandId = fc.categorical_column_with_hash_bucket("phoneBrand", 1000)
  phoneBrand = fc.embedding_column(phoneBrandId, 20)
  phoneResolutionId = fc.categorical_column_with_hash_bucket("phoneResolution", 500)
  phoneResolution = fc.embedding_column(phoneResolutionId, 10)
  phoneOs = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0))
  gender = fc.indicator_column(fc.categorical_column_with_identity("gender", num_buckets=3, default_value=0))
  age_class = fc.indicator_column(fc.categorical_column_with_identity("age_class", num_buckets=7, default_value=0))
  has_baby = fc.indicator_column(fc.categorical_column_with_identity("has_baby", num_buckets=2, default_value=0))
  baby_gender = fc.indicator_column(fc.categorical_column_with_identity("baby_gender", num_buckets=3, default_value=0))
  baby_age = fc.indicator_column(fc.categorical_column_with_identity("baby_age", num_buckets=7, default_value=0))
  grade = fc.indicator_column(fc.categorical_column_with_identity("grade", num_buckets=7, default_value=0))
  rfm_type = fc.indicator_column(fc.categorical_column_with_identity("bi_rfm_type", num_buckets=12, default_value=0))
  city_id = fc.categorical_column_with_hash_bucket("city", 700)
  city = fc.embedding_column(city_id, 16)
  userType = fc.indicator_column(fc.categorical_column_with_identity("user_type", 6, default_value=0))
  hour = fc.indicator_column(fc.categorical_column_with_identity("hour", 24, default_value=0))

  global my_feature_columns
  my_feature_columns = [userType, hour, gender, age_class, has_baby, baby_gender, baby_age, grade, rfm_type,
                        phoneBrand, phoneResolution, phoneOs,
                        pid_embed, sid_embed, bid_embed, cid_embed, c1id_embed, city]
  print("feature columns:", my_feature_columns)
  return my_feature_columns


def parse_exmp(serial_exmp):
  label = fc.numeric_column("label", default_value=0, dtype=tf.int64)
  fea_columns = [label]
  fea_columns += my_feature_columns
  feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
  feats = tf.parse_single_example(serial_exmp, features=feature_spec)
  labels = feats.pop('label')
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

def build_mode_norm(features, mode, params):
  # Build the hidden layers, sized according to the 'hidden_units' param.
  use_batch_norm = params['use_batch_norm']
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  net = fc.input_layer(features, params['feature_columns'])
  if use_batch_norm:
    net = tf.layers.batch_normalization(net, training=is_training)

  for units in params['hidden_units']:
    if use_batch_norm:
      x = tf.layers.dense(net, units=units, activation=None, use_bias=False)
      net = tf.nn.relu(tf.layers.batch_normalization(x, training=is_training))
    else:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  if use_batch_norm:
    x = tf.layers.dense(net, units=params['last_hidden_units'], activation=None, use_bias=False)
    net = tf.nn.elu(tf.layers.batch_normalization(x, training=is_training), name='user_vector_layer')
  else:
    net = tf.layers.dense(net, units=params['last_hidden_units'], activation=tf.nn.relu, name='user_vector_layer')

  return net

def build_mode_norm_test(features, mode, params):
  # Build the hidden layers, sized according to the 'hidden_units' param.
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  fea_net = fc.input_layer(features, params['feature_columns'])
  fea_net = tf.layers.batch_normalization(fea_net, training=is_training)

  #x1 = tf.layers.dense(fea_net, units=256, activation=None, use_bias=False)
  #hidden1 = tf.nn.relu(tf.layers.batch_normalization(x1, training=is_training), name='hidden1')

  #x2 = tf.layers.dense(hidden1, units=128, activation=None, use_bias=False)
  #hidden2 = tf.nn.relu(tf.layers.batch_normalization(x2, training=is_training), name='hidden2')

  #net = tf.layers.dense(hidden2, units=64, activation=tf.tanh, name='user_vector_layer')

  hidden1 = tf.layers.dense(fea_net, units=128, activation=tf.nn.relu, name='hidden1')
  #hidden2 = tf.layers.dense(hidden1, units=128, activation=tf.nn.relu, name='hidden2') 
  net = tf.layers.dense(hidden1, units=64, activation=tf.nn.relu, name='user_vector_layer')

  return net
  #return fea_net,hidden1,hidden2,net

def build_model(features, mode, params):
  net = fc.input_layer(features, params['feature_columns'])
  #net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
      print ("net node count", net.shape[-1].value)
  # Build the last hidden layer, equal to user vector
  net = tf.layers.dense(net, units=params['last_hidden_units'], activation=tf.nn.relu, name='user_vector_layer')
  return net


def my_model(features, labels, mode, params):
  net = build_model(features, mode, params)
  nce_weights = tf.Variable(
    tf.truncated_normal([params['n_classes'], params['last_hidden_units']],
                        stddev=1.0 / math.sqrt(params['last_hidden_units'])), name='nce_weights')
  nce_biases = tf.Variable(tf.zeros([params['n_classes']]), name='nce_biases')
  logits = tf.matmul(net, tf.transpose(nce_weights)) + nce_biases
  top_k_values, top_k_indices = tf.nn.top_k(logits, params["top_k"])

  # Compute predictions.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'user_vector': net,
      'top_k_values': top_k_values,
      'top_k_indices': top_k_indices,
    }
    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(predictions)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Compute logits (1 per class).
    precisions = params['eval_top_n'] if 'eval_top_n' in params else [5, 10, 20, 50, 100]
    metrics = {}
    for k in precisions:
      metrics["recall/recall@" + str(k)] = tf.metrics.recall_at_k(labels, logits, int(k))
      metrics["precision/precision@" + str(k)] = tf.metrics.precision_at_k(labels, logits, int(k))
      correct = tf.nn.in_top_k(logits, tf.squeeze(labels), int(k))
      metrics["accuary/accuary@" + str(k)] = tf.metrics.accuracy(labels=tf.ones_like(labels, dtype=tf.float32), predictions=tf.to_float(correct))

    labels_one_hot = tf.one_hot(labels, params['n_classes'])
    labels_one_hot = tf.reshape(labels_one_hot, (-1, params['n_classes']))
    print("labels_one_hot shape", labels_one_hot.get_shape())
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)
    loss = tf.reduce_mean(loss)
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN
  # Compute train nce loss.
  loss = tf.reduce_mean(tf.nn.nce_loss(
    weights=nce_weights,
    biases=nce_biases,
    labels=labels,
    inputs=net,
    num_sampled=params['num_sampled'],
    num_classes=params['n_classes'],
    num_true=1,
    remove_accidental_hits=True,
    partition_strategy='div',
    name='match_model_nce_loss'))

  optimizer = optimizers.get_optimizer_instance(params["optimizer"], params["learning_rate"])
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(unused_argv):
  set_tfconfig_environ()
  create_feature_columns()
  classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
      'feature_columns': my_feature_columns,
      'hidden_units': FLAGS.hidden_units.split(','),
      'last_hidden_units': FLAGS.last_hidden_units,
      'optimizer': FLAGS.optimizer,
      'learning_rate': FLAGS.learning_rate,
      'dropout_rate': FLAGS.dropout_rate,
      'n_classes': FLAGS.n_classes,
      'num_sampled': FLAGS.num_sampled,
      'top_k': FLAGS.top_k,
      'eval_top_n': FLAGS.eval_top_n.split(',')
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

  # Train the Model.
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: train_input_fn(train_files, batch_size, shuffle_buffer_size),
    max_steps=FLAGS.train_steps
  )
  # Evaluate the model.
  input_fn_for_eval = lambda: eval_input_fn(eval_files, batch_size)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600)
  print("before train and evaluate")
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  print("after train and evaluate")

  # Evaluate accuracy.
  results = classifier.evaluate(input_fn=input_fn_for_eval)
  for key in sorted(results): print('%s: %s' % (key, results[key]))
  print("after evaluate")

  # save
  #nce_weights = tf.Variable(tf.convert_to_tensor(classifier.get_variable_value('nce_weights')), name="nce_weights")
  #saver = tf.train.Saver([nce_weights])  
  
  # restore
  # nce_weights = tf.get_variable("nce_weights", shape=[FLAGS.n_classes, FLAGS.last_hidden_units])
  # saver = tf.train.Saver()

  # with tf.Session() as sess:
    # save item vector
    #init_op = tf.global_variables_initializer()
    #sess.run(init_op)
    #saver.save(sess, FLAGS.output_item_vector)
    
    # restore item vector 
    # saver.restore(sess, FLAGS.output_item_vector)
    # print ("item vectors:", sess.run(nce_weights))   

  # np.savetxt(FLAGS.output_item_vector, nce_weights)

  # print("classifier vars names:", classifier.get_variable_names())
  # print("nce_weights :", classifier.get_variable_value('nce_weights'))
  # print("nce_weights/Adagrad :", classifier.get_variable_value('nce_weights/Adagrad'))
  # print("nce_weights shape :", classifier.get_variable_value('nce_weights').shape)

  # print("nce_biases :", classifier.get_variable_value('nce_biases'))
  # print("nce_biases shape :", classifier.get_variable_value('nce_biases').shape)

  if FLAGS.predict:
    pred = list(classifier.predict(input_fn=input_fn_for_eval))
    #print("pred result example", next(pred))
    import random
    random.shuffle(pred)
    print("pred result example", pred[:50])
  elif FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    print(feature_spec)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    
    print("save item vector")
    nce_weights = classifier.get_variable_value('nce_weights')
    nce_biases = classifier.get_variable_value('nce_biases')
    [rows, cols] = nce_weights.shape
    with tf.gfile.FastGFile(FLAGS.output_item_vector, 'w') as f:
      for i in range(rows):
        f.write(unicode(str(i) + "\t"))
        for j in range(cols):
          f.write(unicode(str(nce_weights[i, j])))
          f.write(u',')
        f.write(unicode(str(nce_biases[i])))
        f.write(u'\n')
  print("quit main")


if __name__ == "__main__":
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  if FLAGS.run_on_cluster: parse_argument()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
