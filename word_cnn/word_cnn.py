from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import os
import json
# for python 2.x
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_integer("embedding_size", 128, "embedding size")
flags.DEFINE_integer("num_filters", 100, "number of filters")
flags.DEFINE_integer("num_classes", 14, "number of classes")
flags.DEFINE_integer("shuffle_buffer_size", 1000000, "dataset shuffle buffer size")
flags.DEFINE_integer("sentence_max_len", 100, "max length of sentences")
flags.DEFINE_integer("batch_size", 128, "number of instances in a batch")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_string("data_dir", "dbpedia/dbpedia_csv", "Directory containing the dataset")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated list of number of window size in each filter")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
FLAGS = flags.FLAGS

def parse_line(line, vocab):
  def get_content(record):
    fields = record.decode().split(",")
    if len(fields) < 3:
      raise ValueError("invalid record %s" % record)
    text = re.sub(r"[^A-Za-z0-9\'\`]", " ", fields[2])
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\`", "\'", text)
    text = text.strip().lower()
    tokens = text.split()
    tokens = [w.strip("'") for w in tokens if len(w.strip("'")) > 0]
    n = len(tokens)  # type: int
    if n > FLAGS.sentence_max_len:
      tokens = tokens[:FLAGS.sentence_max_len]
    if n < FLAGS.sentence_max_len:
      tokens += [FLAGS.pad_word] * (FLAGS.sentence_max_len - n)
    return [tokens, np.int32(fields[0])]

  result = tf.py_func(get_content, [line], [tf.string, tf.int32])
  result[0].set_shape([FLAGS.sentence_max_len])
  result[1].set_shape([])
  # Lookup tokens to return their ids
  ids = vocab.lookup(result[0])
  return {"sentence": ids}, result[1] - 1


def input_fn(path_csv, path_vocab, shuffle_buffer_size, num_oov_buckets):
  """Create tf.data Instance from csv file
  Args:
      path_csv: (string) path containing one example per line
      vocab: (tf.lookuptable)
  Returns:
      dataset: (tf.Dataset) yielding list of ids of tokens and labels for each example
  """
  vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
  # Load txt file, one example per line
  dataset = tf.data.TextLineDataset(path_csv)
  # Convert line into list of tokens, splitting by white space
  dataset = dataset.map(lambda line: parse_line(line, vocab))
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  return dataset


def my_model(features, labels, mode, params):
  sentence = features['sentence']
  # Get word embeddings for each token in the sentence
  embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                               shape=[params["vocab_size"], FLAGS.embedding_size])
  sentence = tf.nn.embedding_lookup(embeddings, sentence) # shape:(batch, sentence_len, embedding_size)
  # add a channel dim, required by the conv2d and max_pooling2d method
  sentence = tf.expand_dims(sentence, -1) # shape:(batch, sentence_len/height, embedding_size/width, channels=1)

  pooled_outputs = []
  for filter_size in params["filter_sizes"]:
      conv = tf.layers.conv2d(
          sentence,
          filters=FLAGS.num_filters,
          kernel_size=[filter_size, FLAGS.embedding_size],
          strides=(1, 1),
          padding="VALID",
          activation=tf.nn.relu)
      pool = tf.layers.max_pooling2d(
          conv,
          pool_size=[FLAGS.sentence_max_len - filter_size + 1, 1],
          strides=(1, 1),
          padding="VALID")
      pooled_outputs.append(pool)
  h_pool = tf.concat(pooled_outputs, 3) # shape: (batch, 1, len(filter_size) * embedding_size, 1)
  h_pool_flat = tf.reshape(h_pool, [-1, FLAGS.num_filters * len(params["filter_sizes"])]) # shape: (batch, len(filter_size) * embedding_size)
  if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
    h_pool_flat = tf.layers.dropout(h_pool_flat, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  logits = tf.layers.dense(h_pool_flat, FLAGS.num_classes, activation=None)

  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  def _train_op_fn(loss):
    return optimizer.minimize(loss, global_step=tf.train.get_global_step())

  my_head = tf.contrib.estimator.multi_class_head(n_classes=FLAGS.num_classes)
  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=_train_op_fn
  )

def main(unused_argv):
  # Load the parameters from the dataset, that gives the size etc. into params
  json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
  assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
  # Loads parameters from json file
  with open(json_path) as f:
    config = json.load(f)
  FLAGS.pad_word = config["pad_word"]
  if config["train_size"] < FLAGS.shuffle_buffer_size:
    FLAGS.shuffle_buffer_size = config["train_size"]
  print("shuffle_buffer_size:", FLAGS.shuffle_buffer_size)

  # Get paths for vocabularies and dataset
  path_words = os.path.join(FLAGS.data_dir, 'words.txt')
  assert os.path.isfile(path_words), "No vocab file found at {}, run build_vocab.py first".format(path_words)
  #words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=config["num_oov_buckets"])

  path_train = os.path.join(FLAGS.data_dir, 'train.csv')
  path_eval = os.path.join(FLAGS.data_dir, 'test.csv')

  classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
      'vocab_size': config["vocab_size"],
      'filter_sizes': map(int, FLAGS.filter_sizes.split(',')),
      'learning_rate': FLAGS.learning_rate,
      'dropout_rate': FLAGS.dropout_rate
    },
    config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  )

  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(path_train, path_words, FLAGS.shuffle_buffer_size, config["num_oov_buckets"]),
    max_steps=FLAGS.train_steps
  )
  input_fn_for_eval = lambda: input_fn(path_eval, path_words, 0, config["num_oov_buckets"])
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

  print("before train and evaluate")
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
  print("after train and evaluate")


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
