#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import feature_column as fc

def truncate(val):
  return tf.clip_by_value(val, -1.0, 1.0)

my_feature_columns = []
def create_feature_columns():
  # user feature
  phoneBrandId = fc.categorical_column_with_hash_bucket("phoneBrand", 1000)
  phoneResolutionId = fc.categorical_column_with_hash_bucket("phoneResolution", 500)
  phoneBrand = fc.embedding_column(phoneBrandId, 20)
  phoneResolution = fc.embedding_column(phoneResolutionId, 10)
  phoneOs = fc.indicator_column(
    fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0))
  # context feature
  matchScore = fc.numeric_column("matchScore", default_value=0.0)
  popScore = fc.numeric_column("popScore", default_value=0.0)
  brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0, normalizer_fn=truncate)
  cate2Prefer = fc.numeric_column("cate2Prefer", default_value=0.0, normalizer_fn=truncate)
  catePrefer = fc.numeric_column("catePrefer", default_value=0.0, normalizer_fn=truncate)
  sellerPrefer = fc.numeric_column("sellerPrefer", default_value=0.0, normalizer_fn=truncate)
  matchType = fc.indicator_column(fc.categorical_column_with_identity("matchType", 9, default_value=0))
  postition = fc.indicator_column(fc.categorical_column_with_identity("position", 201, default_value=200))
  triggerNum = fc.indicator_column(fc.categorical_column_with_identity("triggerNum", 51, default_value=50))
  triggerRank = fc.indicator_column(fc.categorical_column_with_identity("triggerRank", 51, default_value=50))
  sceneType = fc.indicator_column(fc.categorical_column_with_identity("type", 2, default_value=0))
  hour = fc.indicator_column(fc.categorical_column_with_identity("hour", 24, default_value=0))

  global my_feature_columns
  my_feature_columns = [matchScore, matchType, postition, triggerNum, triggerRank, sceneType, hour, phoneBrand,
                        phoneResolution, phoneOs, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer]
  print("feature columns:", my_feature_columns)
  return my_feature_columns


def parse_exmp(serial_exmp):
  click = fc.numeric_column("click", default_value=0, dtype=tf.int64)
  fea_columns = [click]
  fea_columns += my_feature_columns
  feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
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
  feats = tf.parse_single_example(serial_exmp, features=feature_spec)
  labels = feats.pop('click')
  return feats, labels

def train_input_fn(filenames, batch_size, shuffle_buffer_size, num_parallel_readers):
  #dataset = tf.data.TFRecordDataset(filenames)
  files = tf.data.Dataset.list_files(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
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