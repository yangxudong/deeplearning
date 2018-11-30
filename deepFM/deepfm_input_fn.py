#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import feature_column as fc


def truncate(val):
  return tf.clip_by_value(val, -1.0, 1.0)


def create_linear_feature_columns():
  phoneBrand = fc.categorical_column_with_hash_bucket("phoneBrand", 1000)
  phoneResolution = fc.categorical_column_with_hash_bucket("phoneResolution", 500)
  phoneOs = fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0)
  matchScore = fc.numeric_column("matchScore", default_value=0.0)
  popScore = fc.numeric_column("popScore", default_value=0.0)
  brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0, normalizer_fn=truncate)
  cate2Prefer = fc.numeric_column("cate2Prefer", default_value=0.0, normalizer_fn=truncate)
  catePrefer = fc.numeric_column("catePrefer", default_value=0.0, normalizer_fn=truncate)
  sellerPrefer = fc.numeric_column("sellerPrefer", default_value=0.0, normalizer_fn=truncate)
  matchType = fc.categorical_column_with_identity("matchType", 9, default_value=0)
  position = fc.categorical_column_with_identity("position", 201, default_value=200)
  triggerNum = fc.categorical_column_with_identity("triggerNum", 51, default_value=50)
  triggerRank = fc.categorical_column_with_identity("triggerRank", 51, default_value=50)
  sceneType = fc.categorical_column_with_identity("type", 2, default_value=0)
  hour = fc.categorical_column_with_identity("hour", 24, default_value=0)
  columns = [phoneBrand, phoneResolution, phoneOs, matchScore, popScore, brandPrefer, cate2Prefer, catePrefer,
          sellerPrefer, matchType, position, triggerRank, triggerNum, sceneType, hour]
  print("linear feature columns:", columns)
  return columns

def create_interaction_feature_columns(shared_embedding_dim=60):
  # user embedding features
  phoneBrandId = fc.categorical_column_with_hash_bucket("phoneBrand", 1000)
  phoneBrand = fc.shared_embedding_columns([phoneBrandId], shared_embedding_dim)
  phoneResolutionId = fc.categorical_column_with_hash_bucket("phoneResolution", 500)
  phoneResolution = fc.shared_embedding_columns([phoneResolutionId], shared_embedding_dim)
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

  # item embedding features
  pid = fc.categorical_column_with_hash_bucket("productId", 1000000, dtype=tf.int64)
  sid = fc.categorical_column_with_hash_bucket("sellerId", 10240, dtype=tf.int64)
  bid = fc.categorical_column_with_hash_bucket("brandId", 10240, dtype=tf.int64)
  c1id = fc.categorical_column_with_hash_bucket("cate1Id", 100, dtype=tf.int64)
  c2id = fc.categorical_column_with_hash_bucket("cate2Id", 500, dtype=tf.int64)
  cid = fc.categorical_column_with_hash_bucket("cateId", 10240, dtype=tf.int64)

  # shared embedding
  pid_emb = fc.shared_embedding_columns([pids_weighted, pid], shared_embedding_dim, combiner='sum')
  bid_emb = fc.shared_embedding_columns([bids_weighted, bid], shared_embedding_dim, combiner='sum')
  cid_emb = fc.shared_embedding_columns([cids_weighted, cid], shared_embedding_dim, combiner='sum')
  c1id_emb = fc.shared_embedding_columns([c1ids_weighted, c1id], shared_embedding_dim, combiner='sum')
  sid_emb = fc.shared_embedding_columns([sids_weighted, sid], shared_embedding_dim, combiner='sum')
  c2id_emb = fc.shared_embedding_columns([c2id], shared_embedding_dim)

  columns = phoneBrand
  columns += phoneResolution
  columns += pid_emb
  columns += sid_emb
  columns += bid_emb
  columns += cid_emb
  columns += c1id_emb
  columns += c2id_emb
  print("interaction feature columns:", columns)
  return columns


def create_deep_feature_columns():
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
  triggerNum = fc.indicator_column(fc.categorical_column_with_identity("triggerNum", 41, default_value=40))
  triggerRank = fc.indicator_column(fc.categorical_column_with_identity("triggerRank", 41, default_value=40))
  sceneType = fc.indicator_column(fc.categorical_column_with_identity("type", 2, default_value=0))

  columns = [matchScore, matchType, triggerNum, triggerRank, sceneType,
             phoneOs, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer]
  print("deep feature columns:", columns)
  return columns


def parse_exmp(serial_exmp, feature_spec):
  spec = {"click": tf.FixedLenFeature([], tf.int64)}
  spec.update(feature_spec)
  feats = tf.parse_single_example(serial_exmp, features=spec)
  labels = feats.pop('click')
  return feats, labels


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size, num_parallel_readers):
  files = tf.data.Dataset.list_files(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.map(lambda x: parse_exmp(x, feature_spec), num_parallel_calls=8)
  dataset = dataset.repeat().batch(batch_size).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  return dataset


def eval_input_fn(filename, feature_spec):
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.map(lambda x: parse_exmp(x, feature_spec), num_parallel_calls=8)
  dataset = dataset.batch(1)
  return dataset