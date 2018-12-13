#-*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import feature_column as fc


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
  userType = fc.numeric_column("user_type", default_value=0, dtype=tf.int64)
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
    cms_scale, cms_scale_sqrt, userType]
  feature_columns += c2id_embed
  feature_columns += phoneResolution
  feature_columns += phoneBrand
  return feature_columns


def parse_exmp(serial_exmp, feature_spec):
  #share = fc.numeric_column("share", default_value=0, dtype=tf.int64)
  click = fc.numeric_column("click", default_value=0, dtype=tf.int64)
  pay = fc.numeric_column("pay", default_value=0, dtype=tf.int64)
  fea_columns = [click, pay]
  feature_spec.update(tf.feature_column.make_parse_example_spec(fea_columns))
  feats = tf.parse_single_example(serial_exmp, features=feature_spec)
  feats["modified_time_sqrt"] = tf.sqrt(feats["modified_time"])
  feats["modified_time_square"] = tf.square(feats["modified_time"])
  feats["cms_scale_sqrt"] = tf.sqrt(feats["cms_scale"])
  if "behaviorTypes" in feats:
    feats["behaviorTypes"] = feats["behaviorTypes"] - 1
  click = feats.pop('click')
  pay = feats.pop('pay')
  return feats, {'ctr': tf.to_float(click), 'cvr': tf.to_float(pay)}


def train_input_fn(filenames, feature_spec, batch_size, shuffle_buffer_size, num_parallel_readers):
  #dataset = tf.data.TFRecordDataset(filenames)
  files = tf.data.Dataset.list_files(filenames)
  dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
  dataset = dataset.map(lambda x: parse_exmp(x, feature_spec), num_parallel_calls=8)
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
  # Shuffle, repeat, and batch the examples.
  #dataset = dataset.batch(batch_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  # Return the read end of the pipeline.
  return dataset

