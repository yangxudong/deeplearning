"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import json
import os

import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
task_index = int(os.environ["TF_INDEX"])
task_type = os.environ["TF_ROLE"]

tf_config = dict()
worker_num = len(cluster["worker"])
if task_type == "ps":
    tf_config["task"] = {"index":task_index, "type":task_type}
else:
    if task_index == 0:
        tf_config["task"] = {"index":0, "type":"chief"}
    else:
        tf_config["task"] = {"index":task_index-1, "type":task_type}

if worker_num == 1:
    cluster["chief"] = cluster["worker"]
    del cluster["worker"]
else:
    cluster["chief"] = [cluster["worker"][0]]
    del cluster["worker"][0]

tf_config["cluster"] = cluster
os.environ["TF_CONFIG"] = json.dumps(tf_config)
print(json.loads(os.environ["TF_CONFIG"]))

_CSV_COLUMNS = [
    'device_model'
    , 'brand'
    , 'resolution'
    , 'carrier'
    , 'access'
    , 'channel'
    , 'os'
    , 'province'
    , 'city'
    , 'ord_cnt_1d_type'
    , 'ord_cnt_7d_type'
    , 'ord_cnt_15d_type'
    , 'ord_cnt_30d_type'
    , 'ord_amt_1d_type'
    , 'ord_amt_7d_type'
    , 'ord_amt_15d_type'
    , 'ord_amt_30d_type'
    , 'ipv_1d_type'
    , 'ipv_7d_type'
    , 'ipv_15d_type'
    , 'ipv_30d_type'
    , 'product_id'
    , 'cid'
    , 'brand_id'
    , 'c1_id'
    , 'c2_id'
    , 'user_click_rate_1d'
    , 'user_fenxiang_rate_1d'
    , 'user_buy_rate_1d'
    , 'user_click_rate_1w'
    , 'user_fenxiang_rate_1w'
    , 'user_buy_rate_1w'
    , 'user_click_rate_2w'
    , 'user_fenxiang_rate_2w'
    , 'user_buy_rate_2w'
    , 'user_click_rate_1m'
    , 'user_fenxiang_rate_1m'
    , 'user_buy_rate_1m'
    , 'expose_norm_score_1d'
    , 'browse_norm_score_1d'
    , 'fenxiang_norm_score_1d'
    , 'buy_norm_score_1d'
    , 'expose_norm_score_1w'
    , 'browse_norm_score_1w'
    , 'fenxiang_norm_score_1w'
    , 'buy_norm_score_1w'
    , 'expose_norm_score_2w'
    , 'browse_norm_score_2w'
    , 'fenxiang_norm_score_2w'
    , 'buy_norm_score_2w'
    , 'expose_norm_score_1m'
    , 'browse_norm_score_1m'
    , 'fenxiang_norm_score_1m'
    , 'buy_norm_score_1m'
    , 'province_cid'
    , 'province_c2_id'
    , 'province_c1_id'
    , 'province_brand_id'
    , 'city_cid'
    , 'city_c2_id'
    , 'city_c1_id'
    , 'city_brand_id'
    , 'os_cid'
    , 'os_c2_id'
    , 'os_c1_id'
    , 'os_brand_id'
    , 'brand_cid'
    , 'brand_c2_id'
    , 'brand_c1_id'
    , 'brand_brand_id'
    , 'resolution_cid'
    , 'resolution_c2_id'
    , 'resolution_c1_id'
    , 'resolution_brand_id'
    , 'cid_prefer_1d'
    , 'cate1_prefer_1d'
    , 'cate2_prefer_1d'
    , 'brand_prefer_1d'
    , 'cid_prefer_1w'
    , 'cate1_prefer_1w'
    , 'cate2_prefer_1w'
    , 'brand_prefer_1w'
    , 'cid_prefer_2w'
    , 'cate1_prefer_2w'
    , 'cate2_prefer_2w'
    , 'brand_prefer_2w'
    , 'cid_prefer_1m'
    , 'cate1_prefer_1m'
    , 'cate2_prefer_1m'
    , 'brand_prefer_1m'
    , 'position'
    , 'match_type'
    , 'pids'
    , 'cids'
    , 'weights'
    , 'ctr_flag'
]

_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [''], ['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='hdfs://bigdata/tmp/wdl',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--max_steps', type=int, default=10000, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=2048, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default='training_example_data/',
    help='Path to the training data.')

parser.add_argument(
    '--output_model', type=str, default='output_model/',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='test_data/',
    help='Path to the test data.')

#
f = open('fg_file2')
contents = f.read()
f.close()
fg = json.loads(contents)
fg_detail = fg['features']

def build_model_columns():
    base_columns = []
    crossed_columns = []
    deep_columns = []
    for item in fg_detail:
        if item['feature_type'] == 'id_feature':
            hash_bucket_size = item['hash_bucket_size']
            if 'wide_feature' in item.keys() and item['wide_feature'] == True:
                base_columns.append(tf.feature_column.categorical_column_with_hash_bucket(
                    item['expression'], hash_bucket_size=hash_bucket_size))
            elif 'embedding_dimension' in item.keys():
                embedding_dimension = item['embedding_dimension']
                deep_columns.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
                    item['expression'], hash_bucket_size=hash_bucket_size), dimension=embedding_dimension))

        if item['feature_type'] == 'raw_feature':
            if 'wide_feature' in item.keys() and item['wide_feature'] == True:
                base_columns.append(tf.feature_column.numeric_column(item['expression']))
            else:
                deep_columns.append(tf.feature_column.numeric_column(item['expression']))

        if item['feature_type'] == 'combo_feature':
            hash_bucket_size = item['hash_bucket_size']
            crossed_keys = item['expression']
            print(crossed_keys)
            crossed_columns.append(tf.feature_column.crossed_column(crossed_keys, hash_bucket_size=hash_bucket_size))

        if item['feature_type'] in ('lookup_feature', 'match_feature'):
            deep_columns.append(tf.feature_column.numeric_column(item['feature_name']))

    pids = tf.feature_column.categorical_column_with_hash_bucket("pids", 100000)
    cids = tf.feature_column.categorical_column_with_hash_bucket("cids", 10000)
    pid = tf.feature_column.categorical_column_with_hash_bucket("product_id", 100000)
    cid = tf.feature_column.categorical_column_with_hash_bucket("cid", 10000)

    pids_weighted_column = tf.feature_column.weighted_categorical_column(
        categorical_column=pids, weight_feature_key='weights')

    cids_weighted_column = tf.feature_column.weighted_categorical_column(
        categorical_column=cids, weight_feature_key='weights')

    pid_embed = tf.feature_column.shared_embedding_columns([pids_weighted_column, pid], 64, combiner='sum')
    cid_embed = tf.feature_column.shared_embedding_columns([cids_weighted_column, cid], 32, combiner='sum')

    deep_columns += pid_embed
    deep_columns += cid_embed

    wide_columns = base_columns + crossed_columns
    return wide_columns, deep_columns

def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    print(wide_columns)
    print(deep_columns)
    hidden_units = [256, 128, 64]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    #run_config = tf.estimator.RunConfig().replace(
    #    session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
        #config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units)
        #config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units)
        #config=run_config)


def input_fn2(data_file, shuffle, batch_size):
    import random
    import numpy as np
    if shuffle:
        random.shuffle(data_file)

    def get_content(record):
        fields = record.decode('utf-8').split("\t")
        index = 0
        elems = []
	realPidCnt = 0
        for item in fields:
            if _CSV_COLUMNS[index] not in ('pids', 'cids', 'weights'):
                if _CSV_COLUMN_DEFAULTS[index][0] == '':
                    elems.append(item)
                else:
                    try:
                        elems.append(np.float32(item))
                    except:
                        elems.append(np.float32(0.0))
            else:
                if _CSV_COLUMNS[index] != 'weights':
		    item_splits = item.split(',')
		    if _CSV_COLUMNS[index] == 'pids':
		        for item_split in item_splits:
			    if item_split != '':
				realPidCnt += 1 
                    elems.append(item_splits)
                else:
		    weight_splits = [np.float32(x) for x in item.split(',')]
		    weights_new = []
		    nonzero_cnt = 0
		    for weight_split in weight_splits:
			if nonzero_cnt < realPidCnt:
			    if weight_split > 0:
				weights_new.append(weight_split)
			    else:
				weights_new.append(np.float32(0.01))
			else:
			    weights_new.append(np.float32(0.0))
		        nonzero_cnt += 1
                    elems.append(weights_new)
            index += 1

        return elems

    def parse_csv(value):
        print('Parsing', data_file)
        out_type = []
	index = 0
        for item in _CSV_COLUMN_DEFAULTS:
            if item[0] == '' and _CSV_COLUMNS[index] != 'weights':
                out_type.append(tf.string)
            else:
                out_type.append(tf.float32)
	    index += 1
        result = tf.py_func(get_content, [value], out_type)
        index = 0
        for item in result:
            if _CSV_COLUMNS[index] not in ('pids', 'cids', 'weights'):
                result[index].set_shape([])
            else:
                result[index].set_shape([20])
            index += 1
        features = dict(zip(_CSV_COLUMNS, result))
        labels = features.pop('ctr_flag')
        return features, tf.equal(labels, '1.0')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    #if shuffle:
    #    dataset = dataset.shuffle(buffer_size=100000)

    dataset = dataset.map(parse_csv, num_parallel_calls=100)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if shuffle:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    return dataset

def test_input_fn(data_file, batch_size):

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim='\t')
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('ctr_flag')
        return features, tf.equal(labels, '1.0')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv, num_parallel_calls=100)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.

    dataset = dataset.batch(batch_size)
    return dataset

#hdfs://bigdata/user/yipin.yang/search_ad_ctr/wdl/training_example_data/
def main(unused_argv):
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    if not INPUT_PATH:
        FLAGS.train_data = None
    else:
        FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
        FLAGS.test_data = INPUT_PATH.get(FLAGS.test_data)
    import datetime
    today = datetime.date.today()
    train_data_file = FLAGS.train_data
    test_data_file = FLAGS.test_data
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn2(
        train_data_file, True, FLAGS.batch_size), max_steps=FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn2(
        test_data_file, False, FLAGS.batch_size))

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # Evaluate accuracy.
    results = model.evaluate(input_fn=lambda: input_fn2(
        test_data_file, False, FLAGS.batch_size))
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


    if task_type == "worker" and task_index == 0:
        wide_columns, deep_columns = build_model_columns()
        feature_columns_new = set(wide_columns + deep_columns)
        ##print(feature_columns_new)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns_new)
        #print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



