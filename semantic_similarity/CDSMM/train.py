# -*- coding: UTF-8 -*-
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
import os
import json
import tensorflow as tf
from tensorflow import feature_column as fc
from dssm import SemanticModel
from tensorflow import logging
# from load_embedding import build_embedding_initializer

flags = tf.flags
# params for tensorflow
flags.DEFINE_string("checkpointDir", "model_dir", "Directory where checkpoints and event logs are written to.")
flags.DEFINE_string("tables", "", "input tables")
flags.DEFINE_string("volumes", "", "input volume path")
flags.DEFINE_string("outputs", "output", "Directory where model is written to")
flags.DEFINE_string("buckets", "", "input oss bucket info")
flags.DEFINE_integer("num_gpus", 8, "number of gpu cores")
# hyper params
flags.DEFINE_string("train_data", "data/samples", "Path to the training data")
flags.DEFINE_string("eval_data", "data/samples", "Path to the evaluation data.")
flags.DEFINE_string("char_cnn_filter_sizes", "4,7,10", "the size of cnn kernels.")
flags.DEFINE_integer("char_cnn_num_filters", 64, "Number of cnn filters")
flags.DEFINE_string("word_cnn_filter_sizes", "3,4,5", "the size of cnn kernels.")
flags.DEFINE_integer("word_cnn_num_filters", 128, "Number of cnn filters")
flags.DEFINE_string("hidden_units", "256,256,128",
                    "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_string("activations", "relu,relu,tanh", "Comma-separated list of activation functions for deep layers")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("save_checkpoints_secs", 720, "Save checkpoints every this many seconds")
flags.DEFINE_integer("keep_checkpoint_max", 6, "how many checkpoints should be keep")
flags.DEFINE_integer("batch_size", 1536, "Training batch size")
flags.DEFINE_integer("num_parallel_readers", 10, "number of parallel readers for training data")
flags.DEFINE_integer("num_parallel_parser", 8, "number of parallel parser for training data")
flags.DEFINE_integer("buffer_size", 128 * 1024 * 1024, "dataset read buffer size")
flags.DEFINE_integer("max_char_seq_len", 60, "max length of text char sequence")
flags.DEFINE_integer("max_word_seq_len", 30, "max length of text word sequence")
flags.DEFINE_integer("word_vocab_size", 800000, "the length of word vocabulary")
flags.DEFINE_integer("char_vocab_size", 18000, "the length of char vocabulary")
flags.DEFINE_integer("tag_vocab_size", 36, "the length of tag vocabulary")
flags.DEFINE_integer("word_embedding_size", 128, "the dim of word embedding vector")
flags.DEFINE_integer("char_embedding_size", 64, "the dim of char embedding vector")
flags.DEFINE_integer("tag_embedding_size", 8, "the dim of word's tag embedding vector")
flags.DEFINE_float("learning_rate", 0.035, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.0, "Drop out rate")
flags.DEFINE_float("margin", 0.25, "the margin of the softmax loss function")
flags.DEFINE_float("negative_margin", 0.35, "extra margin of the negative samples")
flags.DEFINE_float("smooth", 1.8, "the smooth coefficient of the softmax")
flags.DEFINE_float("t", 2.1, "the coefficient for mining hard examples")
flags.DEFINE_float("negative_t", 1.5, "the coefficient for mining hard examples")
flags.DEFINE_float("l2_regularizer_scale", 0, "the scale for l2 regularizer")
flags.DEFINE_string("optimizer", "Adagrad", "the name of optimizer")
flags.DEFINE_boolean("use_lower_loss", True, "Whether to use lower loss")
flags.DEFINE_boolean("use_batch_norm", False, "Whether to use batch normalization for hidden layers")
flags.DEFINE_integer("num_negative_samples", 10, "how many negative samples should be used for an instance")
flags.DEFINE_boolean("init_embedding", True, "Whether to initialize embeddings from tables")
flags.DEFINE_boolean("log_device_placement", False, "whether to print device placement log")
flags.DEFINE_boolean("profile", False, "whether to print device placement log")
flags.DEFINE_integer("intra_op_parallelism_threads", 64, "for cpu performance optimization")
flags.DEFINE_integer("inter_op_parallelism_threads", 64, "for cpu performance optimization")
flags.DEFINE_integer("log_step_count_steps", 500, "log_step_count_steps")
flags.DEFINE_integer("save_summary_steps", 500, "save_summary_steps")
flags.DEFINE_integer("tf_random_seed", 0, "random seed")
flags.DEFINE_boolean("evaluate", False, "whether to start evaluation process")
flags.DEFINE_boolean("export", False, "whether to export saved model periodically")
flags.DEFINE_boolean("warm_start", False, "whether to load weight from a pre-trained model")
flags.DEFINE_boolean("use_feature", False, "Whether to use extra feature except of text sequence")
flags.DEFINE_string("init_checkpoint", "", "where pre-trained model stores in")

FLAGS = flags.FLAGS


def create_feature_columns():
    anchor_cate = fc.categorical_column_with_hash_bucket("anchor_cate", 15000, dtype=tf.int64)
    anchor_commodity = fc.categorical_column_with_hash_bucket("anchor_commodity", 3000, dtype=tf.int64)

    higher_cate = fc.categorical_column_with_hash_bucket("higher_cate", 15000, dtype=tf.int64)
    higher_commodity = fc.categorical_column_with_hash_bucket("higher_commodity", 3000, dtype=tf.int64)

    lower_cate = fc.categorical_column_with_hash_bucket("lower_cate", 15000, dtype=tf.int64)
    lower_commodity = fc.categorical_column_with_hash_bucket("lower_commodity", 3000, dtype=tf.int64)

    initializer = tf.contrib.layers.xavier_initializer()
    #with tf.variable_scope("feature_embedding",
    #                       regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.l2_regularizer_scale)):
    cate_emb = tf.feature_column.shared_embedding_columns([anchor_cate, higher_cate, lower_cate], 32, initializer=initializer)
    commod_emb = tf.feature_column.shared_embedding_columns([anchor_commodity, higher_commodity, lower_commodity], 16, initializer=initializer)

    columns = []
    columns += cate_emb
    columns += commod_emb
    return columns


def input_fn(filepattern, train=True):
    print("input file pattern:", filepattern)
    d = tf.data.Dataset.list_files(filepattern)
    dataset = d.apply(tf.contrib.data.sloppy_interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=FLAGS.buffer_size),
                      cycle_length=FLAGS.num_parallel_readers))
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    feature_spec = {
        "anchor_item": tf.FixedLenFeature([], tf.int64, default_value=0),  # anchor for prediction
        "anchor_cate": tf.FixedLenFeature([], tf.int64, default_value=0),
        "anchor_commodity": tf.FixedLenFeature([], tf.int64, default_value=0),
        "anchor_title_word": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "anchor_title_word_tag": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "anchor_title_word_weight": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.float32),
        "anchor_title_char": tf.FixedLenFeature([FLAGS.max_char_seq_len], tf.int64),
        "higher_title_word": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "higher_title_word_tag": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "higher_title_word_weight": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.float32),
        "higher_title_char": tf.FixedLenFeature([FLAGS.max_char_seq_len], tf.int64),
        "lower_title_word": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "lower_title_word_tag": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "lower_title_word_weight": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.float32),
        "lower_title_char": tf.FixedLenFeature([FLAGS.max_char_seq_len], tf.int64)
    }
    if FLAGS.use_feature:
        feature_spec.update({
            "higher_cate": tf.FixedLenFeature([], tf.int64, default_value=0),
            "higher_commodity": tf.FixedLenFeature([], tf.int64, default_value=0),
            "lower_cate": tf.FixedLenFeature([], tf.int64, default_value=0),
            "lower_commodity": tf.FixedLenFeature([], tf.int64, default_value=0),
        })

    def parser(x):
        features = tf.parse_example(x, feature_spec)
        return features, 1

    dataset = dataset.map(parser, num_parallel_calls=FLAGS.num_parallel_parser)
    if train:
        dataset = dataset.repeat()
    return dataset


def model_statistics(model):
    def size(v):
        return reduce(lambda x, y: x * y, v.shape)
    print("------------------variables----------------------")
    num_memory = 0
    variables = model.get_variable_names()
    for var in variables:
        value = model.get_variable_value(var)
        print(var, "shape:", value.shape)
        if len(value.shape) > 0:
            num_memory += size(value)
    print("----------------variables end--------------------")
    print("total model size:", num_memory)


def main(_):
    for key in FLAGS.__dict__['__flags'].keys():
        if key in ('h', 'help'):
            continue
        print("%s=%s" % (key, str(FLAGS.__dict__['__flags'][key])))
    print("tensorflow version:", tf.__version__)
    if FLAGS.volumes:
        volumes = FLAGS.volumes.split(",")
        train_files = volumes[0] + "/*.tfr"
        eval_files = volumes[1] + "/*.tfr" if len(volumes) > 1 else train_files
    else:
        train_files = FLAGS.train_data
        eval_files = FLAGS.eval_data
    print("train_data:", train_files)
    print("eval_data:", eval_files)
    if "TF_CONFIG" in os.environ:
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))
    model_params = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate,
        'word_cnn_filter_sizes': map(int, FLAGS.word_cnn_filter_sizes.split(',')),
        'word_cnn_num_filters': FLAGS.word_cnn_num_filters,
        'char_cnn_filter_sizes': map(int, FLAGS.char_cnn_filter_sizes.split(',')),
        'char_cnn_num_filters': FLAGS.char_cnn_num_filters,
        'word_vocab_size': FLAGS.word_vocab_size,
        'char_vocab_size': FLAGS.char_vocab_size,
        'tag_vocab_size': FLAGS.tag_vocab_size,
        'word_embedding_size': FLAGS.word_embedding_size,
        'char_embedding_size': FLAGS.char_embedding_size,
        'tag_embedding_size': FLAGS.tag_embedding_size,
        'margin': FLAGS.margin,
        'negative_margin': FLAGS.negative_margin,
        'smooth': FLAGS.smooth,
        'l2_scale': FLAGS.l2_regularizer_scale,
        't': FLAGS.t, 'negative_t': FLAGS.negative_t,
        'use_lower_loss': FLAGS.use_lower_loss,
        'hidden_units': map(int, FLAGS.hidden_units.split(',')),
        'activations': FLAGS.activations.split(','),
        'use_batch_norm': FLAGS.use_batch_norm,
        'use_feature': FLAGS.use_feature,
        'num_negative_samples': FLAGS.num_negative_samples,
        'warm_start': FLAGS.warm_start,
        'init_checkpoint': FLAGS.init_checkpoint
    }
    if FLAGS.use_feature:
        feature_columns = create_feature_columns()
        num_cols = len(feature_columns)
        anchor_feature_columns = [feature_columns[i] for i in range(0, num_cols, 3)]
        positive_feature_columns = [feature_columns[i] for i in range(1, num_cols, 3)]
        negative_feature_columns = [feature_columns[i] for i in range(2, num_cols, 3)]
        print("anchor_feature_columns =", anchor_feature_columns)
        print("higher_feature_columns =", positive_feature_columns)
        print("lower_feature_columns =", negative_feature_columns)
        model_params.update({
            'anchor_feature_columns': anchor_feature_columns,
            'higher_feature_columns': positive_feature_columns,
            'lower_feature_columns': negative_feature_columns
        })
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        gpu_options=tf.GPUOptions(allow_growth=True, force_gpu_compatible=True)
    )
    if FLAGS.evaluate:
        cluster = {'chief': ['localhost:2221'], 'worker': ['localhost:2222']}
        os.environ['TF_CONFIG'] = json.dumps({'cluster': cluster, 'task': {'type': 'evaluator', 'index': 0}})
        config = tf.estimator.RunConfig(session_config=session_config)
        model = SemanticModel(params=model_params, optimizer=FLAGS.optimizer, model_dir=FLAGS.checkpointDir, config=config)
        evaluation_listener(model, train_files, eval_files)
        return
    else:
        cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl', 16, 0, 0)
        distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus, cross_tower_ops=cross_tower_ops)
        config = tf.estimator.RunConfig(
            distribute=distribution,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            keep_checkpoint_max=FLAGS.keep_checkpoint_max, session_config=session_config,
            log_step_count_steps=FLAGS.log_step_count_steps,
            save_summary_steps=FLAGS.save_summary_steps)
    if FLAGS.tf_random_seed != 0:
        config.replace(tf_random_seed=FLAGS.tf_random_seed)

    # if FLAGS.init_embedding and FLAGS.tables:
    #     tables = FLAGS.tables.split(',')
    #     word_dict = tables[0]
    #     word_embed = tables[1]
    #     word_initializer = build_embedding_initializer(word_dict, word_embed, FLAGS.word_vocab_size, FLAGS.word_embedding_size)
    #     model_params['word_initializer'] = word_initializer
    #     if len(tables) > 2:
    #         char_dict = tables[2]
    #         char_embed = tables[3]
    #         char_initializer = build_embedding_initializer(char_dict, char_embed, FLAGS.char_vocab_size, FLAGS.char_embedding_size)
    #         model_params['char_initializer'] = char_initializer

    model = SemanticModel(params=model_params, optimizer=FLAGS.optimizer, model_dir=FLAGS.checkpointDir, config=config)

    train_hooks = [tf.train.ProfilerHook(save_secs=FLAGS.save_checkpoints_secs, output_dir=FLAGS.buckets)] if FLAGS.profile else []

    logging.info("before train")
    model.train(lambda: input_fn(train_files, True), max_steps=FLAGS.train_steps, hooks=train_hooks)
    logging.info("after train")

    if config.is_chief:
        logging.info("exporting model ...")
        serving_input_receiver_fn = get_serving_input_fn()
        model.export_savedmodel(FLAGS.outputs, serving_input_receiver_fn)
        logging.info("print model variables ...")
        model_statistics(model)
    logging.info("finish main")


def get_serving_input_fn():
    input_feature_spec = {
        "anchor_item": tf.FixedLenFeature([], tf.int64, default_value=0),  # anchor for prediction
        "anchor_cate": tf.FixedLenFeature([], tf.int64, default_value=0),
        "anchor_commodity": tf.FixedLenFeature([], tf.int64, default_value=0),
        "anchor_title_word": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "anchor_title_word_tag": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.int64),
        "anchor_title_word_weight": tf.FixedLenFeature([FLAGS.max_word_seq_len], tf.float32),
        "anchor_title_char": tf.FixedLenFeature([FLAGS.max_char_seq_len], tf.int64)
    }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(input_feature_spec)


def evaluation_listener(model, train_files, eval_files):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_files, True),
        max_steps=FLAGS.train_steps
    )
    serving_input_receiver_fn = get_serving_input_fn()
    if FLAGS.export:
        exporter = tf.estimator.LatestExporter("saved_models", serving_input_receiver_fn, exports_to_keep=5)
    else:
        exporter = None
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_files, False),
        throttle_secs=FLAGS.save_checkpoints_secs,
        steps=None, exporters=exporter
    )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
