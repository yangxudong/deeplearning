from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers
from tensorflow import logging

flags = tf.app.flags
# params for pai tensorflow
flags.DEFINE_string("checkpointDir", "model_dir", "Directory where checkpoints and event logs are written to.")
flags.DEFINE_string("tables", "", "input tables")
flags.DEFINE_string("volumes", "", "input volume path")
flags.DEFINE_string("outputs", "output", "Directory where model is written to")
flags.DEFINE_string("buckets", "", "input oss bucket info")
flags.DEFINE_integer("gpuRequired", 100, "number of gpu cores, 100 represent to one card")
flags.DEFINE_integer("cpuRequired", 600, "number of cpu cores, 100 represent to one card")
flags.DEFINE_integer("num_gpus", 8, "number of gpu cores")
# hyper params
flags.DEFINE_string("train_data", "data/samples", "Path to the training data")
flags.DEFINE_string("eval_data", "data/samples", "Path to the evaluation data.")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("save_checkpoints_secs", 720, "Save checkpoints every this many seconds")
flags.DEFINE_integer("keep_checkpoint_max", 8, "how many checkpoints should be keep")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("num_parallel_readers", 10, "number of parallel readers for training data")
flags.DEFINE_integer("num_parallel_parser", 8, "number of parallel parser for training data")
flags.DEFINE_integer("buffer_size", 128 * 1024 * 1024, "dataset read buffer size")
flags.DEFINE_integer("word_vocab_size", 800000, "the length of word vocabulary")
flags.DEFINE_integer("cate_vocab_size", 23000, "the length of cate vocabulary")
flags.DEFINE_integer("word_embedding_size", 128, "the dim of word embedding vector")
flags.DEFINE_integer("cate_embedding_size", 128, "the dim of cate embedding vector")
flags.DEFINE_float("learning_rate", 0.03, "Learning rate")
flags.DEFINE_string("optimizer", "Adagrad", "the name of optimizer")
flags.DEFINE_integer("num_negative_samples", 64, "how many negative samples should be used for an instance")
flags.DEFINE_boolean("log_device_placement", False, "whether to print device placement log")
flags.DEFINE_boolean("profile", False, "whether to print device placement log")
flags.DEFINE_integer("intra_op_parallelism_threads", 64, "for cpu performance optimization")
flags.DEFINE_integer("inter_op_parallelism_threads", 64, "for cpu performance optimization")
flags.DEFINE_integer("log_step_count_steps", 500, "log_step_count_steps")
flags.DEFINE_integer("tf_random_seed", 0, "random seed")
flags.DEFINE_boolean("evaluate", False, "whether to start evaluation process")

FLAGS = flags.FLAGS


class Doc2Vec(tf.estimator.Estimator):
    """Doc2Vec model (Skipgram)."""
    def __init__(self,
          params,
          model_dir=None,
          optimizer='Adagrad',
          config=None
        ):
        if not optimizer: optimizer = 'Adagrad'
        self.optimizer = optimizers.get_optimizer_instance(optimizer, params["learning_rate"])

        def _model_fn(features, labels, mode, params):
            vocabulary_size = params["vocab_size"]
            doc_vocab_size = params["cate_vocab_size"]
            embedding_size = params["embedding_size"]
            doc_embedding_size = params["doc_embedding_size"]
            if params["embedding_merge"] == "concat":
                hidden_units = embedding_size + doc_embedding_size
            else:
                assert embedding_size == doc_embedding_size
                hidden_units = embedding_size

            # Define Embeddings:
            embeddings = tf.get_variable(name="word_embeddings", shape=[vocabulary_size, embedding_size],
                                         dtype=tf.float32, initializer=tf.random_uniform_initializer(-1.0, 1.0))
            doc_embeddings = tf.get_variable(name="doc_embeddings", shape=[doc_vocab_size, doc_embedding_size],
                                             dtype=tf.float32, initializer=tf.random_uniform_initializer(-1.0, 1.0))

            # NCE loss parameters
            init_stddev = 1.0 / np.sqrt(hidden_units)
            nce_weights = tf.get_variable("nce_weights", [vocabulary_size, hidden_units], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=init_stddev))
            nce_biases = tf.get_variable("nce_biases", [vocabulary_size], dtype=tf.float32,
                                         initializer=tf.zeros_initializer())

            embed = tf.nn.embedding_lookup(embeddings, features["context_word"])
            doc_embed = tf.nn.embedding_lookup(doc_embeddings, features["cate_id"])

            if params["embedding_merge"] == "concat":
                final_embed = tf.concat([embed, doc_embed], axis=1)
            else:
                final_embed = (embed + doc_embed) / 2.0

            # Get loss from prediction
            num_sampled = params["num_negative_samples"]
            loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, tf.expand_dims(labels, 1), final_embed,
                                                 num_sampled, vocabulary_size, remove_accidental_hits=True))
            global_step = tf.train.get_or_create_global_step()
            train_op = self.optimizer.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(Doc2Vec, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config, params=params)


def input_fn(filepattern, train=True):
    print("input file pattern:", filepattern)
    d = tf.data.Dataset.list_files(filepattern)
    dataset = d.apply(tf.contrib.data.sloppy_interleave(
        lambda filename: tf.data.TFRecordDataset(filename, buffer_size=FLAGS.buffer_size),
        cycle_length=FLAGS.num_parallel_readers))
    dataset = dataset.batch(FLAGS.batch_size)
    feature_spec = {
        "context_word": tf.FixedLenFeature([], tf.int64, default_value=0),
        "target_word": tf.FixedLenFeature([], tf.int64, default_value=0),
        "cate_id": tf.FixedLenFeature([], tf.int64, default_value=0)
    }

    def parser(x):
        features = tf.parse_example(x, feature_spec)
        target = features.pop("target_word")
        return features, target

    dataset = dataset.map(parser, num_parallel_calls=FLAGS.num_parallel_parser)
    if train:
        dataset = dataset.repeat()
    return dataset


def main(unused_argv):
    for key in FLAGS.__dict__['__flags'].keys():
        if key in ('h', 'help'):
            continue
        print("%s=%s" % (key, str(FLAGS.__dict__['__flags'][key])))
    print("tensorflow version:", tf.__version__)
    if FLAGS.volumes:
        volumes = FLAGS.volumes.split(",")
        train_files = volumes[0] + "/*.tfr"
        if FLAGS.tf_random_seed != 0:  # for experiment
            train_files = volumes[0] + "/*_0000*.tfr"
        eval_files = volumes[1] + "/*.tfr" if len(volumes) > 1 else train_files
    else:
        train_files = FLAGS.train_data
        eval_files = FLAGS.eval_data
    print("train_data:", train_files)
    print("eval_data:", eval_files)
    model_params = {
        'learning_rate': FLAGS.learning_rate,
        'vocab_size': FLAGS.word_vocab_size,
        'cate_vocab_size': FLAGS.cate_vocab_size,
        'embedding_size': FLAGS.word_embedding_size,
        'doc_embedding_size': FLAGS.cate_embedding_size,
        'num_negative_samples': FLAGS.num_negative_samples,
        'embedding_merge': 'avg'
    }

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
        model = Doc2Vec(params=model_params, optimizer=FLAGS.optimizer, model_dir=FLAGS.checkpointDir, config=config)
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
            save_summary_steps=FLAGS.log_step_count_steps)
    if FLAGS.tf_random_seed != 0:
        config.replace(tf_random_seed=FLAGS.tf_random_seed)

    model = Doc2Vec(params=model_params, optimizer=FLAGS.optimizer, model_dir=FLAGS.checkpointDir, config=config)

    train_hooks = [tf.train.ProfilerHook(save_secs=FLAGS.save_checkpoints_secs, output_dir=FLAGS.buckets)] if FLAGS.profile else []

    logging.info("before train")
    model.train(lambda: input_fn(train_files, True), max_steps=FLAGS.train_steps, hooks=train_hooks)
    logging.info("finish main")


def evaluation_listener(model, train_files, eval_files):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_files, True),
        max_steps=FLAGS.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_files, False),
        throttle_secs=FLAGS.save_checkpoints_secs,
        steps=None
    )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
