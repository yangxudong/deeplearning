#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/7/09 16:22
# @Author  : linhuan
# @File    : estimator_release.py
import os
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
sys.path.append(os.getcwd())
import telenet_fn
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "/data4/ImageData/TelenetData/SevenDtfRFULL", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "/data4/ImageData/TelenetData/SevenDtfRFULL", "Path to the evaluation data.")
flags.DEFINE_integer("train_steps", 100, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "dataset shuffle buffer size")
flags.DEFINE_integer("num_parallel_calls", 8, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 50, "number of steps for saving checkpoint")
#flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
#flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
#                    "Comma-separated list of hostname:port pairs")
#flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
#                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
#flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")

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


def name2label_sigmoid(name):
    label = np.zeros(1,np.float32)
    if name == '0':
        label[0]=0
    else:
        label[0]=1
    return label

# Dataset
def train_input_fn(path_dataset,num_epochs,batch_size):
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''
    train_dataset = tf.data.TFRecordDataset(path_dataset)
    train_dataset = train_dataset.shuffle(FLAGS.shuffle_buffer_size)
    train_dataset = train_dataset.map(read_and_decode_new, num_parallel_calls=FLAGS.num_parallel_calls)
    #dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)#  shuffle 怎么做
    # num_epochs 为整个数据集的迭代次数
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)  # make sure you always have one batch ready to serve
    #train_iterator = train_dataset.make_one_shot_iterator()
    #features, labels = train_iterator.get_next()
    #return features, labels
    return train_dataset

def read_and_decode_new(serialized_example):
    #解析读入的一个样例
    features = tf.parse_single_example(serialized_example,features={
        'image/encoded': tf.FixedLenFeature([],tf.string),
        'image/seq_list': tf.FixedLenFeature([16],tf.string),
       #  'image/format': tf.FixedLenFeature([],tf.string),
        'image/class/label': tf.FixedLenFeature([],tf.int64)
        })
    #将字符串解析成图像对应的像素数组
    # image = tf.decode_raw(features['image/encoded'],tf.uint8)

    # label = tf.cast(features['image/class/label'],tf.int32)
    label = tf.cast(features['image/class/label'], tf.float32)
    label = tf.reshape(label, [-1])

    img_data_jpg = tf.image.decode_jpeg(features['image/encoded'])
    image = tf.reshape(img_data_jpg, [128, 128, 3])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1.0 #

    img_data_seq = tf.image.decode_jpeg(features['image/seq_list'][0]) # 
    image_seq = tf.reshape(img_data_seq, [1,128, 128, 3])
    image_seq = tf.cast(image_seq, tf.float32) * (2. / 255) - 1.0 
    for i in range(1,16): #  
        img_data_s = tf.image.decode_jpeg(features['image/seq_list'][i])
        image_s = tf.reshape(img_data_s, [1,128, 128, 3])
        image_s = tf.cast(image_s, tf.float32) * (2. / 255) - 1.0  # 
        image_seq=tf.concat([image_seq,image_s],0)
    image = tf.expand_dims(image, 0)
    image_all=tf.concat([image,image_seq],0)
    return image_all,label



#def train_telenet(tfrecord_dir, weights_path=None):
#    """
#    :param tfrecord_dir:
#    :param weights_path:
#    :return:
#    """
#    # Set sess configuration
#    sess_config = tf.ConfigProto()
#    #sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
#    runconfig=tf.estimator.RunConfig(session_config=sess_config,keep_checkpoint_max=3)
#    telenet_ctr = tf.estimator.Estimator( model_fn=telenet_fn.telenet_model_fn, model_dir='./model_dir')
#    telenet_ctr.train(input_fn=lambda:train_input_fn(tfrecord_dir,num_epochs=20,batch_size=256),steps=100)
#    return

def main(unused_argv):
  set_tfconfig_environ()
  model = tf.estimator.Estimator(
    model_fn=telenet_fn.telenet_model_fn, 
    config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  )
  if isinstance(FLAGS.train_data, str) and os.path.isdir(FLAGS.train_data):
    train_files = [FLAGS.train_data + '/' + x for x in os.listdir(FLAGS.train_data)] if os.path.isdir(FLAGS.train_data) else FLAGS.train_data
  else:
    train_files = FLAGS.train_data
  if isinstance(FLAGS.eval_data, str) and os.path.isdir(FLAGS.eval_data):
    eval_files = [FLAGS.eval_data + '/' + os.listdir(FLAGS.eval_data)[0]] if os.path.isdir(FLAGS.eval_data) else FLAGS.eval_data
  else:
    eval_files = FLAGS.eval_data
  train_spec = tf.estimator.TrainSpec(
    input_fn=lambda:train_input_fn(train_files, num_epochs=20, batch_size=FLAGS.batch_size),
    max_steps=FLAGS.train_steps
  )
  eval_spec = tf.estimator.EvalSpec(input_fn=lambda:train_input_fn(eval_files, num_epochs=1, batch_size=FLAGS.batch_size), throttle_secs=1800)
  print("before train and evaluate")
  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
  print("after train and evaluate")

  # Evaluate accuracy.
  results = model.evaluate(input_fn=lambda:train_input_fn(tfrecord_dir, num_epochs=1, batch_size=FLAGS.batch_size))
  for key in sorted(results): print('%s: %s' % (key, results[key]))
  print("after evaluate")

  if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    print("exporting model ...")
    feature_spec = {
        'image/encoded': tf.FixedLenFeature([],tf.string),
        'image/seq_list': tf.FixedLenFeature([16],tf.string),
        'image/class/label': tf.FixedLenFeature([],tf.int64)
    }
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
  print("quit main")


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
  # tfrecord_dir= '/data4/ImageData/TelenetData/SevenDtfRFULL'
  # data_set='train'
  # tffile = [os.path.join(tfrecord_dir,n) for n in os.listdir(tfrecord_dir) if data_set in n]
  # print('tf path:',tffile)
  # train_telenet(tffile)
  # print('Done')
