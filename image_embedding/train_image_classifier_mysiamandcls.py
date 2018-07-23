#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import numpy as np
sys.path.append('/home/lunxi.yuan/inception_v3_transfer/inception_v3/models/research/slim')
sys.path.append("/home/lunxi.yuan/python3/lib/python3.6/site-packages")
import tensorflow as tf
#from datasets import dataset_factory
from datasets import dataset_classifyv2
#from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
#from nets import inception


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'checkpoint_steps', 2000,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_integer(
    'num_samples', 0, 'The num of the dataset to load.')

tf.app.flags.DEFINE_integer(
    'num_classes', 1000, 'The num of the classes')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
    
    
tf.app.flags.DEFINE_string(
    'dataset_dir2', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'labels_to_names_path', None, 'Label names file path.')
    
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 299, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')
    
tf.app.flags.DEFINE_integer(
    'reducedims', 128, 'The num of the dims of  reduce dimmension')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.
  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.
  Returns:
    A `Tensor` representing the learning rate.
  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.
  Args:
    learning_rate: A scalar or `Tensor` learning rate.
  Returns:
    An instance of an optimizer.
  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.
  Note that the init_fn is only run when initializing the model during the very
  first global step.
  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    checkpoint_path=tf.train.latest_checkpoint(FLAGS.train_dir)
    return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_model_variables(),
      ignore_missing_vars=FLAGS.ignore_missing_vars)
  
  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.
  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train
  
  
def savefeature(bottleneck_values,featurepath):
    bottleneck_values = np.squeeze(bottleneck_values)  # 将四维数组压缩成一维数组
   
    wholefilename='featureset_test.txt'
    featurefile=os.path.join(featurepath,wholefilename)
    if not os.path.exists(featurefile):
        start_index=0
    else:
        with open(featurefile, 'r') as f:
            lines= f.readlines()
            index, feature = lines[-1].split('_--_iid')
    
        start_index=int(index)+1
    
    for k in range(FLAGS.batch_size):
        
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)  
        with open(featurefile, 'a') as featurefiles:
            content=str(start_index)+'_--_iid'+bottleneck_string+'\n'
            featurefiles.write(content)
        start_index=start_index+1
 
def loss_with_spring(logits_left,logits_right,labels):
    margin = 1.0
    #tf.cast(labels,tf.float32)
    
    labels_t = tf.cast(labels,tf.float32)
    
    left_feature=tf.nn.l2_normalize(logits_left, dim = 1)
    right_feature=tf.nn.l2_normalize(logits_right, dim = 1)
    
    labels_f = tf.subtract(1.0, labels_t, name="1-yi")          # labels_ = !labels;
    eucd2 = tf.pow(tf.subtract(left_feature, right_feature), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
    pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
    # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
    # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")/4
    return loss,eucd
  
def siamese_loss(out1,out2,y,Q=5):
    y = tf.cast(y,tf.float32)
    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1,out2)),1))   
    pos = tf.multiply(tf.multiply(y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(1-y,2*Q),tf.exp(-2.77/Q*E_w))                
    loss = pos + neg                 
    loss = tf.reduce_mean(loss)              
    return loss 

def contrastive_loss(model1, model2, y, margin):
    with tf.name_scope("contrastive-loss"):
        y = tf.cast(y,tf.float32)
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=False))
        tmp= y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_mean(tmp + tmp2)/2     

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as graph:
    
    # Create global_step
    
    global_step = slim.create_global_step() 
    ######################
    # Select the dataset #
    ######################
    
    dataset = dataset_classifyv2.get_dataset(
         FLAGS.dataset_dir, FLAGS.num_samples, FLAGS.num_classes, FLAGS.labels_to_names_path)
    
    dataset2 = dataset_classifyv2.get_dataset(
        FLAGS.dataset_dir2, FLAGS.num_samples, FLAGS.num_classes, FLAGS.labels_to_names_path)
    
    
    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=FLAGS.num_classes,
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    
    
    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10*FLAGS.batch_size)
    [image_left, label_left,pairflag_left] = provider.get(['image', 'label','pairflag'])
    label_left -= FLAGS.labels_offset
    
    
    provider2 = slim.dataset_data_provider.DatasetDataProvider(
        dataset2,
        shuffle=False,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10*FLAGS.batch_size)
    [image_right, label_right,pairflag_right] = provider2.get(['image', 'label','pairflag'])
    label_right -= FLAGS.labels_offset
    
    

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    train_image_size = FLAGS.train_image_size or network_fn.default_image_size

    image_left = image_preprocessing_fn(image_left, train_image_size, train_image_size)
    image_right = image_preprocessing_fn(image_right, train_image_size, train_image_size)

    images_left, labels_left,pairflags_left = tf.train.batch(
        [image_left, label_left,pairflag_left],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels_left = slim.one_hot_encoding(
            labels_left, FLAGS.num_classes)
        
    batch_queue_left = slim.prefetch_queue.prefetch_queue(
            [images_left, labels_left,pairflags_left], capacity=2)
        
    images_right, labels_right,pairflags_right = tf.train.batch(
        [image_right, label_right,pairflag_right],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels_right = slim.one_hot_encoding(
            labels_right, FLAGS.num_classes)
        
    batch_queue_right = slim.prefetch_queue.prefetch_queue(
            [images_right, labels_right,pairflags_right], capacity=2)

    ####################
    # Define the model #
    ####################

    """Allows data parallelism by creating multiple clones of network_fn."""
    images_left, labels_left,pairflags_left = batch_queue_left.dequeue()
    _, end_points_left = network_fn(images_left,reuse=False)
  
    images_right, labels_right,pairflags_right = batch_queue_right.dequeue()
    _, end_points_right= network_fn(images_right,reuse=True)
    
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
        }
      
      
    with tf.variable_scope('ReduceDims'):
        with slim.arg_scope([slim.conv2d],stride=1, padding='SAME',weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
            with slim.arg_scope([slim.conv2d],weights_initializer=slim.variance_scaling_initializer(),
                                    activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=True):
                    net = slim.dropout(end_points_left['AvgPool_1a'], keep_prob=0.8, scope='Dropout_1b')
                    logits_left_256 = slim.conv2d(net,256, [1, 1],scope='Conv2d_1d_1x1')
                    net = slim.dropout(logits_left_256, keep_prob=0.8, scope='Dropout_1b')
                    logits_left = slim.conv2d(net, FLAGS.num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1e_1x1')
                    logits_left = tf.squeeze(logits_left, [1, 2], name='logitssqueeze')
                    logits_left_256 = tf.squeeze(logits_left_256, [1, 2], name='logitssqueeze256')
    with tf.variable_scope('ReduceDims',reuse=True):
        with slim.arg_scope([slim.conv2d],stride=1, padding='SAME',weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
            with slim.arg_scope([slim.conv2d],weights_initializer=slim.variance_scaling_initializer(),
                                    activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=True):
                    net = slim.dropout(end_points_right['AvgPool_1a'], keep_prob=0.8, scope='Dropout_1b')
                    logits_right_256 = slim.conv2d(net,256, [1, 1],scope='Conv2d_1d_1x1')
                    net = slim.dropout(logits_right_256, keep_prob=0.8, scope='Dropout_1b')
                    logits_right = slim.conv2d(net, FLAGS.num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1e_1x1')
                    logits_right = tf.squeeze(logits_right, [1, 2], name='logitssqueeze')
                    logits_right_256=tf.squeeze(logits_right_256, [1, 2], name='logitssqueeze256')
      
     
    #############################
    # Specify the loss function #
    #############################
    losses=[]
    loss,eucd=loss_with_spring(logits_left_256,logits_right_256,pairflags_left)
    losses.append(loss)
    #loss_aux,_=loss_with_spring(end_points_left['AuxLogits'],end_points_right['AuxLogits'],pairflags_left)
    #losses.append(loss_aux)
    #loss = contrastive_loss(logits_left, logits_right, labels_left,2.0)
    regularization_losses = tf.losses.get_regularization_loss(
        scope='ReduceDims',name='total_regularization_loss')
    losses.append(regularization_losses)
        
    cross_entropy_left = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_left, labels=labels_left)
    cross_entropy_mean_left = tf.reduce_mean(cross_entropy_left)
    losses.append(cross_entropy_mean_left)
    #加入辅助分类器
    cross_entropy_leftaux = tf.nn.softmax_cross_entropy_with_logits(
            logits=end_points_left['AuxLogits'], labels=labels_left)
    cross_entropy_mean_leftaux = tf.reduce_mean(cross_entropy_leftaux)*0.4
    losses.append(cross_entropy_mean_leftaux)
    
    """
    cross_entropy_right = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits_right, labels=labels_right)
    cross_entropy_mean_right = tf.reduce_mean(cross_entropy_right)
    losses.append(cross_entropy_mean_right)
    """
          
    total_loss = tf.add_n(losses, name='total_loss')
    # Variables to train.
    variables_to_train = _get_variables_to_train()
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=FLAGS.trainable_scopes)
    with tf.control_dependencies(update_ops):  
        #train_step = tf.train.MomentumOptimizer(0.0001, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)
        train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, 0.9).minimize(
              total_loss,global_step=global_step,var_list=variables_to_train)
              
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))         
    for loss in  losses:
        summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
         
    
    # Add config to avoid 'could not satisfy explicit device' problem 
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    init_fn=_get_init_fn()
    
    #loss_summary = tf.summary.scalar('loss', total_loss)
    # Merge all summaries together.
    #summary_op = tf.summary.merge(loss_summary, name='summary_op')

    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    with tf.Session(graph=graph,config=sess_config) as sess:
        #sess.run(init_l)
        init = tf.global_variables_initializer().run()
        init_fn(sess)
        #sess.run()
        train_summary_dir = os.path.join(FLAGS.train_dir, 'summaries')
     
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)  
        coordtest=tf.train.Coordinator()
        threadstest= tf.train.start_queue_runners(sess=sess,coord=coordtest)
        try:
            for j in range(FLAGS.max_number_of_steps):
                if coordtest.should_stop():
                    break
                _,lossvalue,train_summaries,step= sess.run([train_step,total_loss,summary_op,global_step])
                
                if j%FLAGS.log_every_n_steps ==0:
                    if not os.path.exists(FLAGS.train_dir):
                        os.makedirs(FLAGS.train_dir)
                    train_summary_writer.add_summary(train_summaries, j)
                    print("step=%d - Loss = %.3f"%(j, lossvalue))
                    #print(label_lefts,label_rights)
                    #print(pairflag_lefts,pairflag_rights)
                #savefeature(logits_lefts,'.')
                #savefeature(logits_rights,'.')
                if j%FLAGS.checkpoint_steps ==0 or (j+1)==FLAGS.max_number_of_steps:
                    if not os.path.exists(FLAGS.train_dir):
                        os.makedirs(FLAGS.train_dir)
                    path=os.path.join(FLAGS.train_dir,'model.ckpt')
                    saver.save(sess, path,global_step=step+1)
                    
                
        except tf.errors.OutOfRangeError:
            print ('Done testing -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coordtest.request_stop()
        coordtest.join(threadstest)


if __name__ == '__main__':
  tf.app.run()