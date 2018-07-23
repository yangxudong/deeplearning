#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/26 15:57
# @Author  : linhuan
# @File    : telenet_fn.py

import os
import pickle
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import telenet_model_mobilenet
import sys
#sys.path.append('../')
#from global_configuration import config

def telenet_model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        telenet = telenet_model_mobilenet.TeleNet(phase='Train', hidden_nums=128, layers_nums=1, seq_length=16,num_classes=2)
    else:
        telenet = telenet_model_mobilenet.TeleNet(phase='Test', hidden_nums=128, layers_nums=1, seq_length=16,num_classes=2)
    input_tensor=features[:,0,:,:,:]
    user_tensor=features[:,1:,:,:,:]
    logits_pr, logits_before = telenet.build_telenet(candidata=input_tensor,browsedseq=user_tensor)
    input_labels=labels
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_before, labels=input_labels),name='mean_cost')

    # 预测结果
    # 请注意 predict_max_idx 的 name，在测试model时会用到它
    with tf.name_scope('cal_accuracy'):  #sigmoid
        # predict_max_idx = tf.argmax(logits_before, axis=1, name='predict_max_idx')
        predict_round = tf.round(logits_pr)
        labels_round = tf.round(input_labels)
        predict_correct_vec = tf.equal(predict_round, labels_round)
        accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    #eval
    # Convert to shape [batch_size]
    acc = tf.metrics.accuracy( tf.squeeze(input_labels, 1), tf.squeeze(logits_pr, 1) )
    metrics = {'accuracy': acc}
   #================================================================================


    starter_learning_rate =0.1# config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, tf.train.get_global_step(),
                                               2000, 0.6,
                                               staircase=True)
    all_vars = tf.contrib.slim.get_variables_to_restore()  # tf.get_collection(tf.GraphKeys.VARIABLES)
    vars_base = [v for v in all_vars if
                            v.name.split('/')[0] == 'MobilenetV2' ]
    vars_add = [v for v in all_vars if
                          v.name.split('/')[0] != 'MobilenetV2' ]
    new_all_vars=[]
    new_all_vars.extend(vars_base)
    new_all_vars.extend(vars_add)
  #  #==============变量初始化================
  #  pretrain_saver = tf.train.Saver(vars_base)
  #  def init_fn(scaffold, sess):
  #      pretrain_saver.restore(sess, config.cfg.mobilenetv2path)
  #  scaffold = tf.train.Scaffold(init_fn=init_fn)
  #  #=======================
    with tf.variable_scope('ops_ada'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_add = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, var_list=vars_add,global_step=tf.train.get_global_step())
            optimizer = optimizer_add

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=cost, train_op=optimizer)#,scaffold=scaffold)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=cost, eval_metric_ops=metrics)
