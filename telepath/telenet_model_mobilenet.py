#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implement the crnn model mentioned in An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition paper
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
slim = tf.contrib.slim
import sys
import mobilenet_v2
import telenet_basenet


class TeleNet(telenet_basenet.TeleBaseModel):
    """
        Implement the crnn model for squence recognition
    """
    def __init__(self, phase, hidden_nums, layers_nums, seq_length, num_classes,use_bn=False):
        """

        :param phase:
        """
        super(TeleNet, self).__init__()  # call  TeleBaseModel __int__
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__seq_length = seq_length
        if False:
            self.__num_classes = num_classes
        else:
            self.__num_classes=1   #  use  sigmoid
        self.__use_bn=use_bn
        return

    @property
    def phase(self):
        """

        :return:
        """
        return self.__phase

    @phase.setter
    def phase(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, str):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if value.lower() not in ['test', 'train']:
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
        return

    def __conv_stage(self, inputdata, out_dims, name=None):
        """
        Traditional conv stage in VGG format
        :param inputdata:
        :param out_dims:
        :return:
        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims, kernel_size=3, stride=1, use_bias=False, name=name)
        relu = self.relu(inputdata=conv)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def __feature_sequence_extraction(self, inputdata):
        """
        Implement the 2.1 Part Feature Sequence Extraction
        :param inputdata: eg. batch*32*100*3 NHWC format
        :return:
        """
        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64, name='conv1')  # batch*16*50*64
        conv2 = self.__conv_stage(inputdata=conv1, out_dims=128, name='conv2')  # batch*8*25*128
        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3')  # batch*8*25*256
        relu3 = self.relu(conv3) # batch*8*25*256
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4')  # batch*8*25*256
        relu4 = self.relu(conv4)  # batch*8*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID')  # batch*4*25*256
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5')  # batch*4*25*512
        relu5 = self.relu(conv5)  # batch*4*25*512
        if self.phase.lower() == 'train':
            bn5 = self.layerbn(inputdata=relu5, is_training=True)
        else:
            bn5 = self.layerbn(inputdata=relu5, is_training=False)  # batch*4*25*512
        conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6')  # batch*4*25*512
        relu6 = self.relu(conv6)  # batch*4*25*512
        if self.phase.lower() == 'train':
            bn6 = self.layerbn(inputdata=relu6, is_training=True)
        else:
            bn6 = self.layerbn(inputdata=relu6, is_training=False)  # batch*4*25*512
        max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1], stride=[2, 1])  # batch*2*25*512
        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7')  # batch*1*25*512
        relu7 = self.relu(conv7)  # batch*1*25*512
        return relu7
    def __mobilenetV2feature_sequence_extraction(self, inputdata,reuseflag):
        """
        use inception V3 model
        :param inputdata: eg. batch*128*128*1
        :return:
        """
        # arg_scope = inception_v3_arg_scope()
        # with slim.arg_scope(arg_scope) as scope:
        #     shape=inputdata.get_shape().as_list()
        #     # if self.phase.lower() == 'train':
        #     #     logits, end_points = inception_v3(inputdata, is_training=True,reuse=reuseflag,num_classes=1001)
        #     # else:
        #     #     logits, end_points = inception_v3(inputdata, is_training=False, reuse=reuseflag, num_classes=1001)
        #     logits, end_points = inception_v3(inputdata, is_training=False, reuse=reuseflag, num_classes=1001)
        #     # fc_out=slim.fully_connected(logits,512,scope='fc512')
        #     return end_points['PreLogits']   # batch ,1,1,2048

        # with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
        #     logits, endpoints = mobilenet_v2.mobilenet(inputdata, depth_multiplier=1.0,reuse=reuseflag)
        #     return endpoints['global_pool']  # batch ,1,1,1280

        if self.__use_bn:
            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
                logits, endpoints = mobilenet_v2.mobilenet(inputdata, depth_multiplier=1.0, reuse=reuseflag)
                return endpoints['global_pool']  # batch ,1,1,1280
        else:
            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                logits, endpoints = mobilenet_v2.mobilenet(inputdata, depth_multiplier=1.0, reuse=reuseflag)
                return endpoints['global_pool']  # batch ,1,1,1280

    def __downfc(self, input_data,reuseflag):
        """
        Implement the map to sequence part of the network mainly used to convert the cnn feature map to sequence used in
        later stacked lstm layers
        :param inputdata:
        :return:
        """
        with tf.variable_scope('Downfc',reuse=reuseflag):
            if self.phase.lower() == 'train':
                input_data = self.dropout(inputdata=input_data, keep_prob=0.8)
            conv1 = tf.layers.conv2d(inputs=input_data, filters=640, kernel_size=[1, 1],padding="same", activation=tf.nn.relu)
            shape = conv1.get_shape().as_list()[1:]
            if None not in shape:
                conv1 = tf.reshape(conv1, [-1, int(np.prod(shape))])
            else:
                conv1 = tf.reshape(conv1, tf.stack([tf.shape(conv1)[0], -1]))
            fc_out1 = tf.layers.dense(conv1, 320,activation=tf.nn.relu)
            if self.phase.lower() == 'train':
                fc_out1 = tf.layers.dropout(inputs=fc_out1, rate=0.2, training=True)
            fc_out = tf.layers.dense(fc_out1, self.__hidden_nums,activation=tf.nn.relu)
            #  原来只有一层全连接
            # fc_out=self.fullyconnect(inputdata=input_data,out_dim=self.__hidden_nums)
            outdata=tf.expand_dims(fc_out, 1)
            return outdata
        # return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata):
        """
        Implement the sequence label part of the network
        :param inputdata:
        :return:
        """
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # # forward lstm cell
            # fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums, self.__hidden_nums]]
            # # Backward direction cells
            # bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in [self.__hidden_nums, self.__hidden_nums]]
            list_f = list(self.__hidden_nums*np.ones((self.__layers_nums)))
            list_b = list(self.__hidden_nums*np.ones((self.__layers_nums)))
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_f]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_b]

            stack_lstm_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, inputdata,dtype=tf.float32)

            if self.phase.lower() == 'train':
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer, keep_prob=0.8)

            # # 用处不大
            raw_pred = tf.layers.dense(stack_lstm_layer[:, -1, :], self.__num_classes*2,activation=tf.nn.relu)  # output based on the last output step
            # logits = tf.nn.softmax(raw_pred)
            # # 用处不大 作为中间变量
            #raw_pred = tf.layers.dense(stack_lstm_layer[:, -1, :],100)  # output based on the last output step
        return raw_pred  # stack_lstm_layer[:, -1, :]


    def __lr_label(self, candidata,browsedseq,dnnout):   # batch ,50   ;  batch, 100 ; batch , 50
        """
        Implement the sequence label part of the network
        :param inputdata:
        :return:
        """
        with tf.variable_scope('CTRLayers'):
            all_f=tf.concat([candidata,browsedseq,dnnout],axis=1)
            #  增加 fc 层
            if True:
                fc_pred=tf.layers.dense(all_f, 256,activation=tf.nn.relu)
                raw_pred = tf.layers.dense(fc_pred, self.__num_classes)
            else:
                raw_pred = tf.layers.dense(all_f, self.__num_classes)  # output based on the last output step 默认没有激活函数
            #all_f=tf.concat([candidata,browsedseq,dnnout],axis=1)
            #raw_pred = tf.layers.dense(all_f, self.__num_classes)  # output based on the last output step 默认没有激活函数
            logits = tf.nn.sigmoid(raw_pred)

        return logits , raw_pred


    def __dnnseq(self, inputdata):
        """
        Implement the sequence label part of the network
        :param inputdata:    batch *1 * 512
        :return:
        """
        with tf.variable_scope('DNNLayers'):
            shape = inputdata.get_shape().as_list()   # batch , per user browse num, lstm hidden num
            inputdata_new=tf.reshape(inputdata,shape=[-1,1,shape[1]*shape[2]]) #

            if self.phase.lower() == 'train':
                inputdata_new = self.dropout(inputdata=inputdata_new, keep_prob=0.8)
            dnn1 = tf.layers.dense(inputdata_new, 512,activation=tf.nn.relu)  # output based on the last output step
            if self.phase.lower() == 'train':
                dnn1 = self.dropout(inputdata=dnn1, keep_prob=0.8)
            dnn2 = tf.layers.dense(dnn1, 512,activation=tf.nn.relu)  # output based on the last output step
            if self.phase.lower() == 'train':
                dnn2 = self.dropout(inputdata=dnn2, keep_prob=0.8)
            dnn3 = tf.layers.dense(dnn2, 128,activation=tf.nn.relu)  # output based on the last output step
            dnn3 = tf.squeeze(dnn3, [1])
        return dnn3

    def build_telenet(self, candidata,browsedseq):
        """

        :param inputdata:
        :return:
        """
        #1、 candidate image
        candidata_f = self.__mobilenetV2feature_sequence_extraction(inputdata=candidata,reuseflag=False)
        c_f = self.__downfc(candidata_f,False)  # batch ,1,50
        c_f = tf.squeeze(c_f,[1])
        if True:
            #2、 user browsedseq image
            # first apply the cnn feature extraction stage
            shape = browsedseq.get_shape().as_list()  # batch, per user browsed num ,h,w,c
            Timestep=self.__seq_length
            browsed_f = self.__mobilenetV2feature_sequence_extraction(inputdata=browsedseq[:, 0, :, :, :], reuseflag=True)
            # second apply the map to sequence stage
            sequence = self.__downfc(browsed_f, True)
            for i in range(shape[1]-1):
                browsed_f = self.__mobilenetV2feature_sequence_extraction(inputdata=browsedseq[:,i+1,:,:,:],reuseflag=True)
                # second apply the map to sequence stage
                sequence = tf.concat([self.__downfc(browsed_f,True),sequence],axis=1)
            # third apply the sequence label stage
            blstm_out = self.__sequence_label(inputdata=sequence)
            #3、 dnn out
            dnn_out=self.__dnnseq(sequence)
            crt_out,logits_before=self.__lr_label(c_f,blstm_out,dnn_out)
            return crt_out, logits_before
        else:   #  浏览记录test
            raw_pred = tf.layers.dense(c_f, self.__num_classes)  # output based on the last output step
            logits = tf.nn.sigmoid(raw_pred,name='test_sigmoid')
            return logits,raw_pred
