#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/4 23:34
# @Author  : linhuan
# @File    : img2tfrecord.py
import tensorflow as tf
import os
import random
import math
import sys
import get_data_list
#
sys.path.append('../')
from global_configuration import config
#
_MAX_SEQ_NUM= config.cfg.UserBrowseNum
#数据块
_PER_NUM=20000  # 每个数据块记录的训练样本数
# 空图像
_Z_PIC='./0.jpg'
# 图片的路径
_IMG_PATH='/data4/ImageData/TelenetData/7yue'
#定义tfrecord 的路径和名称
def _get_dataset_filename(dataset_dir,split_name,shard_id,num_shards):
    output_filename = 'image_%s_%05d-of-%05d.tfrecords' % (split_name,shard_id,num_shards)
    return os.path.join(dataset_dir,output_filename)
def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
def bytes_seq_feature(list_v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_v))
#图片转换城tfexample函数
def image_to_tfexample(image_data,list_v,image_format,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/seq_list':bytes_seq_feature(list_v),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id)
    }))
#数据转换城tfrecorad格式
def _convert_dataset(split_name,candidate_img, seq_imgs, labels,dataset_dir):
    assert split_name in ['train','test']
    #计算数据块个数
    #_PER_NUM 每个数据块样本数个数
    num_shards=int(math.ceil(len(candidate_img)*1.0 / _PER_NUM))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(num_shards):
            #定义tfrecord的路径名字
                output_filename = _get_dataset_filename(dataset_dir,split_name,shard_id,num_shards)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    #每个数据块开始的位置
                    start_ndx = shard_id * _PER_NUM
                    #每个数据块结束的位置
                    end_ndx = min((shard_id+1) * _PER_NUM,len(candidate_img))
                    for i in range(start_ndx,end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d '% (i+1,len(candidate_img),shard_id))
                            sys.stdout.flush()
                            # 设置时间过滤
                           # if int(candidate_img[i][3])<=1526745798   :   # 6月14 号为分界线 每天训练
                           #     continue 
                            #读取图片
                            image_data = tf.gfile.FastGFile(os.path.join(_IMG_PATH,candidate_img[i][1])+'.jpg','rb').read()
                            #获取图片的id
                            class_id = int(labels[i])
                            #浏览记录
                            list_v=[]
                            s_i = seq_imgs[i]
                            if len(s_i)>=_MAX_SEQ_NUM:
                                start_index_j = len(s_i) - _MAX_SEQ_NUM
                                for index in range(start_index_j,len(s_i)):
                                    image_t = tf.gfile.FastGFile(os.path.join(_IMG_PATH, s_i[index][1]) + '.jpg', 'rb').read()
                                    list_v.append(image_t)
                            else:
                                continue
                                start_index_j = _MAX_SEQ_NUM -len(s_i)
                                for index in range(start_index_j):
                                    image_t = tf.gfile.FastGFile(_Z_PIC,'rb').read()
                                    list_v.append(image_t)
                                for index in range(start_index_j,_MAX_SEQ_NUM):
                                    image_t = tf.gfile.FastGFile(os.path.join(_IMG_PATH, s_i[index-start_index_j][1]) + '.jpg', 'rb').read()
                                    list_v.append(image_t)
                           # for index in range(min(len(s_i),_MAX_SEQ_NUM)):#最长的浏览记录
                           #     image_t = tf.gfile.FastGFile(os.path.join(_IMG_PATH, s_i[index][1]) + '.jpg', 'rb').read()
                           #     #=========
                           #     #img_data_jpg =tf.image.decode_jpeg(image_t) #tf.image.decode_image
                           #     #image = tf.reshape(img_data_jpg, [128, 128, 3])
                           #     #==================
                           #     list_v.append(image_t)
                           # if len(list_v)<_MAX_SEQ_NUM:
                           #     for i_add in range(_MAX_SEQ_NUM-len(list_v)):
                           #         image_t = tf.gfile.FastGFile(_Z_PIC,'rb').read()
                           #         list_v.append(image_t)
                            #生成tfrecord文件
                            example = image_to_tfexample(image_data,list_v,b'jpg',class_id)
                            #写入数据
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError  as e:
                            print ('could not read:',candidate_img[i][1])
                            print ('error:' , e)
                            print ('skip it \n')
    sys.stdout.write('\n')
    sys.stdout.flush()

def filter_seq_imgs(candidate_img, seq_imgs, labels):
    all_num=len(seq_imgs)
    candidate_img_new=[]
    seq_imgs_new=[]
    labels_new=[]
    for i in range(all_num):
        # if len(seq_imgs[i])>=config.cfg.UserBrowseNum:
        if len(seq_imgs[i]) >= 6:
            candidate_img_new.append(candidate_img[i])
            seq_imgs_new.append(seq_imgs[i])
            labels_new.append(labels[i])

    return candidate_img_new,seq_imgs_new,labels_new
if __name__ == '__main__':
   # 修改 图片库的位置
    DATASET_DIR='/data4/ImageData/TelenetData/5'
    if not os.path.exists( DATASET_DIR):
        os.makedirs( DATASET_DIR)
    pickle_dir='/home/huan.lin/py3t/Telenet/data_pre/newcsv/5.pickle'
    # get pre_data
    data_generator = get_data_list.GetDataList(cart_data_path='',user_action_path='',sava_pickle=pickle_dir)
    candidate_img, seq_imgs, labels=data_generator.read_data()

    if True: # 是否过滤
        candidate_img, seq_imgs, labels=filter_seq_imgs(candidate_img, seq_imgs, labels)
    # 测试 tfrecord=====================
   # candidate_img=candidate_img[:3000]
   # seq_imgs=seq_imgs[:3000]
   # labels=labels[:3000]
    #===============================
    tolal_num=len(candidate_img)
    #print image_filename_list
    train_percent=config.cfg.TRAIN.PERCENT
    train_image_num = int(tolal_num * train_percent)
    # training database
    candidate_img_train = candidate_img[: train_image_num]
    seq_imgs_train = seq_imgs[: train_image_num]
    labels_train = labels[: train_image_num]
    # validataion database
    candidate_img_val = candidate_img[train_image_num :]
    seq_imgs_val = seq_imgs[train_image_num :]
    labels_val = labels[train_image_num :]
    print('total num: ',len(candidate_img),'train num:  ',len(candidate_img_train),'val num:   ',len(candidate_img_val))
    # 数据转换
    _convert_dataset('train', candidate_img_train, seq_imgs_train, labels_train,DATASET_DIR)
    _convert_dataset('test', candidate_img_val, seq_imgs_val,labels_val, DATASET_DIR)
   # _convert_dataset('train', candidate_img, seq_imgs, labels,DATASET_DIR) # 只有train tfrecord
