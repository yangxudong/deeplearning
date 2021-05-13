# -*- coding: utf-8 -*-
# @Time    : 2020-11-05 17:49
# @Author  : WenYi
# @Contact : wenyi@cvte.com
# @Description :  script description

import numpy as np
import pandas as pd
from esmm import CTCVRNet
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model_train import train_model

## 疑问:输入数据，是否应该为ctr data 和 ctcvr data 呢？

# data include ctr data and cvr data, ctr data include ctr user data and ctr item data,
# user data include numerical data and categorical data
# item data include numerical data and categorical data
# we generate sample data include user feature data and item feature data
# user feature data include 5 numerical data and 5 categorical data
# item feature data include 5 numerical data and 5 categorical data
ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                           columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),
                                                columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                           columns=['item_cate_{}'.format(i) for i in range(3)])


ctr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
ctr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
ctr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
ctr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)), columns=['item_cate_{}'.format(i) for i in range(3)])

cvr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['user_numerical_{}'.format(i) for i in range(5)])
cvr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),
                                         columns=['user_cate_{}'.format(i) for i in range(5)])
cvr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),
                                              columns=['item_numerical_{}'.format(i) for i in range(5)])
cvr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),
                                         columns=['item_cate_{}'.format(i) for i in range(3)])

ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

train_data = [ctr_user_numerical_feature_train, ctr_user_cate_feature_train,
              ctr_item_numerical_feature_train,ctr_item_cate_feature_train,
              cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
              cvr_item_numerical_feature_train, cvr_item_cate_feature_train,
              ctr_target_train, cvr_target_train]
val_data = [ctr_user_numerical_feature_val, ctr_user_cate_feature_val,
            ctr_item_numerical_feature_val,ctr_item_cate_feature_val,
            cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
            cvr_item_numerical_feature_val, cvr_item_cate_feature_val,
            ctr_target_val, cvr_target_val]
pred_data = [ctr_user_numerical_feature_train.iloc[0:20], ctr_user_cate_feature_train.iloc[0:20],
              ctr_item_numerical_feature_train.iloc[0:20],ctr_item_cate_feature_train.iloc[0:20],
              cvr_user_numerical_feature_train.iloc[0:20], cvr_user_cate_feature_train.iloc[0:20],
              cvr_item_numerical_feature_train.iloc[0:20], cvr_item_cate_feature_train.iloc[0:20]]

cate_feature_dict = {}
user_cate_feature_dict = {}
item_cate_feature_dict = {}
for idx, col in enumerate(ctr_user_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
    user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
for idx, col in enumerate(ctr_item_cate_feature_train.columns):
    cate_feature_dict[col] = ctr_item_cate_feature_train[col].max() + 1
    item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)

# ctcvr = CTCVRNet(cate_feature_dict)
# ctcvr_model = ctcvr.build(user_cate_feature_dict, item_cate_feature_dict)
# opt = optimizers.Adam(lr=0.003, decay=0.0001)
# ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
#                     metrics=[tf.keras.metrics.AUC()])

# keras model save path
filepath = "esmm_best.h5"

# call back function
# checkpoint = ModelCheckpoint(
#     filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
# earlystopping = EarlyStopping(
#     monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
# callbacks = [checkpoint, reduce_lr, earlystopping]

# trian model
ctcvr_model = train_model(cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict, train_data, val_data, filepath)

# load model
# ctcvr_model = tf.keras.models.load_model('esmm_best.h5')

# model predict
[ctr_pred, ctcvr_pred] = ctcvr_model.predict(pred_data)

# get cvr predict
cvr_pred = ctcvr_pred/ctr_pred


# 参考：https://github.com/busesese/ESMM
# 问题：实现这个模型的时候怎么训练，损失函数怎么写，数据怎么构造？
# 这里我们可以看到主任务是CVR任务，副任务是CTR任务，实际生产的数据是用户曝光数据，点击数据和转化数据，
# 那么曝光和点击数据可以构造副任务的CTR模型（正样本：曝光&点击、负样本：曝光&未点击），
# 曝光和转化数据(转化必点击)构造的是CTCVR任务（正样本：点击&转化、负样本：点击&未转化），
# 模型的输出有3个，CTR模型输出预测的pCTR,CVR模型输出预测的pCVR,联合模型输出预测的pCTCVR=pCTR*pCVR，
# 由于CVR模型的输出标签不好直接构造，因此这里损失函数loss = ctr的损失函数 + ctcvr的损失函数，
# 因为pctcvr=pctr*pcvr，所以loss中也充分利用到CVR模型的参数。

# 综上,
# 模型的Input分为2个数据集：1）CTR任务数据、2）CTCVR任务数据
# 模型的Output也是2个预测结果：1）pCTR、2）pCTCVR
# 而pCVR = pCTCVR / pCTR



