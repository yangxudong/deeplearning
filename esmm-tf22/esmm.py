# -*- coding: utf-8 -*-
# @Time    : 2020-10-28 10:11
# @Author  : WenYi
# @Contact : 1244058349@qq.com
# @Description :  ESMM model for CTR and CVR predict task


import tensorflow as tf
# 是否使用GPU
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class CTCVRNet:
	def __init__(self, cate_feautre_dict):
		self.embed = dict()
		for k, v in cate_feautre_dict.items():
			self.embed[k] = layers.Embedding(v, 64)
	
	def build_ctr_model(self, ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
	                    ctr_item_cate_input, ctr_user_cate_feature_dict, ctr_item_cate_feature_dict):
		user_embeddings, item_embeddings = [], []
		for k, v in ctr_user_cate_feature_dict.items():
			embed = self.embed[k](tf.reshape(ctr_user_cate_input[:, v[0]], [-1, 1]))
			embed = layers.Reshape((64,))(embed)
			user_embeddings.append(embed)
		
		for k, v in ctr_item_cate_feature_dict.items():
			embed = self.embed[k](tf.reshape(ctr_item_cate_input[:, v[0]], [-1, 1]))
			embed = layers.Reshape((64,))(embed)
			item_embeddings.append(embed)
		user_feature = layers.concatenate([ctr_user_numerical_input] + user_embeddings, axis=-1)
		item_feature = layers.concatenate([ctr_item_numerical_input] + item_embeddings, axis=-1)
		
		user_feature = layers.Dropout(0.5)(user_feature)
		user_feature = layers.BatchNormalization()(user_feature)
		user_feature = layers.Dense(128, activation='relu')(user_feature)
		user_feature = layers.Dense(64, activation='relu')(user_feature)
		
		item_feature = layers.Dropout(0.5)(item_feature)
		item_feature = layers.BatchNormalization()(item_feature)
		item_feature = layers.Dense(128, activation='relu')(item_feature)
		item_feature = layers.Dense(64, activation='relu')(item_feature)
		
		dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
		dense_feature = layers.Dropout(0.5)(dense_feature)
		dense_feature = layers.BatchNormalization()(dense_feature)
		dense_feature = layers.Dense(64, activation='relu')(dense_feature)
		pred = layers.Dense(1, activation='sigmoid', name='ctr_output')(dense_feature)
		return pred
	
	def build_cvr_model(self, cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
	                    cvr_item_cate_input, cvr_user_cate_feature_dict, cvr_item_cate_feature_dict):
		user_embeddings, item_embeddings = [], []
		for k, v in cvr_user_cate_feature_dict.items():
			embed = self.embed[k](tf.reshape(cvr_user_cate_input[:, v[0]], [-1, 1]))
			embed = layers.Reshape((64,))(embed)
			user_embeddings.append(embed)
		
		for k, v in cvr_item_cate_feature_dict.items():
			embed = self.embed[k](tf.reshape(cvr_item_cate_input[:, v[0]], [-1, 1]))
			embed = layers.Reshape((64,))(embed)
			item_embeddings.append(embed)
		user_feature = layers.concatenate([cvr_user_numerical_input] + user_embeddings, axis=-1)
		item_feature = layers.concatenate([cvr_item_numerical_input] + item_embeddings, axis=-1)
		
		user_feature = layers.Dropout(0.5)(user_feature)
		user_feature = layers.BatchNormalization()(user_feature)
		user_feature = layers.Dense(128, activation='relu')(user_feature)
		user_feature = layers.Dense(64, activation='relu')(user_feature)
		
		item_feature = layers.Dropout(0.5)(item_feature)
		item_feature = layers.BatchNormalization()(item_feature)
		item_feature = layers.Dense(128, activation='relu')(item_feature)
		item_feature = layers.Dense(64, activation='relu')(item_feature)
		
		dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
		dense_feature = layers.Dropout(0.5)(dense_feature)
		dense_feature = layers.BatchNormalization()(dense_feature)
		dense_feature = layers.Dense(64, activation='relu')(dense_feature)
		pred = layers.Dense(1, activation='sigmoid', name='cvr_output')(dense_feature)
		return pred
	
	def build(self, user_cate_feature_dict, item_cate_feature_dict):
		# CTR model input
		ctr_user_numerical_input = layers.Input(shape=(5,))
		ctr_user_cate_input = layers.Input(shape=(5,))
		ctr_item_numerical_input = layers.Input(shape=(5,))
		ctr_item_cate_input = layers.Input(shape=(3,))
		
		# CVR model input
		cvr_user_numerical_input = layers.Input(shape=(5,))
		cvr_user_cate_input = layers.Input(shape=(5,))
		cvr_item_numerical_input = layers.Input(shape=(5,))
		cvr_item_cate_input = layers.Input(shape=(3,))
		
		ctr_pred = self.build_ctr_model(ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
		                                ctr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
		cvr_pred = self.build_cvr_model(cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
		                                cvr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
		ctcvr_pred = tf.multiply(ctr_pred, cvr_pred)
		model = Model(
			inputs=[ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
			        cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input],
			outputs=[ctr_pred, ctcvr_pred])
		
		return model
