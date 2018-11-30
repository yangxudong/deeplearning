import tensorflow as tf


def build_deep_layers(x0, mode, params):
  # Build the hidden layers, sized according to the 'hidden_units' param.
  use_batch_norm = params['use_batch_norm'] if 'use_batch_norm' in params else False
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  net = x0
  for units in params['hidden_units']:
    if use_batch_norm:
      x = tf.layers.dense(net, units=units, activation=None, use_bias=False)
      net = tf.nn.relu(tf.layers.batch_normalization(x, training=is_training))
    else:
      net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
  return net


def build_cross_layers(x0, params):
  num_layers = params['num_cross_layers']
  x = x0
  for i in range(num_layers):
    x = cross_layer(x0, x, 'cross_{}'.format(i))
  return x


def cross_layer(x0, x, name):
  with tf.variable_scope(name):
    input_dim = x0.get_shape().as_list()[1]
    w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
    return x0 * xb + b + x


def dcn_logit_fn(features, mode, params):
  x0 = tf.feature_column.input_layer(features, params['feature_columns'])
  last_deep_layer = build_deep_layers(x0, mode, params)
  last_cross_layer = build_cross_layers(x0, params)
  last_layer = tf.concat([last_cross_layer, last_deep_layer], 1)
  logits = tf.layers.dense(last_layer, 1)
  return logits
