import tensorflow as tf


def build_deep_layers(x0, params):
  # Build the hidden layers, sized according to the 'hidden_units' param.
  net = x0
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
  return net


def build_cross_layers(x0, params):
  num_layers = params['num_cross_layers']
  x = x0
  for i in range(num_layers):
    x = cross_layer2(x0, x, 'cross_{}'.format(i))
  return x


def cross_layer(x0, x, name):
  with tf.variable_scope(name):
    input_dim = x0.get_shape().as_list()[1]
    w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    xx0 = tf.expand_dims(x0, -1)  # shape <?, d, 1>
    mat = tf.matmul(xx0, tf.reshape(x, [-1, 1, input_dim]))  # shape <?, d, d>
    return tf.tensordot(mat, w, 1) + b + x  # shape <?, d>

def cross_layer2(x0, x, name):
  with tf.variable_scope(name):
    input_dim = x0.get_shape().as_list()[1]
    w = tf.get_variable("weight", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable("bias", [input_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    xb = tf.tensordot(tf.reshape(x, [-1, 1, input_dim]), w, 1)
    return x0 * xb + b + x

def dcn_model_fn(features, labels, mode, params):
  x0 = tf.feature_column.input_layer(features, params['feature_columns'])
  last_deep_layer = build_deep_layers(x0, params)
  last_cross_layer = build_cross_layers(x0, params)
  last_layer = tf.concat([last_cross_layer, last_deep_layer], 1)
  my_head = tf.contrib.estimator.binary_classification_head(thresholds=[0.5])
  logits = tf.layers.dense(last_layer, units=my_head.logits_dimension)
  optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
  return my_head.create_estimator_spec(
    features=features,
    mode=mode,
    labels=labels,
    logits=logits,
    train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
  )

