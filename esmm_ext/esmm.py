import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers
from .din_logit_fn import din_logit_fn
from .dcn_logit_fn import dcn_logit_fn
from .dupn_logit_fn import dupn_logit_fn

def _base_logit_fn(features, mode, params):
  net = tf.feature_column.input_layer(features, params['feature_columns'])
  # Build the hidden layers, sized according to the 'hidden_units' param.
  for units in params['hidden_units']:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
      net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
  # Compute logits
  logits = tf.layers.dense(net, 1, activation=None)
  return logits


class ESMM(tf.estimator.Estimator):
  """An estimator for Tensorflow ESMM model"""
  def __init__(self,
    params,
    model_dir=None,
    optimizer='Adagrad',
    config=None,
    warm_start_from=None,
  ):
    if not optimizer: optimizer = 'Adagrad'
    self.optimizer = optimizers.get_optimizer_instance(optimizer, params["learning_rate"])
    self.logit_fn_dict = {"base": _base_logit_fn, "din": din_logit_fn, "dcn": dcn_logit_fn, "dupn": dupn_logit_fn}

    def _model_fn(features, labels, mode, params):
      logit_fn = self.logit_fn_dict[params["sub_model"]]
      with tf.variable_scope('ctr_model'):
        ctr_logits = logit_fn(features, mode, params)
      with tf.variable_scope('cvr_model'):
        cvr_logits = logit_fn(features, mode, params)

      ctr = tf.sigmoid(ctr_logits, name="CTR")
      cvr = tf.sigmoid(cvr_logits, name="CVR")
      ctcvr = tf.multiply(ctr, cvr, name="CTCVR")
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
          'ctcvr': ctcvr,
          'ctr': ctr,
          'cvr': cvr
        }
        export_outputs = {
          'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

      y = labels['cvr']
      cvr_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(y, ctcvr), name="cvr_loss")
      ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['ctr'], logits=ctr_logits),
                               name="ctr_loss")
      loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

      ctr_accuracy = tf.metrics.accuracy(labels=labels['ctr'],
                                         predictions=tf.to_float(tf.greater_equal(ctr, 0.5)))
      cvr_accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(ctcvr, 0.5)))
      ctr_auc = tf.metrics.auc(labels['ctr'], ctr)
      cvr_auc = tf.metrics.auc(y, ctcvr)
      metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy, 'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
      tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
      tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
      tf.summary.scalar('ctr_auc', ctr_auc[1])
      tf.summary.scalar('cvr_auc', cvr_auc[1])
      if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

      # Create training op.
      assert mode == tf.estimator.ModeKeys.TRAIN
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = self.optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    super(ESMM, self).__init__(
      model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)
