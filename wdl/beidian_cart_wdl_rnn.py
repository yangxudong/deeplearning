# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow estimators for Linear and DNN joined training models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import six

from tensorflow.python.estimator import estimator
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.estimator import model_fn
from tensorflow.python.ops import array_ops
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import tf_export

import os
import argparse
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
task_index = int(os.environ["TF_INDEX"])
task_type = os.environ["TF_ROLE"]

tf_config = dict()
worker_num = len(cluster["worker"])
if task_type == "ps":
    tf_config["task"] = {"index":task_index, "type":task_type}
else:
    if task_index == 0:
        tf_config["task"] = {"index":0, "type":"chief"}
    else:
        tf_config["task"] = {"index":task_index-1, "type":task_type}

if worker_num == 1:
    cluster["chief"] = cluster["worker"]
    del cluster["worker"]
else:
    cluster["chief"] = [cluster["worker"][0]]
    del cluster["worker"][0]

tf_config["cluster"] = cluster
os.environ["TF_CONFIG"] = json.dumps(tf_config)
print(json.loads(os.environ["TF_CONFIG"]))
# The default learning rates are a historical artifact of the initial
# implementation.
_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


def _check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear '
            'model.')


def _linear_learning_rate(num_linear_feature_columns):
    """Returns the default learning rate of the linear model.

    The calculation is a historical artifact of this initial implementation, but
    has proven a reasonable choice.

    Args:
      num_linear_feature_columns: The number of feature columns of the linear
        model.

    Returns:
      A float.
    """
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(_LINEAR_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _add_hidden_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _dnn_logit_fn_builder(units, hidden_units, feature_columns, rnn_feature_columns, activation_fn,
                          dropout, input_layer_partitioner):
    """Function builder for a dnn logit_fn.

    Args:
      units: An int indicating the dimension of the logit layer.  In the
        MultiHead case, this should be the sum of all component Heads' logit
        dimensions.
      hidden_units: Iterable of integer number of hidden units per layer.
      feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
      activation_fn: Activation function applied to each layer.
      dropout: When not `None`, the probability we will drop out a given
        coordinate.
      input_layer_partitioner: Partitioner for input layer.

    Returns:
      A logit_fn (see below).

    Raises:
      ValueError: If units is not an int.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def dnn_logit_fn(features, mode):
        """Deep Neural Network logit_fn.

        Args:
          features: This is the first item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `Tensor` or `dict` of same.
          mode: Optional. Specifies if this training, evaluation or prediction. See
                `ModeKeys`.

        Returns:
          A `Tensor` representing the logits, or a list of `Tensor`'s representing
          multiple logits in the MultiHead case.
        """
        with variable_scope.variable_scope(
                'input_from_feature_columns',
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner):
            net = feature_column_lib.input_layer(
                features=features, feature_columns=feature_columns)
            if rnn_feature_columns != None:
                rnn_features_embedding = feature_column_lib.input_layer(features=features, feature_columns=rnn_feature_columns)
                rnn_features_embedding = tf.reshape(rnn_features_embedding, [-1, FLAGS.rnn_length, FLAGS.rnn_input_size])
                cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.rnn_hidden_size)
                att_wrapper = tf.contrib.rnn.AttentionCellWrapper(cell=cell, attn_length=10)
		outputs, _ = tf.nn.dynamic_rnn(att_wrapper, rnn_features_embedding, dtype=tf.float32)
                outputs = tf.reshape(outputs, [-1, FLAGS.rnn_length * FLAGS.rnn_hidden_size])
                net = array_ops.concat([net, outputs], 1)

        for layer_id, num_hidden_units in enumerate(hidden_units):
            with variable_scope.variable_scope(
                            'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
                net = core_layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=init_ops.glorot_uniform_initializer(),
                    name=hidden_layer_scope)
                if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
                    net = core_layers.dropout(net, rate=dropout, training=True)
            _add_hidden_layer_summary(net, hidden_layer_scope.name)

        with variable_scope.variable_scope('logits', values=(net,)) as logits_scope:
            logits = core_layers.dense(
                net,
                units=units,
                activation=None,
                kernel_initializer=init_ops.glorot_uniform_initializer(),
                name=logits_scope)
        _add_hidden_layer_summary(logits, logits_scope.name)

        return logits

    return dnn_logit_fn


def _dnn_linear_combined_model_fn(features,
                                  labels,
                                  mode,
                                  head,
                                  linear_feature_columns=None,
                                  linear_optimizer='Ftrl',
                                  dnn_feature_columns=None,
                                  rnn_feature_columns=None,
                                  dnn_optimizer='Adagrad',
                                  dnn_hidden_units=None,
                                  dnn_activation_fn=nn.relu,
                                  dnn_dropout=None,
                                  input_layer_partitioner=None,
                                  config=None):
    """Deep Neural Net and Linear combined model_fn.

    Args:
      features: dict of `Tensor`.
      labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
        `int32` or `int64` in the range `[0, n_classes)`.
      mode: Defines whether this is training, evaluation or prediction.
        See `ModeKeys`.
      head: A `Head` instance.
      linear_feature_columns: An iterable containing all the feature columns used
        by the Linear model.
      linear_optimizer: string, `Optimizer` object, or callable that defines the
        optimizer to use for training the Linear model. Defaults to the Ftrl
        optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used by
        the DNN model.
      dnn_optimizer: string, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN model. Defaults to the Adagrad
        optimizer.
      dnn_hidden_units: List of hidden units per DNN layer.
      dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
        will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability we will drop out a given DNN
        coordinate.
      input_layer_partitioner: Partitioner for input layer.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      An `EstimatorSpec` instance.

    Raises:
      ValueError: If both `linear_feature_columns` and `dnn_features_columns`
        are empty at the same time, or `input_layer_partitioner` is missing,
        or features has the wrong type.
    """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    if not linear_feature_columns and not dnn_feature_columns:
        raise ValueError(
            'Either linear_feature_columns or dnn_feature_columns must be defined.')

    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    # Build DNN Logits.
    dnn_parent_scope = 'dnn'

    if not dnn_feature_columns:
        dnn_logits = None
    else:
        dnn_optimizer = optimizers.get_optimizer_instance(
            dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
        _check_no_sync_replicas_optimizer(dnn_optimizer)
        if not dnn_hidden_units:
            raise ValueError(
                'dnn_hidden_units must be defined when dnn_feature_columns is '
                'specified.')
        dnn_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas))
        with variable_scope.variable_scope(
                dnn_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=dnn_partitioner):

            dnn_logit_fn = _dnn_logit_fn_builder(  # pylint: disable=protected-access
                units=head.logits_dimension,
                hidden_units=dnn_hidden_units,
                feature_columns=dnn_feature_columns,
                rnn_feature_columns=rnn_feature_columns,
                activation_fn=dnn_activation_fn,
                dropout=dnn_dropout,
                input_layer_partitioner=input_layer_partitioner)
            dnn_logits = dnn_logit_fn(features=features, mode=mode)

    linear_parent_scope = 'linear'

    if not linear_feature_columns:
        linear_logits = None
    else:
        linear_optimizer = optimizers.get_optimizer_instance(
            linear_optimizer,
            learning_rate=_linear_learning_rate(len(linear_feature_columns)))
        _check_no_sync_replicas_optimizer(linear_optimizer)
        with variable_scope.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            logit_fn = linear._linear_logit_fn_builder(  # pylint: disable=protected-access
                units=head.logits_dimension,
                feature_columns=linear_feature_columns)
            linear_logits = logit_fn(features=features)
            _add_layer_summary(linear_logits, scope.name)

    # Combine logits and build full model.
    if dnn_logits is not None and linear_logits is not None:
        logits = dnn_logits + linear_logits
    elif dnn_logits is not None:
        logits = dnn_logits
    else:
        logits = linear_logits

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        global_step = training_util.get_global_step()
        if dnn_logits is not None:
            train_ops.append(
                dnn_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=dnn_parent_scope)))
        if linear_logits is not None:
            train_ops.append(
                linear_optimizer.minimize(
                    loss,
                    var_list=ops.get_collection(
                        ops.GraphKeys.TRAINABLE_VARIABLES,
                        scope=linear_parent_scope)))

        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


@tf_export('estimator.DNNLinearCombinedWithRNNClassifier')
class DNNLinearCombinedWithRNNClassifier(estimator.Estimator):
    """An estimator for TensorFlow Linear and DNN joined classification models.

    Note: This estimator is also known as wide-n-deep.

    Example:

    ```python
    numeric_feature = numeric_column(...)
    categorical_column_a = categorical_column_with_hash_bucket(...)
    categorical_column_b = categorical_column_with_hash_bucket(...)

    categorical_feature_a_x_categorical_feature_b = crossed_column(...)
    categorical_feature_a_emb = embedding_column(
        categorical_column=categorical_feature_a, ...)
    categorical_feature_b_emb = embedding_column(
        categorical_id_column=categorical_feature_b, ...)

    estimator = DNNLinearCombinedClassifier(
        # wide settings
        linear_feature_columns=[categorical_feature_a_x_categorical_feature_b],
        linear_optimizer=tf.train.FtrlOptimizer(...),
        # deep settings
        dnn_feature_columns=[
            categorical_feature_a_emb, categorical_feature_b_emb,
            numeric_feature],
        dnn_hidden_units=[1000, 500, 100],
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),
        # warm-start settings
        warm_start_from="/path/to/checkpoint/dir")

    # To apply L1 and L2 regularization, you can set optimizers as follows:
    tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001,
        l2_regularization_strength=0.001)
    # It is same for FtrlOptimizer.

    # Input builders
    def input_fn_train: # returns x, y
      pass
    estimator.train(input_fn=input_fn_train, steps=100)

    def input_fn_eval: # returns x, y
      pass
    metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
    def input_fn_predict: # returns x, None
      pass
    predictions = estimator.predict(input_fn=input_fn_predict)
    ```

    Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

    * for each `column` in `dnn_feature_columns` + `linear_feature_columns`:
      - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `_WeightedCategoricalColumn`, two features: the first
        with `key` the id column name, the second with `key` the weight column
        name. Both features' `value` must be a `SparseTensor`.
      - if `column` is a `_DenseColumn`, a feature with `key=column.name`
        whose `value` is a `Tensor`.

    Loss is calculated by using softmax cross entropy.

    @compatibility(eager)
    Estimators are not compatible with eager execution.
    @end_compatibility
    """

    def __init__(self,
                 model_dir=None,
                 linear_feature_columns=None,
                 linear_optimizer='Ftrl',
                 dnn_feature_columns=None,
                 rnn_feature_columns=None,
                 dnn_optimizer='Adagrad',
                 dnn_hidden_units=None,
                 dnn_activation_fn=nn.relu,
                 dnn_dropout=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None,
                 warm_start_from=None,
                 loss_reduction=losses.Reduction.SUM):
        """Initializes a DNNLinearCombinedClassifier instance.

        Args:
          model_dir: Directory to save model parameters, graph and etc. This can
            also be used to load checkpoints from the directory into a estimator
            to continue training a previously saved model.
          linear_feature_columns: An iterable containing all the feature columns
            used by linear part of the model. All items in the set must be
            instances of classes derived from `FeatureColumn`.
          linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
            the linear part of the model. Defaults to FTRL optimizer.
          dnn_feature_columns: An iterable containing all the feature columns used
            by deep part of the model. All items in the set must be instances of
            classes derived from `FeatureColumn`.
          dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
            the deep part of the model. Defaults to Adagrad optimizer.
          dnn_hidden_units: List of hidden units per layer. All layers are fully
            connected.
          dnn_activation_fn: Activation function applied to each layer. If None,
            will use `tf.nn.relu`.
          dnn_dropout: When not None, the probability we will drop out
            a given coordinate.
          n_classes: Number of label classes. Defaults to 2, namely binary
            classification. Must be > 1.
          weight_column: A string or a `_NumericColumn` created by
            `tf.feature_column.numeric_column` defining feature column representing
            weights. It is used to down weight or boost examples during training. It
            will be multiplied by the loss of the example. If it is a string, it is
            used as a key to fetch weight tensor from the `features`. If it is a
            `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
            then weight_column.normalizer_fn is applied on it to get weight tensor.
          label_vocabulary: A list of strings represents possible label values. If
            given, labels must be string type and have any value in
            `label_vocabulary`. If it is not given, that means labels are
            already encoded as integer or float within [0, 1] for `n_classes=2` and
            encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
            Also there will be errors if vocabulary is not provided and labels are
            string.
          input_layer_partitioner: Partitioner for input layer. Defaults to
            `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
          config: RunConfig object to configure the runtime settings.
          warm_start_from: A string filepath to a checkpoint to warm-start from, or
            a `WarmStartSettings` object to fully configure warm-starting.  If the
            string filepath is provided instead of a `WarmStartSettings`, then all
            weights are warm-started, and it is assumed that vocabularies and Tensor
            names are unchanged.
          loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
            to reduce training loss over batch. Defaults to `SUM`.

        Raises:
          ValueError: If both linear_feature_columns and dnn_features_columns are
            empty at the same time.
        """
        linear_feature_columns = linear_feature_columns or []
        dnn_feature_columns = dnn_feature_columns or []
        self._feature_columns = (
            list(linear_feature_columns) + list(dnn_feature_columns))
        if not self._feature_columns:
            raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                             'must be defined.')
        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)

        def _model_fn(features, labels, mode, config):
            """Call the _dnn_linear_combined_model_fn."""
            return _dnn_linear_combined_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                dnn_feature_columns=dnn_feature_columns,
                rnn_feature_columns=rnn_feature_columns,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=dnn_activation_fn,
                dnn_dropout=dnn_dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config)

        super(DNNLinearCombinedWithRNNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)




"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""

_CSV_COLUMNS = [
    'device_model'
    , 'brand'
    , 'resolution'
    , 'carrier'
    , 'access'
    , 'channel'
    , 'os'
    , 'province'
    , 'city'
    , 'ord_cnt_1d_type'
    , 'ord_cnt_7d_type'
    , 'ord_cnt_15d_type'
    , 'ord_cnt_30d_type'
    , 'ord_amt_1d_type'
    , 'ord_amt_7d_type'
    , 'ord_amt_15d_type'
    , 'ord_amt_30d_type'
    , 'ipv_1d_type'
    , 'ipv_7d_type'
    , 'ipv_15d_type'
    , 'ipv_30d_type'
    , 'product_id'
    , 'cid'
    , 'brand_id'
    , 'c1_id'
    , 'c2_id'
    , 'user_click_rate_1d'
    , 'user_fenxiang_rate_1d'
    , 'user_buy_rate_1d'
    , 'user_click_rate_1w'
    , 'user_fenxiang_rate_1w'
    , 'user_buy_rate_1w'
    , 'user_click_rate_2w'
    , 'user_fenxiang_rate_2w'
    , 'user_buy_rate_2w'
    , 'user_click_rate_1m'
    , 'user_fenxiang_rate_1m'
    , 'user_buy_rate_1m'
    , 'expose_norm_score_1d'
    , 'browse_norm_score_1d'
    , 'fenxiang_norm_score_1d'
    , 'buy_norm_score_1d'
    , 'expose_norm_score_1w'
    , 'browse_norm_score_1w'
    , 'fenxiang_norm_score_1w'
    , 'buy_norm_score_1w'
    , 'expose_norm_score_2w'
    , 'browse_norm_score_2w'
    , 'fenxiang_norm_score_2w'
    , 'buy_norm_score_2w'
    , 'expose_norm_score_1m'
    , 'browse_norm_score_1m'
    , 'fenxiang_norm_score_1m'
    , 'buy_norm_score_1m'
    , 'province_cid'
    , 'province_c2_id'
    , 'province_c1_id'
    , 'province_brand_id'
    , 'city_cid'
    , 'city_c2_id'
    , 'city_c1_id'
    , 'city_brand_id'
    , 'os_cid'
    , 'os_c2_id'
    , 'os_c1_id'
    , 'os_brand_id'
    , 'brand_cid'
    , 'brand_c2_id'
    , 'brand_c1_id'
    , 'brand_brand_id'
    , 'resolution_cid'
    , 'resolution_c2_id'
    , 'resolution_c1_id'
    , 'resolution_brand_id'
    , 'cid_prefer_1d'
    , 'cate1_prefer_1d'
    , 'cate2_prefer_1d'
    , 'brand_prefer_1d'
    , 'cid_prefer_1w'
    , 'cate1_prefer_1w'
    , 'cate2_prefer_1w'
    , 'brand_prefer_1w'
    , 'cid_prefer_2w'
    , 'cate1_prefer_2w'
    , 'cate2_prefer_2w'
    , 'brand_prefer_2w'
    , 'cid_prefer_1m'
    , 'cate1_prefer_1m'
    , 'cate2_prefer_1m'
    , 'brand_prefer_1m'
    , 'position'
    , 'match_type'
    , 'pids'
    , 'cids'
    , 'weights'
    , 'ctr_flag'
]

_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [''], [''], [''],
                        [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
                        [''], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''], [''], [''], ['']
                        , [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default='wdl/',
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--max_steps', type=int, default=10000, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=1024, help='Number of examples per batch.')

parser.add_argument(
    '--rnn_length', type=int, default=21, help='rnn list length.')

parser.add_argument(
    '--rnn_input_size', type=int, default=8, help='rnn each element size.')

parser.add_argument(
    '--rnn_hidden_size', type=int, default=8, help='rnn hidden size')

parser.add_argument(
    '--train_data', type=str, default='training_example_data/part-00000',
    help='Path to the training data.')

parser.add_argument(
    '--output_model', type=str, default='output_model/',
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default='test_data/part-00002',
    help='Path to the test data.')

#
f = open('fg_file2')
contents = f.read()
f.close()
fg = json.loads(contents)
fg_detail = fg['features']

def build_model_columns():
    base_columns = []
    crossed_columns = []
    deep_columns = []
    for item in fg_detail:
        if item['feature_type'] == 'id_feature':
            hash_bucket_size = item['hash_bucket_size']
            if 'wide_feature' in item.keys() and item['wide_feature'] == True:
                base_columns.append(tf.feature_column.categorical_column_with_hash_bucket(
                    item['expression'], hash_bucket_size=hash_bucket_size))
            elif 'embedding_dimension' in item.keys():
                embedding_dimension = item['embedding_dimension']
                deep_columns.append(tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
                    item['expression'], hash_bucket_size=hash_bucket_size), dimension=embedding_dimension))

        if item['feature_type'] == 'raw_feature':
            if 'wide_feature' in item.keys() and item['wide_feature'] == True:
                base_columns.append(tf.feature_column.numeric_column(item['expression']))
            else:
                deep_columns.append(tf.feature_column.numeric_column(item['expression']))

        if item['feature_type'] == 'combo_feature':
            hash_bucket_size = item['hash_bucket_size']
            crossed_keys = item['expression']
            print(crossed_keys)
            crossed_columns.append(tf.feature_column.crossed_column(crossed_keys, hash_bucket_size=hash_bucket_size))

        if item['feature_type'] in ('lookup_feature', 'match_feature'):
            deep_columns.append(tf.feature_column.numeric_column(item['feature_name']))

    #pids = tf.feature_column.categorical_column_with_hash_bucket("pids", 100000)
    cids = tf.feature_column.categorical_column_with_hash_bucket("cids", 10000)
    pid = tf.feature_column.categorical_column_with_hash_bucket("product_id", 100000)
    cid = tf.feature_column.categorical_column_with_hash_bucket("cid", 10000)

    #pids_weighted_column = tf.feature_column.weighted_categorical_column(
    #    categorical_column=pids, weight_feature_key='weights')

    cids_weighted_column = tf.feature_column.weighted_categorical_column(
        categorical_column=cids, weight_feature_key='weights')

    pids = []
    for i in range(20):
        pids.append(tf.feature_column.categorical_column_with_hash_bucket("pids" + str(i), 100000))
    pids.append(pid)
    pid_embed = tf.feature_column.shared_embedding_columns(pids, 8, combiner='sum')
    cid_embed = tf.feature_column.shared_embedding_columns([cids_weighted_column, cid], 16, combiner='sum')

    #deep_columns += pid_embed
    deep_columns += cid_embed
    rnn_columns = pid_embed

    wide_columns = base_columns + crossed_columns
    return wide_columns, deep_columns, rnn_columns

def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns, rnn_columns = build_model_columns()
    print(wide_columns)
    print(deep_columns)
    hidden_units = [512, 256,128]

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    #run_config = tf.estimator.RunConfig().replace(
    #    session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
        #config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units)
        #config=run_config)
    else:
        return DNNLinearCombinedWithRNNClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            rnn_feature_columns=rnn_columns,
            dnn_hidden_units=hidden_units)
        #config=run_config)


def input_fn2(data_file, shuffle, batch_size):
    import random
    import numpy as np
    if shuffle:
        random.shuffle(data_file)

    def get_content(record):
        fields = record.decode('utf-8').split("\t")
        index = 0
        elems = []
        realPidCnt = 0
        for item in fields:
            if _CSV_COLUMNS[index] not in ('pids', 'cids', 'weights'):
                if _CSV_COLUMN_DEFAULTS[index][0] == '':
                    elems.append(item)
                else:
                    try:
                        elems.append(np.float32(item))
                    except:
                        elems.append(np.float32(0.0))
            else:
                if _CSV_COLUMNS[index] != 'weights':
                    item_splits = item.split(',')
                    elems_tmp = []
                    if _CSV_COLUMNS[index] == 'pids':
                        for item_split in item_splits:
                            elems_tmp.insert(0, item_split)
                            if item_split != '':
                                realPidCnt += 1
                        elems += elems_tmp
                    else:
                        elems.append(item_splits)
                else:
                    weight_splits = [np.float32(x) for x in item.split(',')]
                    weights_new = []
                    nonzero_cnt = 0
                    for weight_split in weight_splits:
                        if nonzero_cnt < realPidCnt:
                            if weight_split > 0:
                                weights_new.append(weight_split)
                            else:
                                weights_new.append(np.float32(0.01))
                        else:
                            weights_new.append(np.float32(0.0))
                        nonzero_cnt += 1
                    elems.append(weights_new)
            index += 1

        return elems

    def parse_csv(value):
        print('Parsing', data_file)

        _CSV_COLUMNS_REAL = ['device_model'
            , 'brand'
            , 'resolution'
            , 'carrier'
            , 'access'
            , 'channel'
            , 'os'
            , 'province'
            , 'city'
            , 'ord_cnt_1d_type'
            , 'ord_cnt_7d_type'
            , 'ord_cnt_15d_type'
            , 'ord_cnt_30d_type'
            , 'ord_amt_1d_type'
            , 'ord_amt_7d_type'
            , 'ord_amt_15d_type'
            , 'ord_amt_30d_type'
            , 'ipv_1d_type'
            , 'ipv_7d_type'
            , 'ipv_15d_type'
            , 'ipv_30d_type'
            , 'product_id'
            , 'cid'
            , 'brand_id'
            , 'c1_id'
            , 'c2_id'
            , 'user_click_rate_1d'
            , 'user_fenxiang_rate_1d'
            , 'user_buy_rate_1d'
            , 'user_click_rate_1w'
            , 'user_fenxiang_rate_1w'
            , 'user_buy_rate_1w'
            , 'user_click_rate_2w'
            , 'user_fenxiang_rate_2w'
            , 'user_buy_rate_2w'
            , 'user_click_rate_1m'
            , 'user_fenxiang_rate_1m'
            , 'user_buy_rate_1m'
            , 'expose_norm_score_1d'
            , 'browse_norm_score_1d'
            , 'fenxiang_norm_score_1d'
            , 'buy_norm_score_1d'
            , 'expose_norm_score_1w'
            , 'browse_norm_score_1w'
            , 'fenxiang_norm_score_1w'
            , 'buy_norm_score_1w'
            , 'expose_norm_score_2w'
            , 'browse_norm_score_2w'
            , 'fenxiang_norm_score_2w'
            , 'buy_norm_score_2w'
            , 'expose_norm_score_1m'
            , 'browse_norm_score_1m'
            , 'fenxiang_norm_score_1m'
            , 'buy_norm_score_1m'
            , 'province_cid'
            , 'province_c2_id'
            , 'province_c1_id'
            , 'province_brand_id'
            , 'city_cid'
            , 'city_c2_id'
            , 'city_c1_id'
            , 'city_brand_id'
            , 'os_cid'
            , 'os_c2_id'
            , 'os_c1_id'
            , 'os_brand_id'
            , 'brand_cid'
            , 'brand_c2_id'
            , 'brand_c1_id'
            , 'brand_brand_id'
            , 'resolution_cid'
            , 'resolution_c2_id'
            , 'resolution_c1_id'
            , 'resolution_brand_id'
            , 'cid_prefer_1d'
            , 'cate1_prefer_1d'
            , 'cate2_prefer_1d'
            , 'brand_prefer_1d'
            , 'cid_prefer_1w'
            , 'cate1_prefer_1w'
            , 'cate2_prefer_1w'
            , 'brand_prefer_1w'
            , 'cid_prefer_2w'
            , 'cate1_prefer_2w'
            , 'cate2_prefer_2w'
            , 'brand_prefer_2w'
            , 'cid_prefer_1m'
            , 'cate1_prefer_1m'
            , 'cate2_prefer_1m'
            , 'brand_prefer_1m'
            , 'position'
            , 'match_type'
            , 'pids0'
            , 'pids1'
            , 'pids2'
            , 'pids3'
            , 'pids4'
            , 'pids5'
            , 'pids6'
            , 'pids7'
            , 'pids8'
            , 'pids9'
            , 'pids10'
            , 'pids11'
            , 'pids12'
            , 'pids13'
            , 'pids14'
            , 'pids15'
            , 'pids16'
            , 'pids17'
            , 'pids18'
            , 'pids19'
            , 'cids'
            , 'weights'
            , 'ctr_flag']
        out_type = []
        index = 0
        for item in _CSV_COLUMN_DEFAULTS:
            if item[0] == '' and _CSV_COLUMNS_REAL[index] != 'weights':
                out_type.append(tf.string)
            else:
                out_type.append(tf.float32)
            index += 1
        result = tf.py_func(get_content, [value], out_type)
        index = 0
        for item in result:
            if _CSV_COLUMNS_REAL[index] not in ('cids', 'weights'):
                result[index].set_shape([])
            else:
                result[index].set_shape([20])
            index += 1
        features = dict(zip(_CSV_COLUMNS_REAL, result))
        print(features)
        labels = features.pop('ctr_flag')
        return features, tf.equal(labels, '1.0')

    # Extract lines from input files using the Dataset API.
    print(data_file)
    dataset = tf.data.TextLineDataset(data_file)

    #if shuffle:
    #    dataset = dataset.shuffle(buffer_size=100000)

    dataset = dataset.map(parse_csv, num_parallel_calls=100)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    if shuffle:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    return dataset

def test_input_fn(data_file, batch_size):

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim='\t')
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('ctr_flag')
        return features, tf.equal(labels, '1.0')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv, num_parallel_calls=100)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.

    dataset = dataset.batch(batch_size)
    return dataset

#hdfs://bigdata/user/yipin.yang/search_ad_ctr/wdl/training_example_data/
def main(unused_argv):
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
    if not INPUT_PATH:
        FLAGS.train_data = None
    else:
        FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
        FLAGS.test_data = INPUT_PATH.get(FLAGS.test_data)
    import datetime
    train_data_file = FLAGS.train_data
    test_data_file = FLAGS.test_data
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn2(
        train_data_file, True, FLAGS.batch_size), max_steps=FLAGS.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn2(
        test_data_file, False, FLAGS.batch_size))

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # Evaluate accuracy.
    results = model.evaluate(input_fn=lambda: input_fn2(
        test_data_file, True, FLAGS.batch_size), steps=200)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


    if task_type == "worker" and task_index == 0:
        wide_columns, deep_columns, rnn_columns = build_model_columns()
        feature_columns_new = set(wide_columns + deep_columns + rnn_columns)
        ##print(feature_columns_new)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns_new)
        #print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




