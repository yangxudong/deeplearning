#构建模型
import tensorflow as tf
import numpy as np

from dataset_xcxhome import csv_input_fn,X_COLUMN_NAMES,Z_COLUMN_NAMES
from feature_column_xcxhome import x_feature_columns,z_feature_columns   #导入特征列

train_steps,eval_steps=20000,2000
train_dir,test_dir=r'data/train',r'data/test'
batch_size=256
gamma=1#sigmoid函数偏置
lamda=0.5#l2正则,参数平方和*λ/2
learning_rate = 0.0001


def my_model(features,labels, mode, params):
    #get Tensor x and Tensor z
    x_features,z_features={},{}
    for key in X_COLUMN_NAMES:
        x_features[key]=features[key]
    for key in Z_COLUMN_NAMES:
        z_features[key]=features[key]
    x = tf.feature_column.input_layer(x_features,params['x_feature_columns'])
    z = tf.feature_column.input_layer(z_features, params['z_feature_columns'])
    x_dim = x.shape[1]  # 用户向量维度
    z_dim = z.shape[1]  # 商品向量维度
    #模型参数
    W = tf.Variable(np.random.normal(scale=0.001,size=[x_dim, z_dim]).astype(np.float32))
    mu = tf.Variable(np.random.normal(scale=0.001,size=[1]).astype(np.float32))
    #构建模型
    t = tf.matmul(x,W)
    t = tf.multiply(t,z)
    s = tf.reduce_sum(t,axis=1)+mu
    logits = s+gamma
    # mode=predict
    probabilities = tf.nn.sigmoid(logits)
    predicted_classes = tf.cast(tf.round(probabilities),tf.float32)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': probabilities,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # loss
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))  # 交叉熵
    l1_regular = tf.contrib.layers.l1_regularizer(lamda)(W)
    l2_regular = tf.contrib.layers.l2_regularizer(lamda)(W)
    loss = cross_entropy
    #accuracy
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    #auc
    auc = tf.metrics.auc(labels=labels,predictions=predicted_classes,name='auc_op')
    metrics = {'accuracy': accuracy,'auc':auc}
    #mode=eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,eval_metric_ops=metrics)

    #mode=train
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'x_feature_columns': x_feature_columns,
            'z_feature_columns': z_feature_columns,
            'n_classes':2
        })

    classifier.train(
        input_fn=lambda: csv_input_fn(train_dir, batch_size),
        steps=train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda: csv_input_fn(test_dir, batch_size),steps=eval_steps)

    print(eval_result)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)