from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_numeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = price_column._get_dense_tensor(builder)
    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    # 使用input_layer
    price_transformed_tensor = feature_column.input_layer(price, [price_column])
    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor]))

#test_numeric()

def test_bucketized_column():
    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])
    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])
    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

#test_bucketized_column()

def test_categorical_column_with_vocabulary_list():
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

#test_categorical_column_with_vocabulary_list()

def test_embedding():
    tf.set_random_seed(1)
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_tensor = color_column._get_sparse_tensors(builder)

    color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

#test_embedding()

def test_categorical_column_with_hash_bucket():
    color_data = {'color': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

#test_categorical_column_with_hash_bucket()

def test_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_dense_tensor = feature_column.input_layer(color_data, [color_column_embed])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

test_embedding_column_with_hash_bucket()

def test_shared_embedding_column_with_hash_bucket():
    print('*' * 40)
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

#test_shared_embedding_column_with_hash_bucket()
