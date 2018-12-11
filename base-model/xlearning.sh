#!/bin/sh

set -e -x 
export GRPC_VERBOSITY="DEBUG"
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/base-model/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/base-model-output/${bizdate}
#hadoop fs -rm -f hdfs://bigdata/user/xudong.yang/mainstream/samples/pt=${bizdate}/_SUCCESS
train_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/samples | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/eval     | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "mainstream_dnn_base_model" \
   --board-logdir hdfs://bigdata/tmp/base-model/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/mainstream/model/base-model-weighted.py \
   --launch-cmd "python base-model-weighted.py --save_checkpoints_steps=10000 --train_steps=200000 --batch_size=256 --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/base-model/${bizdate} --output_model hdfs://bigdata/tmp/base-model-output/${bizdate}" \
   --worker-memory 10G \
   --worker-num 6 \
   --worker-cores 8 \
   --ps-memory 3G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

#hadoop fs -mkdir /user/xudong.yang/mainstream/model_output_v2/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/base-model-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/model_output_v2/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/model_output_v2