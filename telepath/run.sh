#!/bin/sh

set -e -x 
export GRPC_VERBOSITY="DEBUG"
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/telepath/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/telepath-output/${bizdate}
#hadoop fs -rm -f hdfs://bigdata/user/xudong.yang/telepath/samples/pt=${bizdate}/_SUCCESS

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "telepath_dnn" \
   --board-logdir hdfs://bigdata/tmp/telepath/${bizdate} \
   --input hdfs://bigdata/user/huan.lin/TeleNet/TFrecord#train_data\
   --files hdfs://bigdata/user/xudong.yang/telepath/model/conv_blocks.py,hdfs://bigdata/user/xudong.yang/telepath/model/estimator_release.py,hdfs://bigdata/user/xudong.yang/telepath/model/mobilenet.py,hdfs://bigdata/user/xudong.yang/telepath/model/mobilenet_v2.py,hdfs://bigdata/user/xudong.yang/telepath/model/telenet_basenet.py,hdfs://bigdata/user/xudong.yang/telepath/model/telenet_fn.py,hdfs://bigdata/user/xudong.yang/telepath/model/telenet_model_mobilenet.py \
   --launch-cmd "python estimator_release.py --train_steps=100 --train_data train_data --eval_data train_data --model_dir hdfs://bigdata/tmp/telepath/${bizdate} --output_model hdfs://bigdata/tmp/telepath-output/${bizdate}" \
   --worker-memory 10G \
   --worker-num 6 \
   --worker-cores 8 \
   --ps-memory 3G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

#hadoop fs -mkdir /user/xudong.yang/telepath/model_output
#model_fold=`hadoop fs -ls hdfs://bigdata/tmp/telepath-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
#hadoop fs -rm -r -f /user/xudong.yang/telepath/model_output/*
#hadoop fs -cp ${model_fold}/* /user/xudong.yang/telepath/model_output
