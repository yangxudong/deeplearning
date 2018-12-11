#!/bin/sh
set -e -x 
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/esmm-model/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/esmm-model-output/${bizdate}
hadoop fs -rm -f hdfs://bigdata/user/xudong.yang/bd_mainstream/samples/pt=${bizdate}/_SUCCESS

train_data_file=`hadoop fs -ls /user/xudong.yang/bd_mainstream/samples | tail -n 5 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls /user/xudong.yang/bd_mainstream/eval     | tail -n 5 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "bd_mainstream_esmm_model" \
   --board-logdir hdfs://bigdata/tmp/esmm-model/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/mainstream/model/esmm.py \
   --launch-cmd "python esmm.py --hidden_units=512,256 --learning_rate=0.005 --shuffle_buffer_size=300000 --save_checkpoints_steps=10000 --train_steps=100000 --batch_size=512\
     --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/esmm-model/${bizdate} --output_model hdfs://bigdata/tmp/esmm-model-output/${bizdate}" \
   --worker-memory 10G \
   --worker-num 6 \
   --worker-cores 8 \
   --ps-memory 3G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

#hadoop fs -mkdir /user/xudong.yang/mainstream/esmm_model_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/esmm-model-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/esmm_model_output/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/esmm_model_output