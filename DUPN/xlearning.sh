#!/bin/sh
set -e -x 
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/dupn-share/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/dupn-share-output/${bizdate}

train_data_file=`hadoop fs -ls hdfs://bigdata/user/xudong.yang/bd_mainstream/new_samples/ | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls hdfs://bigdata/user/xudong.yang/bd_mainstream/new_eval/     | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "bd_mainstream_share_dupn_model" \
   --board-logdir hdfs://bigdata/tmp/dupn-share/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/mainstream/model/dupn.py \
   --launch-cmd "python dupn.py --hidden_units=512,256 --learning_rate=0.005 --shuffle_buffer_size=10000 --save_checkpoints_steps=10000 --train_steps=100000 --batch_size=256\
     --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/dupn-share/${bizdate} --output_model hdfs://bigdata/tmp/dupn-share-output/${bizdate}" \
   --worker-memory 15G \
   --worker-num 5 \
   --worker-cores 8 \
   --ps-memory 5G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

set +e
hadoop fs -test -e /user/xudong.yang/mainstream/dupn_share_output/
[ $? -ne 0 ] && hadoop fs -mkdir /user/xudong.yang/mainstream/dupn_share_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/dupn-share-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/dupn_share_output/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/dupn_share_output