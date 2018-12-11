#!/bin/sh
set -e -x 
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/deepfm/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/deepfm-output/${bizdate}

train_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/samples | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/eval     | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "mainstream_deepfm_model" \
   --board-logdir hdfs://bigdata/tmp/deepfm/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/deepfm/train_deepfm.py,hdfs://bigdata/user/xudong.yang/deepfm/deepfm_input_fn.py,hdfs://bigdata/user/xudong.yang/deepfm/deepfm.py \
   --launch-cmd "python train_deepfm.py --learning_rate=0.01 --shuffle_buffer_size=30000 --save_checkpoints_steps=10000 --train_steps=200000 --batch_size=256\
     --dropout_rate=0.5 --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/deepfm/${bizdate} --output_model hdfs://bigdata/tmp/deepfm-output/${bizdate}" \
   --worker-memory 12G \
   --worker-num 6 \
   --worker-cores 8 \
   --ps-memory 4G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

set +e
hadoop fs -test -e /user/xudong.yang/mainstream/deepfm_output/
[ $? -ne 0 ] && hadoop fs -mkdir /user/xudong.yang/mainstream/deepfm_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/deepfm-output/${bizdate} | tail -n 1 | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/deepfm_output/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/deepfm_output