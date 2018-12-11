#!/bin/sh
set -e -x 
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/bbdin/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/bbdin-output/${bizdate}

train_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/samples | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls /user/xudong.yang/mainstream/eval     | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "mainstream_din_model" \
   --board-logdir hdfs://bigdata/tmp/bbdin/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/din/train_bb.py,hdfs://bigdata/user/xudong.yang/din/bb_input_fn.py,hdfs://bigdata/user/xudong.yang/din/deep_interest_network.py \
   --launch-cmd "python train_bb.py --learning_rate=0.01 --attention_hidden_units=16 --shuffle_buffer_size=25600 --save_checkpoints_steps=10000 --train_steps=200000 --batch_size=256\
     --dropout_rate=0.5 --optimizer=Adagrad --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/bbdin/${bizdate} --output_model hdfs://bigdata/tmp/bbdin-output/${bizdate}" \
   --worker-memory 12G \
   --worker-num 10 \
   --worker-cores 8 \
   --ps-memory 10G \
   --ps-num 1 \
   --ps-cores 8 \
   --queue default


#hadoop fs -mkdir /user/xudong.yang/mainstream/bbdin_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/bbdin-output/${bizdate} | tail -n 1 | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/bbdin_output/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/bbdin_output