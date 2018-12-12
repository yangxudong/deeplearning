#!/bin/sh
set -e -x 
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/esmm-dcn/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/esmm-dcn-output/${bizdate}

train_data_file=`hadoop fs -ls /user/xudong.yang/bd_mainstream/samples | tail -n 5 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls /user/xudong.yang/bd_mainstream/eval     | tail -n 5 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "bd_mainstream_esmm_dcn" \
   --board-logdir hdfs://bigdata/tmp/esmm-dcn/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/esmm/esmm.py,hdfs://bigdata/user/xudong.yang/esmm/train.py,hdfs://bigdata/user/xudong.yang/esmm/dcn_input_fn.py,hdfs://bigdata/user/xudong.yang/esmm/dcn_logit_fn.py,hdfs://bigdata/user/xudong.yang/esmm/din_logit_fn.py \
   --launch-cmd "python train.py --hidden_units=256,256 --learning_rate=0.0005 --shuffle_buffer_size=12800 --save_checkpoints_steps=10000 --train_steps=10000 --batch_size=128\
     --use_batch_norm=false --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/esmm-dcn/${bizdate} --output_model hdfs://bigdata/tmp/esmm-dcn-output/${bizdate}" \
   --worker-memory 15G \
   --worker-num 8 \
   --worker-cores 8 \
   --ps-memory 5G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

set +e
hadoop fs -test -e /user/xudong.yang/mainstream/esmm_dcn_output/
[ $? -ne 0 ] && hadoop fs -mkdir /user/xudong.yang/mainstream/esmm_dcn_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/esmm-dcn-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/esmm_dcn_output/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/esmm_dcn_output

################################################################################################################################################################
#!/bin/sh
set -e -x
export HADOOP_HOME=/data2/hadoop-2.6.0-cdh5.4.0

hadoop fs -rm -r -f hdfs://bigdata/tmp/dupn-esmm/${bizdate}
hadoop fs -rm -r -f hdfs://bigdata/tmp/dupn-esmm-output/${bizdate}

train_data_file=`hadoop fs -ls hdfs://bigdata/user/xudong.yang/bd_mainstream/new_samples/ | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
test_data_file=`hadoop fs -ls hdfs://bigdata/user/xudong.yang/bd_mainstream/new_eval/     | tail -n 7 |awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`

/opt/beibei/xlearning/bin/xl-submit \
   --app-type "tensorflow" \
   --input-strategy "Placeholder" \
   --app-name "bd_mainstream_esmm_dupn_model" \
   --board-logdir hdfs://bigdata/tmp/dupn-esmm/${bizdate} \
   --input ${train_data_file}#train_data\
   --input ${test_data_file}#eval_data\
   --files hdfs://bigdata/user/xudong.yang/esmm/ \
   --launch-cmd "python esmm/train.py --hidden_units=512,256 --learning_rate=0.005 --shuffle_buffer_size=10000 --save_checkpoints_steps=10000 --train_steps=120000 --batch_size=256\
     --train_data train_data --eval_data eval_data --model_dir hdfs://bigdata/tmp/dupn-esmm/${bizdate} --output_model hdfs://bigdata/tmp/dupn-esmm-output/${bizdate}" \
   --worker-memory 15G \
   --worker-num 6 \
   --worker-cores 8 \
   --ps-memory 5G \
   --ps-num 1 \
   --ps-cores 5 \
   --queue default

set +e
hadoop fs -test -e /user/xudong.yang/mainstream/dupn_esmm_output/
[ $? -ne 0 ] && hadoop fs -mkdir /user/xudong.yang/mainstream/dupn_esmm_output/
model_fold=`hadoop fs -ls hdfs://bigdata/tmp/dupn-esmm-output/${bizdate} | awk -F" " '{print $8}' | xargs | sed -e 's/ /,/g'`
hadoop fs -rm -r -f /user/xudong.yang/mainstream/dupn_esmmoutput/*
hadoop fs -cp ${model_fold}/* /user/xudong.yang/mainstream/dupn_esmm_output
