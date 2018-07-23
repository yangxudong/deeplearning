'''创建dataset与input_fn'''

import tensorflow as tf
import os

shuffle_buffer_size=140000

X_COLUMN_NAMES=\
[
    'province','region','city','city_level','browser','os','ipv_7d_type','ipv_15d_type','ipv_30d_type',
    'ipv_60d_type','gender','age_style','baby1_gender','baby1_age','grade_id',
    'is_new_byr_m','is_active_m','is_new_1y','bi_user_value','bi_user_actv','bi_rfm_type'
]

Z_COLUMN_NAMES=\
[
    'c2_id','baby_gender','baby_age','price_level','expod_uv_1d','expod_uv_1d_srch','clk_uv_1d',
    'clk_cnt_rate_1d','clk_uv_rate_1d','crt_ord_byr_cnt_1d','expod_uv_1w','expod_uv_1w_srch','clk_uv_1w',
    'clk_cnt_rate_1w','clk_uv_rate_1w','crt_ord_byr_cnt_1w','clk_cnt_rate_1m','clk_uv_rate_1m',
    'expod_uv_1m_srch','clk_cnt_rate_2w','clk_uv_rate_2w','clk_uv_2w','expod_uv_2w_srch','crt_trd_cnt_1w',
    'crt_ord_itm_cnt_1w','crt_trd_cnt_1w_srch_d_lead','crt_ord_pba_1w','crt_trd_cnt_2w','crt_ord_itm_cnt_2w',
    'crt_trd_cnt_2w_srch_d_lead','crt_ord_pba_2w'
]

Y_COLUMN_NAME=['label']

CSV_COLUMN_NAMES=\
[
    #用户特征列21个
    'province','region','city','city_level','browser','os','ipv_7d_type','ipv_15d_type','ipv_30d_type',
    'ipv_60d_type','gender','age_style','baby1_gender','baby1_age','grade_id',
    'is_new_byr_m','is_active_m','is_new_1y','bi_user_value','bi_user_actv','bi_rfm_type',
    #商品特征列31个
    'c2_id','baby_gender','baby_age','price_level','expod_uv_1d','expod_uv_1d_srch','clk_uv_1d',
    'clk_cnt_rate_1d','clk_uv_rate_1d','crt_ord_byr_cnt_1d','expod_uv_1w','expod_uv_1w_srch','clk_uv_1w',
    'clk_cnt_rate_1w','clk_uv_rate_1w','crt_ord_byr_cnt_1w','clk_cnt_rate_1m','clk_uv_rate_1m',
    'expod_uv_1m_srch','clk_cnt_rate_2w','clk_uv_rate_2w','clk_uv_2w','expod_uv_2w_srch','crt_trd_cnt_1w',
    'crt_ord_itm_cnt_1w','crt_trd_cnt_1w_srch_d_lead','crt_ord_pba_1w','crt_trd_cnt_2w','crt_ord_itm_cnt_2w',
    'crt_trd_cnt_2w_srch_d_lead','crt_ord_pba_2w',
    #标签列
    'label'
]

CSV_TYPES=\
[
    #用户特征默认值
    [''],[''],[''],[''],[0],[''],[0],[0],[0],
    [0],[0],[0],[0],[0],[0],
    [0],[0],[''],[''],[0],[''],
    #商品特征默认值
    [0],[0],[0],[''],[0],[0],[0],
    [0],[0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],[0],
    [0],[0],[0],[0],[0],
    [0],[0],
    #标签默认值
    [0.0]
]

def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES,field_delim='\t')
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    labels = features.pop('label')
    return features,labels

def csv_input_fn(csv_dir, batch_size):
    csv_paths=[]
    for filename in os.listdir(csv_dir):
        csv_paths.append(os.path.join(csv_dir,filename))
    dataset = tf.data.TextLineDataset(csv_paths)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat().batch(batch_size)
    return dataset