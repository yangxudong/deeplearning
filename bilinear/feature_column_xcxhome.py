#-*- coding:utf-8 -*-

#定义feature_columns
import tensorflow as tf
from tensorflow import feature_column as fc

#-----------------------用户特征列-----------------------------
province=fc.indicator_column(fc.categorical_column_with_vocabulary_file('province','resource/province'))
region=fc.indicator_column(fc.categorical_column_with_vocabulary_list('region',['东北','华中','华东','华北','西北','华南','西南']))
city=fc.indicator_column(fc.categorical_column_with_vocabulary_file('city','resource/city'))
city_level=fc.indicator_column(fc.categorical_column_with_vocabulary_list('city_level',['一线城市','新一线城市','二线城市','三线城市','四线城市','五线城市']))
browser=fc.indicator_column(fc.categorical_column_with_vocabulary_list('browser',[0,1]))
os=fc.indicator_column(fc.categorical_column_with_vocabulary_list('os',['Android','android','devtools','unknown','iPhone','ios']))
ipv_7d_type=fc.indicator_column(fc.categorical_column_with_vocabulary_list('ipv_7d_type',[1,2,3,4]))
ipv_15d_type=fc.indicator_column(fc.categorical_column_with_vocabulary_list('ipv_15d_type',[1,2,3,4]))
ipv_30d_type=fc.indicator_column(fc.categorical_column_with_vocabulary_list('ipv_30d_type',[1,2,3,4]))
ipv_60d_type=fc.indicator_column(fc.categorical_column_with_vocabulary_list('ipv_60d_type',[1,2,3,4]))
#prefer_score=fc.numeric_column('prefer_score')
gender=fc.indicator_column(fc.categorical_column_with_vocabulary_list('gender',[0,1]))
#age=fc.numeric_column('age')
age_style=fc.indicator_column(fc.categorical_column_with_vocabulary_list('age_style',[00,10,20,30,40,50,60,70,80,90]))
baby1_gender=fc.indicator_column(fc.categorical_column_with_vocabulary_list('baby1_gender',[1100,1099,1098]))
baby1_age=fc.indicator_column(fc.categorical_column_with_vocabulary_list('baby1_age',[1409,6718,1349,1348,1351,1350]))
grade_id=fc.indicator_column(fc.categorical_column_with_vocabulary_list('grade_id',[1,2,3,4,5,6]))
is_new_byr_m=fc.indicator_column(fc.categorical_column_with_vocabulary_list('is_new_byr_m',[0,1]))
is_active_m=fc.indicator_column(fc.categorical_column_with_vocabulary_list('is_active_m',[0,1]))
is_new_1y=fc.indicator_column(fc.categorical_column_with_vocabulary_list('is_new_1y',['新客','1次购买者','近1年多单']))
bi_user_value=fc.indicator_column(fc.categorical_column_with_vocabulary_list('bi_user_value',['超级会员','高价值会员','普通高频会员','普通低频会员','低值高频会员','低值低频会员','一次购买者']))
bi_user_actv=fc.indicator_column(fc.categorical_column_with_vocabulary_list('bi_user_actv',[1,2,3,4]))
bi_rfm_type=fc.indicator_column(fc.categorical_column_with_vocabulary_list('bi_rfm_type',['流失','沉睡','1次购买者','低值低频会员','新客','低值高频会员','超级会员','近1年没购买','普通低频会员','普通高频会员','高价值会员']))


x_feature_columns=\
[
    province, region, city, city_level, browser, os,ipv_7d_type, ipv_15d_type, ipv_30d_type, ipv_60d_type,
    gender, age_style, baby1_gender,baby1_age, grade_id, is_new_byr_m, is_active_m, is_new_1y, bi_user_value,
    bi_user_actv, bi_rfm_type
]



#-----------------------------商品特征列------------------------------
c2_id=fc.indicator_column(fc.categorical_column_with_vocabulary_file('c2_id','resource/c2_id',dtype=tf.int32))
baby_gender=fc.indicator_column(fc.categorical_column_with_vocabulary_list('baby_gender',[1,2,3,4]))
baby_age=fc.indicator_column(fc.categorical_column_with_vocabulary_list('baby_age',[1,2,3,4]))
price_level=fc.indicator_column(fc.categorical_column_with_vocabulary_list('price_level',['A','B','C','D']))
expod_uv_1d=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_1d',[1,2,3,4]))
expod_uv_1d_srch=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_1d_srch',[1,2,3,4]))
clk_uv_1d=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_1d',[1,2,3,4]))
clk_cnt_rate_1d=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_cnt_rate_1d',[1,2,3,4]))
clk_uv_rate_1d=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_rate_1d',[1,2,3,4]))
crt_ord_byr_cnt_1d=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_byr_cnt_1d',[1,2,3,4]))
expod_uv_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_1w',[1,2,3,4]))
expod_uv_1w_srch=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_1w_srch',[1,2,3,4]))
clk_uv_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_1w',[1,2,3,4]))
clk_cnt_rate_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_cnt_rate_1w',[1,2,3,4]))
clk_uv_rate_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_rate_1w',[1,2,3,4]))
crt_ord_byr_cnt_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_byr_cnt_1w',[1,2,3,4]))
clk_cnt_rate_1m=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_cnt_rate_1m',[1,2,3,4]))
clk_uv_rate_1m=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_rate_1m',[1,2,3,4]))
expod_uv_1m_srch=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_1m_srch',[1,2,3,4]))
clk_cnt_rate_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_cnt_rate_2w',[1,2,3,4]))
clk_uv_rate_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_rate_2w',[1,2,3,4]))
clk_uv_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('clk_uv_2w',[1,2,3,4]))
expod_uv_2w_srch=fc.indicator_column(fc.categorical_column_with_vocabulary_list('expod_uv_2w_srch',[1,2,3,4]))
crt_trd_cnt_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_trd_cnt_1w',[1,2,3,4]))
crt_ord_itm_cnt_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_itm_cnt_1w',[1,2,3,4]))
crt_trd_cnt_1w_srch_d_lead=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_trd_cnt_1w_srch_d_lead',[1,2,3,4]))
crt_ord_pba_1w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_pba_1w',[1,2,3,4]))
crt_trd_cnt_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_trd_cnt_2w',[1,2,3,4]))
crt_ord_itm_cnt_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_itm_cnt_2w',[1,2,3,4]))
crt_trd_cnt_2w_srch_d_lead=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_trd_cnt_2w_srch_d_lead',[1,2,3,4]))
crt_ord_pba_2w=fc.indicator_column(fc.categorical_column_with_vocabulary_list('crt_ord_pba_2w',[1,2,3,4]))


z_feature_columns=\
[
    c2_id, baby_gender,baby_age, price_level, expod_uv_1d, expod_uv_1d_srch,clk_uv_1d,
    clk_cnt_rate_1d, clk_uv_rate_1d, crt_ord_byr_cnt_1d,expod_uv_1w, expod_uv_1w_srch, clk_uv_1w, clk_cnt_rate_1w,
    clk_uv_rate_1w, crt_ord_byr_cnt_1w, clk_cnt_rate_1m,clk_uv_rate_1m, expod_uv_1m_srch, clk_cnt_rate_2w,
    clk_uv_rate_2w, clk_uv_2w, expod_uv_2w_srch, crt_trd_cnt_1w,crt_ord_itm_cnt_1w, crt_trd_cnt_1w_srch_d_lead,
    crt_ord_pba_1w,crt_trd_cnt_2w, crt_ord_itm_cnt_2w, crt_trd_cnt_2w_srch_d_lead,crt_ord_pba_2w
]

