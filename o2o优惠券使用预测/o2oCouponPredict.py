# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : HuJian
# * Email         : swjtu_hj@163.com
# * Create time   : 2018-10-31 16:43
# * Last modified : 2018-10-31 16:43
# * Filename      : o2oCouponPredict.py
# * Description   : 天池O2O优惠券代码分享学习
# **********************************************************
import pandas as pd
import numpy as np
from datetime import date
import datetime as dt

#将数据分为3个数据集 利用滑窗法
# 数据特征提取         测试集
# 1月1日到4月13日   4月14日到5月14日
# 2月1日到5月14日   5月15日到6月15日
# 3月15日到6月30日  7月1日到7月31日

'''
1.提取用户特征
    距离
    用户的平均距离，用户的最小距离，用户的最大距离
    使用优惠券买的物品数，买的总数，收到的优惠券数
    使用优惠券买的/总共收到的优惠券
'''
#利用pandas读取csv个格式的数据,header=None表示原文件没有索引
#keep_default_na=False 会使原本是nan的数据存为""而不是null
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv', header=0, keep_default_na=False)
#print(off_train.head())
#                                                                           消费日期
#   User_id  Merchant_id Coupon_id Discount_rate Distance Date_received      Date
#0  1439408         2632      null          null        0          null  20160217
#1  1439408         4663     11002        150:20        1      20160528      null
#2  1439408         2632      8591          20:1        0      20160217      null
#3  1439408         2632      1078          20:1        0      20160319      null
#4  1439408         2632      8591          20:1        0      20160613      null

off_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
 
 
off_test = pd.read_csv("data/ccf_offline_stage1_test_revised.csv",header=0,keep_default_na=False)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']

on_train = pd.read_csv("data/ccf_online_stage1_train.csv",header=0,keep_default_na=False)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']

#———————————————————依据滑窗法划分训练集和测试集————————————————————————————
# 领券    消费的      包括
# 领券    未消费的    包括
# 未领券  消费的      包括
# 未领券  未消费的

#使数据集3等于test集
dataset3 = off_test

#数据集3的特征为 取线下数据中领券和用券日期大于3月15日和小于6月30日的
feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]

#提取数据集2的测试集
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
#在2月1日到5月14日之间使用了券,只要领取时间在2月1日到5月14日之间,并包括没有数据中没有领取券的
feature2 = off_train[(off_train.date>='20160201')&(off_train.date<='20160514')|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]

#同理可得
dataset1 = off_train[(off_train.date_received>='201604014')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date>='20160101')&(off_train.date<='20160413')|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]
#—————————————————————————————————————完毕——————————————————————————————————

#t代表数据的特征t ,t1 ,t2........

#对于测试集3
#t-用户的Id出现次数
t = dataset3[['user_id']]
#相当于给原有数据加上一列，这个月用户收取的所有优惠券数目，并初始化为1
t['this_month_user_received_all_coupon_count'] = 1
#将t按照用户id进行分组，然后统计所有用户收取的优惠券数目,并初始化一个索引值
t = t.groupby('user_id').agg('sum').reset_index()

#t1-相同优惠券Id和用户Id数
t1 = dataset3[['user_id', 'coupon_id']]
#提取这个月用户收到的相同的优惠券的数量
t1['this_month_user_receive_same_coupon_count'] = 1
t1 = t1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()

#t2- user_id&coupon_id&date_received
t2 = dataset3[['user_id', 'coupon_id', 'date_received']]
t2.date_received = t2.date_received.astype('str')
t2 = t2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
t2['received_number'] = t2.date_received.apply(lambda s: len(s.split(':')))
t2 = t2[t2.received_number>1]

#print(t2.head())

#最大接受的日期
t2['max_date_received'] = t2.date_received.apply(lambda x: max([int(i) for i in x.split(':')]))
#最小接受的日期
t2['min_date_received'] = t2.date_received.apply(lambda x: min([int(i) for i in x.split(':')]))

t2 = t2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

t3 = dataset3[['user_id','coupon_id','date_received']]
#将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
t3 = pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
#这个优惠券最近接受时间
t3['this_month_user_receive_same_coupon_lastone']= t3.max_date_received-t3.date_received.astype(int)
#这个优惠券最远接受时间
t3['this_month_user_receive_same_coupon_firstone'] = t3.date_received.astype(int)-t3.min_date_received

def is_firstlastone(x):
    if x==0:
        return 1
    elif x >0:
        return 0
    else:
        return -1 #表示优惠券只接受了一次

t3.this_month_user_receive_same_coupon_lastone = t3.this_month_user_received_same_coupon_lastone.apply(is_firstlastone)
t3.this_month_user_receive_same_coupon_firstone = t3.this_month_user_received_same_coupon_firstone.apply(is_firstlastone)
t3 = t3[['user_id', 'coupon_id', 'date_received', 'this_month_user_received_same_coupon_lastone', 'this_month_user_received_same_coupon_firstone']]

#提取第四个特征,一个用户所接收到的所有优惠券的数量
t4 = dataset3[['user_id', 'date_received']]
t4['this_day_receive_all_coupon_count'] = 1
t4 = t4.groupby([['user_id', 'date_received']]).agg('sum').reset_index()

#提取第五个特征,一个用户不同时间所接收到不同优惠券的数量
t6 = dataset3[['user_id', 'coupon_id', 'date_received']]
t6.date_received = t6.date_received.astype('str')
t6=t6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x:':'.join(x)).reset_index()
#重命名inplace代表深拷贝
t6.rename(columns={'date_received':'dates'}, inplace = True)

def get_day_gap_before(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        #将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))-dt.date(int(d[0:4]),int(d[4:6]),int(d[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)

def get_day_gap_after(s):
    date_received,dates = s.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]),int(d[4:6]),int(d[6:8]))-dt.datetime(int(date_received[0:4]),int(date_received[4:6]),int(date_received[6:8]))).days
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)

t7 = dataset3[['user_id','coupon_id','date_received']]
#将t6和t7融合
t7 = pd.merge(t7,t6,on=['user_id','coupon_id'],how='left')
#注意这里所有的时间格式都已经是'str'格式
t7['date_received_date'] = t7.date_received.astype('str')+'-'+t7.dates
#print(t7)
t7['day_gap_before'] = t7.date_received_date.apply(get_day_gap_before)
t7['day_gap_after'] = t7.date_received_date.apply(get_day_gap_after)
t7 = t7[['user_id','coupon_id','date_received','day_gap_before','day_gap_after']]

#将所有特征融合在一张表中
other_feature3 = pd.merge(t1,t,on='user_id')
other_feature3 = pd.merge(other_feature3,t3,on=['user_id','coupon_id'])
other_feature3 = pd.merge(other_feature3,t4,on=['user_id','date_received'])
other_feature3 = pd.merge(other_feature3,t5,on=['user_id','coupon_id','date_received'])
other_feature3 = pd.merge(other_feature3,t7,on=['user_id','coupon_id','date_received'])
other_feature3.to_csv('other_feature3.csv',index=None)
#print(other_feature3)











