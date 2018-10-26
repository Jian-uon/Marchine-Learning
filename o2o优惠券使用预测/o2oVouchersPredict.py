# * 赛题介绍：本赛题目标是预测投放的优惠券是否核销。
# * 数据分析: 其中包含一些常见的数据处理，比如str to numeric。
# * 特征工程: 主要是简单的提取了折扣相关信息和时间信息。
# * 模型建立: 简单的线性模型。
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression
 
dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
#dfon = pd.read_csv('data/ccf_online_stage1_train.csv')
print('data read end.')

print('offline data:')
print(dfoff.head())

#折扣类型转换:
#缺失返回np.nan
#XX:YY类型返回1
#0.XX类型返回0
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0

#转换折扣->float
#缺失返回1.0
#XX:YY类型返回float的折扣率
#0.XX类型直接返回折扣率
def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

#获取XX:YY类型打折满足条件
def getDiscountDemand(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

#获取XX:YY类型打折满足条件下减少的值
def getDiscountSaved(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

print("tool is ok.")

#在上述函数下获取表格新的四列
#rate, demand, saved, type
#将距离中na值填充为-1
def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_demand'] = df['Discount_rate'].apply(getDiscountDemand)
    df['discount_saved'] = df['Discount_rate'].apply(getDiscountSaved)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df

dfoff = processData(dfoff)
dftest = processData(dftest)

#下边处理Date

date_received = dfoff['Date_received'].unique()
#pd.notnull()返回序列对应的布尔类型的值
date_received = sorted(date_received[pd.notnull(date_received)])

#下边处理购买信息
date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date'])

couponbydate = dfoff[dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_recevied', 'Date']

buybydate = dfoff[dfoff['Date_received'].notnull() & dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()

print('end')

#将文件中日期转为date类型来获取是星期几
def getWeekDay(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday()+1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekDay)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekDay)

#weekday_type:周末为1,工作日为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x:1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x:1 if x in [6,7] else 0)

#将weekday转换成独热码
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf
 
tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf

#判断是否在15天内使用优惠券
#不存在返回-1
#在15天内返回1
#不在返回0
def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0
#axis=1表示对第二维(y轴方向)进行操作 有点没太懂。。。
dfoff['label'] = dfoff.apply(label, axis=1)
print('end')

#data split
print("-----data split------")
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()
print('end')

#根据特征训练模型
original_feature = ['discount_rate','discount_type','discount_demand','discount_saved','distance', 'weekday', 'weekday_type'] + weekdaycols
#......这也太暴力了
print("----train-----")
model = SGDClassifier(#lambda:
    loss='log',
    penalty='elasticnet',
    fit_intercept=True,
    max_iter=100,
    shuffle=True,
    alpha = 0.01,
    l1_ratio = 0.01,
    n_jobs=1,
    class_weight=None
)
model.fit(train[original_feature], train['label'])

#保存模型
print("---save model---")
with open('1_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('1_model.pkl', 'rb') as f:
    model = pickle.load(f)

# test prediction for submission
y_test_pred = model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit2.csv', index=False, header=False)
dftest1.head()

