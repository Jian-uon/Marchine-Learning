# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : HuJian
# * Email         : swjtu_hj@163.com
# * Create time   : 2018-11-09 14:11
# * Last modified : 2018-11-09 14:11
# * Filename      : StudentPerformancePredict.py
# * Description   : TianChi notebook course
# **********************************************************
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('StudentPerformance.csv')
print(df.head())
print('shape: ', df.shape)
print(df.isnull().sum())
print(df.describe(include='all'))
print(df.info())
print('Relation', df['Relation'].unique())
print('Class', df['Class'].unique())

#sns.countplot(x = 'Class', order = ['L', 'M', 'H'], data = df)
#plt.show()

#sns.countplot(x='gender', data=df)
#plt.show()

#sns.set(rc={'figure.figsize':(14,8)})
#sns.countplot(x='Topic', data=df)
#plt.show()

#sns.set(rc={'figure.figsize':(20,10)})
#sns.countplot(x='Topic', hue = 'Class', hue_order=['L', 'M', 'H'], data=df)
#plt.show()

#sns.countplot(x='gender', hue='Class', data=df, order =['M', 'F'], hue_order=['L', 'M', 'H'])
#plt.show()

#sns.set(rc={'figure.figsize':(14,7)})
#sns.countplot(x='Topic', data=df, hue='gender')
#plt.show()

df_temp = df[['Topic', 'gender']]
df_temp['Count'] = 1
df_temp = df_temp.groupby(['Topic', 'gender']).agg('sum').reset_index()
print(df_temp.head(6))

df_temp2 = df_temp
df_temp2 = df_temp2.groupby('Topic').agg('sum').reset_index()
print(df_temp2.head())

df_temp = pd.merge(df_temp, df_temp2, on = ('Topic'), how= 'left')
print(df_temp.head())

df_temp['gender proportion in topic'] = df_temp['Count_x']/df_temp['Count_y']
print(df_temp.head())

#sns.countplot(x = 'SectionID', hue ='Class', data= df, hue_order= ['L', 'M', 'H'])
#plt.show()

'''
fig, axis = plt.subplots(2, 2, figsize=(14,10))
sns.barplot(x='Class', y = 'VisITedResources', data=df, order=['L', 'M', 'H'], ax = axis[0,0])
sns.barplot(x = 'Class',y = 'AnnouncementsView',data = df, order = ['L','M','H'],ax = axis[0,1])
sns.barplot(x = 'Class',y = 'raisedhands',data = df, order = ['L','M','H'],ax = axis[1,0])
sns.barplot(x = 'Class',y = 'Discussion',data = df, order = ['L','M','H'],ax = axis[1,1])
plt.show()
'''

#sns.set(rc={'figure.figsize':(15,10)})
#sns.swarmplot(x='Class', y='raisedhands', hue='gender', data=df, palette='coolwarm', order=['L', 'M', 'H'])
#plt.show()

#sns.set(rc={'figure.figsize':(8,6)})
#sns.boxplot(x='Class', y='Discussion', data=df, order=['L', 'M', 'H'])
#plt.show()

#fig, axis = plt.subplots(2,1,figsize=(10,10))
#sns.regplot(x='raisedhands', y='Discussion',order=4, data=df, ax=axis[0])
#sns.regplot(x = 'raisedhands',y = 'AnnouncementsView',order = 4, data = df, ax = axis[1])
#plt.show()

#corr = df[['VisITedResources','raisedhands','AnnouncementsView','Discussion']].corr()
#print(corr)
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#plt.show()

X = df.drop('Class', axis=1)
print(X.head())
Y = df['Class']
X = pd.get_dummies(X)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print(y_test.head(5))

Logit = LogisticRegression()
Logit.fit(X_train, y_train)
Predict = Logit.predict(X_test)
print('Predict', Predict)
Score = accuracy_score(y_test, Predict)
print(Score)

# Feature Engineering 特征工程，删除了原有的 Section
df2 = df
X2 = df2.drop('Class', axis=1)
X2 = X2.drop('SectionID', axis=1)
Y2 = df2['Class']
X2 = pd.get_dummies(X2)
print(X2.head())

X2_train, X2_test, y2_train, y2_test = train_test_split(X2,Y2, test_size = 0.2,random_state = 10)
Logit = LogisticRegression()
Logit.fit(X2_train, y2_train)
Predict = Logit.predict(X2_test)
Score = accuracy_score(y2_test, Predict)
print(Score)

#增加新特征
df3 = df
df3['DiscussionPlusVisit'] = df3['Discussion'] + df3['VisITedResources']
X3 = df3.drop('Class',axis = 1)
Y3 = df3['Class']
X3 = pd.get_dummies(X3) 
print(X3.head(5))

X3_train, X3_test, y3_train, y3_test = train_test_split(X3,Y3, test_size = 0.2,random_state = 10)
Logit = LogisticRegression()
Logit.fit(X3_train, y3_train)
Predict = Logit.predict(X3_test)
Score = accuracy_score(y3_test, Predict)
print(Score)



