# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:38:51 2020

@author: KHS
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

df0 = pd.read_csv('./KSJR_Car_Hacking_D_training-0(DS_CV).csv')
df1 = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV).csv')
df2 = pd.read_csv('./KSJR_Car_Hacking_D_training-2(DS_CV).csv')
df_x0 = df0[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x1 = df1[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x2 = df2[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y0 = df0['Class']
df_y1 = df1['Class']
df_y2 = df2['Class']
df_xd = pd.concat([df_x0,df_x1,df_x2]) # 드라이브 x
df_yd = pd.concat([df_y0,df_y1,df_y2]) # 드라이브 y

df0 = pd.read_csv('./KSJR_Car_Hacking_S_training-0(DS_CV).csv')
df1 = pd.read_csv('./KSJR_Car_Hacking_S_training-1(DS_CV).csv')
df2 = pd.read_csv('./KSJR_Car_Hacking_S_training-2(DS_CV).csv')
df_x0 = df0[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x1 = df1[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x2 = df2[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y0 = df0['Class']
df_y1 = df1['Class']
df_y2 = df2['Class']
df_xs = pd.concat([df_x0,df_x1,df_x2]) # 스탑 x 
df_ys = pd.concat([df_y0,df_y1,df_y2]) # 스탑 y

test_d_df = pd.read_csv ('./KSJR_Car_Hacking_D_prediction(DS_CV).csv')
test_xd = test_d_df[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
#test_yd = test_d_df['Class']

test_s_df = pd.read_csv ('./KSJR_Car_Hacking_S_prediction(DS_CV).csv')
test_xs = test_s_df[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
#test_ys = test_s_df['Class']

'''
scaler = StandardScaler()
scaler.fit(df_x)
scaler.fit(test_x)
x_test_scaled = scaler.transform(test_x)
x_scaled = scaler.transform(df_x)
'''

model_d = LGBMClassifier (random_state = 0, metric = 'binary_error',boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 100, num_leaves = 16, objective = 'binary') 
model_d.fit(df_xd, df_yd, verbose = 2)
pred_yd = model_d.predict(test_xd)

model_s = LGBMClassifier (random_state = 0, metric = 'binary_error',boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 100, num_leaves = 16, objective = 'binary') 
model_s.fit(df_xs, df_ys, verbose = 2)
pred_ys = model_s.predict(test_xs)

df_d = pd.DataFrame(pred_yd)
df_d.to_csv("./pred_d1.csv", index = True)
df_s = pd.DataFrame(pred_ys)
df_s.to_csv("./pred_s1.csv", index = True)

df_d_last = pd.read_csv('./pred_d1.csv')
df_s_last = pd.read_csv('./pred_s1.csv')

pred_yd = df_d_last['0']
pred_ys = df_s_last['0']
pred_y = pd.concat([pred_yd,pred_ys])


df_result = pd.DataFrame(pred_y)
df_result.to_csv("./정답_1.csv", index = True,index_label=False)
