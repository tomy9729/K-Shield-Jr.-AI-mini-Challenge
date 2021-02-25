import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier

df0 = pd.read_csv('./KSJR_Car_Hacking_D_training-0(DS_CV)_0.csv')
df1 = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV)_0.csv')
df2 = pd.read_csv('./KSJR_Car_Hacking_D_training-2(DS_CV)_0.csv')
df_x0 = df0[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x1 = df1[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_x2 = df2[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]

df_y0 = df0['Class']
df_y1 = df1['Class']
df_y2 = df2['Class']

df_x = pd.concat([df_x0,df_x1,df_x2])
df_y = pd.concat([df_y0,df_y1,df_y2])
kfold = KFold(n_splits=10, shuffle = True, random_state = 0) #10-fold cross validation 

model9 = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1, eval_metric = 'error',objective = 'binary:logistic',use_label_encoder=False)
scores9 = cross_val_score(model9, df_x, df_y, cv=kfold) 
model3 = RandomForestClassifier (n_estimators =100, max_depth = 12, min_samples_leaf = 8, min_samples_split = 8, random_state =0) 
scores3 = cross_val_score(model3, df_x, df_y, cv=kfold) 
model4 = LGBMClassifier (random_state = 0, metric = 'binary_error',boosting_type = 'gbdt', learning_rate = 0.1, n_estimators = 100, num_leaves = 16, objective = 'binary') 
scores4 = cross_val_score(model4, df_x, df_y, cv=kfold) 

print("LGBMClassifier Acc: "+str(scores4.mean())) 
print("RandomForestClassifier Acc: "+str(scores3.mean())) 
print("Gradient Boosting Acc: "+str(scores9.mean()))
