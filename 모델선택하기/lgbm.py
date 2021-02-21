import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
#from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV).csv')
df_x = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']


model4 = LGBMClassifier (n_estimators =100) 


kfold = KFold(n_splits=10, shuffle = True, random_state = 0) #10-fold cross validation 

scores4 = cross_val_score(model4, df_x, df_y, cv=kfold) 

print("LGBMClassifier Acc: "+str(scores4.mean())) 
