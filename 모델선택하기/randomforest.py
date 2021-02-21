import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
#from xgboost import XGBClassifier, plot_importance

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV).csv')
df_x = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']


model3 = RandomForestClassifier (n_estimators =10, max_depth = 12, min_samples_leaf = 8, min_samples_split = 8, random_state =0) 


kfold = KFold(n_splits=10, shuffle = True, random_state = 0) #10-fold cross validation 

scores3 = cross_val_score(model3, df_x, df_y, cv=kfold) 

print("RandomForestClassifier Acc: "+str(scores3.mean())) 
