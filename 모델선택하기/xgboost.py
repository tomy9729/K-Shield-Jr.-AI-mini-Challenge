import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, plot_importance

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV)_0.csv')
df_x = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']


model9 = XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.05, eval_metric = 'error', eta = 0.1, objective = 'binary:logistic',use_label_encoder=False)

kfold = KFold(n_splits=3, shuffle = True, random_state = 0) #10-fold cross validation 

scores9 = cross_val_score(model9, df_x, df_y, cv=kfold) 

print("Gradient Boosting Acc: "+str(scores9.mean()))
