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

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV).csv')
df_x = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']

model1 = LogisticRegression() 
model2 = KNeighborsClassifier(n_neighbors = 6) 
model3 = RandomForestClassifier (n_estimators =10) 
model4 = svm.SVC (kernel ='linear') 
model5 = svm.SVC (kernel ='poly') 
model6 = svm.SVC (kernel ='rbf') 
model7 = svm.SVC (kernel ='sigmoid') 
model8 = GaussianNB() 
model9 = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
model10 = LGBMClassifier()

kfold = KFold(n_splits=2, shuffle = True, random_state = 0) #10-fold cross validation 
scores1 = cross_val_score(model1, df_x, df_y, cv=kfold) 
scores2 = cross_val_score(model2, df_x, df_y, cv=kfold) 
scores3 = cross_val_score(model3, df_x, df_y, cv=kfold) 
scores4 = cross_val_score(model4, df_x, df_y, cv=kfold) 
scores5 = cross_val_score(model5, df_x, df_y, cv=kfold) 
scores6 = cross_val_score(model6, df_x, df_y, cv=kfold)
scores7 = cross_val_score(model7, df_x, df_y, cv=kfold) 
scores8 = cross_val_score(model8, df_x, df_y, cv=kfold) 
scores9 = cross_val_score(model9, df_x, df_y, cv=kfold) 
scores10 = cross_val_score(model10, df_x, df_y, cv=kfold) 

print("LogisticRefression Acc: "+str(scores1.mean())) 
print("KNeighborsClassifier Acc: "+str(scores2.mean())) 
print("RandomForestClassifier Acc: "+str(scores3.mean())) 
print("SVC_linear Acc: "+str(scores4.mean())) 
print("SVC_poly Acc: "+str(scores5.mean())) 
print("SVC_rbf Acc: "+str(scores6.mean())) 
print("SVC_sigmoid Acc: "+str(scores7.mean())) 
print("GaussianNB Acc: "+str(scores8.mean())) 
print("Gradient Boosting Acc: "+str(scores9.mean()))
print("LGBM Acc: "+str(scores10.mean()))
