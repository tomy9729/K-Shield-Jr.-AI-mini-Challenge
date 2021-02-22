import pandas as pd
from xgboost import XGBClassifier, plot_importance 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import pprint 
from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=10, shuffle = True, random_state = 0) 
pp = pprint.PrettyPrinter (width=80, indent=4)

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV)_0.csv')
df_x = df[['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']

train_x, test_x , train_y , test_y = train_test_split(df_x,df_y,test_size=0.3,random_state=10)
print(train_x.shape, test_x.shape)


base_model = RandomForestClassifier(random_state =0)
base_model.fit(train_x,train_y) 
base_acc = base_model.score(test_x,test_y)

param_grid = {'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

grid_search = GridSearchCV(RandomForestClassifier(random_state =0), param_grid, cv=kfold) 
grid_search.fit(train_x,train_y) 
pp.pprint(grid_search.best_params_)

best_model = grid_search.best_estimator_ 
best_accuracy = best_model.score(test_x, test_y) 
print(best_accuracy) 
