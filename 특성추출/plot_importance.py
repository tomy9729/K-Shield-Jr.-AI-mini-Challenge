import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('./KSJR_Car_Hacking_D_training-1(DS_CV)_0.csv')
df_x = df[['Arbitration_ID','Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']]
df_y = df['Class']

model = XGBClassifier(eval_metric = 'error',use_label_encoder=False,max_depth=5, n_estimators=100, learning_rate=0.1)
model.fit(df_x, df_y,verbose=2)
fig, ax = plt.subplots()
plot_importance(model,ax=ax)
