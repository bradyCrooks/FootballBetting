# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:12:26 2021

@author: brady
"""
# %% Import
import sys
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats, get_multi_seasons


season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'

# %% get data
data = get_multi_seasons(1617, 3, season_dir, eos_stats_path)

X = data.loc[:,'Ht_Position':]
y = data.loc[:, 'HomeWin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

# %% model set 1
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=3500, 
                          reg_alpha = 0.3,
                          max_depth=3, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))

# 0.728
# 0.615

# %% Tuning n_estimators
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=3, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))

# 0.748
# 0.620
# Possible more to be squeezed out but its marginal 
# Even here it's not a huge amount better

# %% Tuning Max Depth
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# 0.786
# 0.632

# %% Tuning learning rate

model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.05,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# Learning rate of 0.05 is a bit better on train and same on test 
# Values less than 0.01 don't do anythin
# Values other than 0.05 or 0.01 are worse

# %% Tune subsample
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.05,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# Little affect from changing between 0.8 and 1

# %% Tune colsample_bytree
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.05,  
                          colsample_bytree = 0.8,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# 0.850
# 0.636

# %% Tune Gamma
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.05,  
                          colsample_bytree = 0.8,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# Gamma around 10 seems best

# %% Tuning Reg
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.05,  
                          colsample_bytree = 0.8,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=10000, 
                          reg_alpha = 0.3,
                          max_depth=5, 
                          gamma=10)

model.fit(X_train, y_train)

y_pred_train = model.predict_proba(X_train)
y_pred_test = model.predict_proba(X_test)
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))
# messing with reg doesn't get a whole lot

# %% Look at feature importance
from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(model)
pyplot.show()

# %%
# find best feature
imps = pd.DataFrame({"labels": list(X_train.columns), "imp": model.feature_importances_}).set_index("labels").sort_values("imp", ascending=False)


#%%

columns = [imps.index[0]]

model.fit(X_train[columns], y_train)
y_pred_train = model.predict_proba(X_train[columns])
y_pred_test = model.predict_proba(X_test[columns])
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))

train_roc = [roc_auc_score(y_train, y_pred_train[:,1])]
test_roc = [roc_auc_score(y_test, y_pred_test[:,1])]


corr = X_train.corr()
model_size = 1
for _ in imps.index[1:]:
    imps_tmp = imps.drop(index=columns)
    sum_corr = abs(corr.loc[columns, list(imps_tmp.index)]).sum().to_frame("corr")
    
    mrmr = imps_tmp.merge(sum_corr, left_index=True, right_index=True)
    mrmr["score"] = mrmr.imp / (mrmr["corr"] / len(columns))
    
    new_x = mrmr.sort_values("score", ascending=False).index[0]
    columns.append( new_x)
    
    model.fit(X_train[columns], y_train)
    y_pred_train = model.predict_proba(X_train[columns])
    y_pred_test = model.predict_proba(X_test[columns])
    train_roc.append(roc_auc_score(y_train, y_pred_train[:,1]))
    test_roc.append(roc_auc_score(y_test, y_pred_test[:,1]))
    
    
log = pd.DataFrame({"feature": columns, "train_roc": train_roc, "test_roc": test_roc})


# %% Retrain and test on unseen data
model = xgb.XGBClassifier(silent=False, 
                          scale_pos_weight=1,
                          learning_rate=0.01,  
                          colsample_bytree = 0.4,
                          subsample = 0.8,
                          objective='binary:logistic', 
                          n_estimators=2000, 
                          reg_alpha = 5,
                          max_depth=3, 
                          gamma=15)

select = columns[:39]

model.fit(X_train[select], y_train)

y_pred_train = model.predict_proba(X_train[select])
y_pred_test = model.predict_proba(X_test[select])
print(roc_auc_score(y_train, y_pred_train[:,1]))
print(roc_auc_score(y_test, y_pred_test[:,1]))

E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=False)
val_data = E1_1718.getFixAndStats()

y_val = val_data.loc[:,"HomeWin"]
X_val = val_data.loc[:,select]
y_new_pred = model.predict_proba(X_val)
print(roc_auc_score(y_val, y_new_pred[:,1]))

# %%
from sklearn.calibration import CalibratedClassifierCV

lr = CalibratedClassifierCV(method="sigmoid")
lr.fit(model.predict_proba(X_train[select])[:,1].reshape(-1,1), y_train)

y_cal_pred = lr.predict_proba(y_pred_test[:,1].reshape(-1,1))

# %%
import numpy as np
import matplotlib.pyplot as plt
rel_df = pd.DataFrame({"pred_bins": pd.cut(y_cal_pred[:,1], np.arange(0,11)/10, labels=np.arange(0,10)/10), "pred": y_cal_pred[:,1], "actual": y_test})
rel_df = rel_df.groupby("pred_bins").agg(np.mean)

plt.plot(rel_df.pred, rel_df.actual)
plt.plot(rel_df.index, rel_df.index)
plt.show()


# %%
E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=True)
val_data = E1_1718.getFixAndStats()
odds = val_data["B365H"]
result = val_data["HomeWin"]
x_val= val_data.loc[:, select]
raw_pred = model.predict_proba(x_val)
pred = lr.predict_proba(raw_pred[:,1].reshape(-1,1))[:,1]

# %%
import scoring

bets = scoring.bet_decider(pred, odds)
payout = scoring.simulation(bets, odds, result)
print(payout[0])

# %%
df = pd.DataFrame({"pred":pred,"bets": bets, "odds": odds, "payout":payout[1]})
    
    