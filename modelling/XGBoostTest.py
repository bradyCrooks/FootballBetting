# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:10:51 2019

.58 is score to beat on validation
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

# %% define model


params = {"objective":"binary:logistic",'colsample_bytree': 0.4,'learning_rate': 0.01,
                'max_depth': 5, 'alpha': 10}


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

y_pred = model.predict_proba(X_test)

print(roc_auc_score(y_test, y_pred[:,1]))
                            
# %% Test validated score

E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=False)
val_data = E1_1718.getFixAndStats()


y_val = val_data.loc[:,"HomeWin"]
X_val = val_data.loc[:,"Ht_Position":]
y_new_pred = model.predict_proba(X_val)
print(roc_auc_score(y_val, y_new_pred[:,1]))

# %% Reliability plot
import numpy as np
rel_df = pd.DataFrame({"pred_bins": pd.cut(y_pred[:,1], np.arange(0,11)/10, labels=np.arange(0,10)/10), "pred": y_pred[:,1], "actual": y_test})
rel_df = rel_df.groupby("pred_bins").agg(np.mean)

# %% 
import matplotlib.pyplot as plt
plt.plot(rel_df.pred, rel_df.actual)
plt.plot(rel_df.index, rel_df.index)
plt.show()

# %% Platt scaling for calibration
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression()
# lr.fit(model.predict_proba(X_train)[:,1].reshape(-1,1), y_train)

# y_cal_pred = lr.predict_proba(y_pred[:,1].reshape(-1,1))


# %%
from sklearn.calibration import CalibratedClassifierCV

lr = CalibratedClassifierCV(method="sigmoid")
lr.fit(model.predict_proba(X_train)[:,1].reshape(-1,1), y_train)

y_cal_pred = lr.predict_proba(y_pred[:,1].reshape(-1,1))

# %%
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
x_val= val_data.loc[:, "Ht_Position":]
raw_pred = model.predict_proba(x_val)
pred = lr.predict_proba(raw_pred[:,1].reshape(-1,1))[:,1]

# %%
import scoring

bets = scoring.bet_decider(pred, odds)
payout = scoring.simulation(bets, odds, result)
print(payout[0])

# %%
df = pd.DataFrame({"pred":pred,"bets": bets, "odds": odds, "payout":payout[1]})


