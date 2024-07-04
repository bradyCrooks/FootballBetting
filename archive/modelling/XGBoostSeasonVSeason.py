# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:22:26 2021

Train on 3 seasons data 
Test on following season as this will be more typically of how the model is 
run

@author: brady
"""
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

X_train = data.loc[:,'Ht_Position':]
y_train_home = data.loc[:, 'HomeWin']
y_train_away = data.loc[:, 'AwayWin']

E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=True)
test_data = E1_1718.getFixAndStats()
X_test = test_data.loc[:,'Ht_Position':]
y_test_home = test_data.loc[:,'HomeWin']
y_test_away = test_data.loc[:,'AwayWin']

# %% Train home model

model_params = {'silent':False, 
                'scale_pos_weight':1,
                'learning_rate':0.01,  
                'colsample_bytree': 0.4,
                'subsample': 0.8,
                'eval_metric':"logloss",
                'objective':'binary:logistic', 
                'n_estimators':3500, 
                'reg_alpha': 0.3,
                'max_depth':3, 
                'gamma':10}

model_home = xgb.XGBClassifier(**model_params)

model_home.fit(X_train, y_train_home)

# Evaluation
y_h_train_pred = model_home.predict_proba(X_train)
y_h_test_pred = model_home.predict_proba(X_test)

print(roc_auc_score(y_train_home, y_h_train_pred[:,1]))
print(roc_auc_score(y_test_home, y_h_test_pred[:,1]))
print(log_loss(y_train_home, y_h_train_pred[:,1]))
print(log_loss(y_test_home, y_h_test_pred[:,1]))


# %% Selection
imps = pd.DataFrame({"labels": list(X_train.columns), "imp": model_home.feature_importances_}).set_index("labels").sort_values("imp", ascending=False)

columns = [imps.index[0]]

model_home.fit(X_train[columns], y_train_home)
y_pred_train = model_home.predict_proba(X_train[columns])
y_pred_test = model_home.predict_proba(X_test[columns])

train_roc = [roc_auc_score(y_train_home, y_pred_train[:,1])]
test_roc = [roc_auc_score(y_test_home, y_pred_test[:,1])]

train_ll = [log_loss(y_train_home, y_pred_train[:,1])]
test_ll = [log_loss(y_test_home, y_pred_test[:,1])]


corr = X_train.corr()
model_size = 1
for _ in imps.index[1:]:
    imps_tmp = imps.drop(index=columns)
    sum_corr = abs(corr.loc[columns, list(imps_tmp.index)]).sum().to_frame("corr")
    
    mrmr = imps_tmp.merge(sum_corr, left_index=True, right_index=True)
    mrmr["score"] = mrmr.imp / (mrmr["corr"] / len(columns))
    
    new_x = mrmr.sort_values("score", ascending=False).index[0]
    columns.append( new_x)
    
    model_home.fit(X_train[columns], y_train_home)
    y_pred_train = model_home.predict_proba(X_train[columns])
    y_pred_test = model_home.predict_proba(X_test[columns])
    train_roc.append(roc_auc_score(y_train_home, y_pred_train[:,1]))
    test_roc.append(roc_auc_score(y_test_home, y_pred_test[:,1]))
    train_ll.append(log_loss(y_train_home, y_pred_train[:,1]))
    test_ll.append(log_loss(y_test_home, y_pred_test[:,1]))
    
    
log = pd.DataFrame({"feature": columns, "train_roc": train_roc, "test_roc": test_roc, "train_ll": train_ll, "test_ll": test_ll})

# %% Find vars
co_index = log.sort_values(by="test_ll").index[0]
var_sel_home = list(log.feature.iloc[:co_index + 1])
model_home.fit(X_train[var_sel_home], y_train_home)

# %% Mrmr

def mrmr_selection(X_train, X_test, y_train, y_test, model):
    """Perform mrmr feature selection using xgboost's feature importance.    

    Parameters
    ----------
    X_train : pd.Series
    X_test : pd.Series
    y_train : pd.Series
    y_test : pd.Series
    model : xgboost.XGBClassifier
        Already fitted model object.

    Returns
    -------
    log : pd.Dataframe
        Log of improvement of performance with added features.

    """
    imps = pd.DataFrame({"labels": list(X_train.columns), "imp": model.feature_importances_}).set_index("labels").sort_values("imp", ascending=False)

    columns = [imps.index[0]]
    
    model.fit(X_train[columns], y_train)
    y_pred_train = model.predict_proba(X_train[columns])
    y_pred_test = model.predict_proba(X_test[columns])
    
    train_roc = [roc_auc_score(y_train, y_pred_train[:,1])]
    test_roc = [roc_auc_score(y_test, y_pred_test[:,1])]
    
    train_ll = [log_loss(y_train, y_pred_train[:,1])]
    test_ll = [log_loss(y_test, y_pred_test[:,1])]
    
    
    corr = X_train.corr()
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
        train_ll.append(log_loss(y_train, y_pred_train[:,1]))
        test_ll.append(log_loss(y_test, y_pred_test[:,1]))
        
        
    log = pd.DataFrame({"feature": columns, "train_roc": train_roc, "test_roc": test_roc, "train_ll": train_ll, "test_ll": test_ll})
    
    return log

def get_vars_mrmr_log(mrmr_log):
    """Get optimal point based on test performance for which features to use.

    Parameters
    ----------
    mrmr_log : pd.Dataframe
        Log produced by mrmr_selection function.

    Returns
    -------
    var_sel : list
        List of variables to use for model.

    """
    co_index = mrmr_log.sort_values(by="test_ll").index[0]
    var_sel = list(mrmr_log.feature.iloc[:co_index + 1])
    return var_sel

# %% Train away model

model_away = xgb.XGBClassifier(**model_params)

model_away.fit(X_train, y_train_away)

# Evaluation
y_h_train_pred = model_away.predict_proba(X_train)
y_h_test_pred = model_away.predict_proba(X_test)

print(roc_auc_score(y_train_away, y_h_train_pred[:,1]))
print(roc_auc_score(y_test_away, y_h_test_pred[:,1]))
print(log_loss(y_train_away, y_h_train_pred[:,1]))
print(log_loss(y_test_away, y_h_test_pred[:,1]))


# %% var selection
away_log = mrmr_selection(X_train, X_test, y_train_away, y_test_away, model_away)
# %%
var_sel_away = get_vars_mrmr_log(away_log)
model_away.fit(X_train[var_sel_away], y_train_home)

# %% Calibration
from sklearn.calibration import CalibratedClassifierCV


y_h_train_pred = model_home.predict_proba(X_train[var_sel_home])
y_a_train_pred = model_away.predict_proba(X_train[var_sel_away])

cal_home = CalibratedClassifierCV(method="sigmoid")
cal_home.fit(y_h_train_pred[:,1].reshape(-1,1), y_train_home)

cal_away = CalibratedClassifierCV(method="sigmoid")
cal_away.fit(y_a_train_pred[:,1].reshape(-1,1), y_train_away)

# %% 
y_d_train_pred = (y_h_train_pred + y_a_train_pred) * -1
y_train_draw = data.loc[:, 'Draw']
cal_draw = CalibratedClassifierCV(method="sigmoid")
cal_draw.fit(y_d_train_pred[:,1].reshape(-1,1), y_train_draw)


# %%
y_test_draw = test_data.loc[:,'Draw']
model_draw = xgb.XGBClassifier(**model_params)

model_draw.fit(X_train, y_train_draw)

# Evaluation
y_d_train_pred = model_draw.predict_proba(X_train)
y_d_test_pred = model_draw.predict_proba(X_test)

print(roc_auc_score(y_train_draw, y_d_train_pred[:,1]))
print(roc_auc_score(y_test_draw, y_d_test_pred[:,1]))
print(log_loss(y_train_draw, y_d_train_pred[:,1]))
print(log_loss(y_test_draw, y_d_test_pred[:,1]))


draw_log = mrmr_selection(X_train, X_test, y_train_draw, y_test_draw, model_draw)

var_sel_draw = get_vars_mrmr_log(draw_log)
model_draw.fit(X_train[var_sel_draw], y_train_draw)


# %%
y_d_train_pred = model_draw.predict_proba(X_train[var_sel_draw])

cal_draw = CalibratedClassifierCV(method="sigmoid")
cal_draw.fit(y_d_train_pred[:,1].reshape(-1,1), y_train_draw)

# %%
def pred_pipe(X, model, cal):
    pred_0 = model.predict_proba(X)[:,1].reshape(-1,1)
    return cal.predict_proba(pred_0)[:,1]

pred_h = pred_pipe(X_test[var_sel_home], model_home, cal_home)
pred_a = pred_pipe(X_test[var_sel_away], model_away, cal_away)
pred_d = pred_pipe(X_test[var_sel_draw], model_draw, cal_draw)


# %%
import scoring
import numpy as np
odds = test_data.loc[:,["B365H", "B365D", "B365A"]].values
results = test_data.loc[:,["HomeWin", "Draw", "AwayWin"]].values
preds = np.vstack((pred_h, pred_d, pred_a)).T
#%%

bets = scoring.bet_decider(preds, odds)

# %% 
payout = scoring.simulation(bets, odds, results)
print(payout[0])