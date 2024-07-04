# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:41:50 2019

Introduce validation versus actual odds

@author: brady
"""

# %% set up imports
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss
import statsmodels.api as sm
from numpy import random
import pandas as pd

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats

season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'

# Get data 

numSeasonDict = {}

num_season = 4
print("Number of seasons: {}".format(num_season))
seasons = []
start_season = 1617
season = start_season

for i in range(num_season):
    seasons.append(str(season))
    season -= 101


for season in seasons:
    teamStats = GetTeamStats('E1', season, season_dir, eos_stats_path)
    teamData = teamStats.getFixAndStats()
    
    if season == str(start_season):
        data = teamData.copy()
    
    else: 
        data = pd.concat([data,teamData], axis=0)
    
data.reset_index(drop=True)

data['Date'] = pd.to_datetime(data.Date)
data.sort_values(['Date', 'HomeTeam']).reset_index(drop=True)

#Get Samples

X = data.loc[:,'Ht_Position':]
y = data.loc[:, 'HomeWin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

# Recursively go through RFE to get best number of features and best features

logReg = LogisticRegression()
idealNumFeat = 0
scoreDict = {}
low_score = 10000000000

# This is probably refinable
for numFeat in range(1, len(X_train.columns) + 1):
    rfe = RFE(logReg, numFeat)
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    logReg.fit(X_train_rfe,y_train)
    y_prob = logReg.predict_proba(X_test_rfe)
    
    score = log_loss(y_test, y_prob)
    
    scoreDict[numFeat] = score
    
    if score < low_score:
        low_score = score
        idealNumFeat = numFeat


print("Best score on test set: {}".format(low_score))
print("Number features obtained with: {}".format(idealNumFeat))    


# %%
# Validation      
E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=True)
val_data = E1_1718.getFixAndStats()

# %%
y_val = val_data.loc[:,"HomeWin"]
X_val = val_data.loc[:,"Ht_Position":]
odds = val_data.loc[:,"PSH"]

rfe = RFE(logReg, idealNumFeat)
X_train_rfe = rfe.fit_transform(X_train,y_train)
X_val_rfe = rfe.transform(X_val)
logReg.fit(X_train_rfe,y_train)
y_prob = logReg.predict_proba(X_val_rfe)
valLogLoss = log_loss(y_val, y_prob) 
y_prob_1 = y_prob[:,1]
valAUC = roc_auc_score(y_val, y_prob_1)
gini = valAUC * 2 - 1

print("Log Loss score: {}".format(valLogLoss))
print("Area under roc curve: {}".format(valAUC))
print("Gini Coeff: {}".format(gini))

# %%

sim_betting = val_data.loc[:,["HomeWin", "PSH"]]
sim_betting['bet_prob'] = sim_betting.PSH.apply(lambda x: 1 / x)
sim_betting['pred_prob'] = y_prob_1
sim_betting['place_bet'] = (sim_betting.pred_prob > sim_betting.bet_prob) 
sim_betting['bet_good'] = (sim_betting.place_bet == sim_betting.HomeWin)

# %%
def bet_value(bet_place, success, odds):
    value = 0
    if bet_place == True:
        if success == True:
            value = odds
    
        else:
            value = -1
    
    return value
# %%

sim_betting['bet_value'] = sim_betting.apply(lambda row: bet_value(row['place_bet'], row.bet_good, row.PSH), axis=1)

# %%
rand_value = 0
for i in range(1000):
    sim_betting['rand_bet'] = random.randint(0, 2, 552)
    sim_betting['rand_bet_good'] = (sim_betting.rand_bet == sim_betting.HomeWin)
    sim_betting['rand_bet_value'] = sim_betting.apply(lambda row: bet_value(row['rand_bet'], row.rand_bet_good, row.PSH), axis=1)
    rand_value += sim_betting.rand_bet_value.sum()

rand_value = rand_value / 1000
# %% 
print(sim_betting['bet_value'].sum())
print(rand_value)

# %% follow the odds 

sim_betting['odds_place_bet'] = (sim_betting.bet_prob > 0.5)
sim_betting['odds_success'] = (sim_betting.odds_place_bet == sim_betting.HomeWin)
sim_betting['odds_value'] = sim_betting.apply(lambda row: bet_value(row['odds_place_bet'], row['odds_success'], row.PSH), axis=1)


# %% 

sim_betting['odds_value'].sum()

# %% try different gaps

gaps = range(0, 101)
gaps = [float(gap)/ 100 for gap in gaps]

gap_scores = {}
for gap in gaps:
    sim_betting['place_bet_g'] = (sim_betting.pred_prob - gap > sim_betting.bet_prob)
    sim_betting['bet_good_g'] = (sim_betting.place_bet_g == sim_betting.HomeWin)
    sim_betting['g_bet_value'] = sim_betting.apply(lambda row: bet_value(row.place_bet_g, row.bet_good_g, row.PSH), axis=1)
    
    gap_scores[gap] = sim_betting['g_bet_value'].sum()
    print('{}: {}'.format(gap, gap_scores[gap]))


# %% bet everything
sim_betting['et_bet'] = 1
sim_betting['et_value'] = sim_betting.apply(lambda row: bet_value(row.et_bet, row.HomeWin, row.PSH), axis=1)
    