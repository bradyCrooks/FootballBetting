# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:05:43 2019

Change scoring of model to logloss and introduce better metrics

Results - suggesting 4 seasons of data

@author: brady
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 19:17:24 2019

Same as version 3 but iterating the number of seasons from 1 to 5


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

import pandas as pd

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats

season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'

# Get data 

numSeasonDict = {}

for num_season in range(1, 6):
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
            
    # Validation      
    E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path)
    val_data = E1_1718.getFixAndStats()
    y_val = val_data.loc[:,"HomeWin"]
    X_val = val_data.loc[:,"Ht_Position":]
    
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
    
    numSeasonDict[num_season] = (low_score, valLogLoss, valAUC, gini)