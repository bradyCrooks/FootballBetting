# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 18:35:07 2019

Score and validate models

@author: brady
"""
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import pandas as pd
import numpy as np
import sys

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats

class EvalModel:
    def __init__(self, model, model_info, features, response, league, valSeason, season_dir, eos_stats_dir):
        self.model = model
        self.model_info = model_info
        
        self.features = features
        self.response = response
                        
        self.valSeasonData = self._loadValSeason(league, valSeason, season_dir, eos_stats_dir)
        self.predProbs = self._getPredProbs() 
        
    def _loadValSeason(self, league, valSeason, season_dir, eos_stats_dir):
        
        valSeasonObj = GetTeamStats(league, valSeason, season_dir, eos_stats_dir, include_odds=True)
        return valSeasonObj.getFixAndStats()
    
    def _getPredProbs(self):
        
        return self.model.predict_proba(self.valSeasonData[self.features])
    
    def scoreModel(self):
        return score_model(self.valSeasonData[self.response], self.predProbs, details = self.model_info)
    
    
        
        

def score_model(y_true, y_pred_prob, details=''):
    """Calculate log loss, roc auc score and gini for model
    
    args:
        y_true (series): True value of result 1 or 0
        y_pred_prob (pandas.array): 2D predicted probability of result
        
    return:
        scores (dictionary): dictionary containing scores
    """
    
    scores = {}
    scores["details"] = details
    scores["log_loss"] = log_loss(y_true, y_pred_prob)
    scores["brier_score_loss"] = brier_score_loss(y_true. y_pred_prob[:,1])
    scores["roc_auc_score"] = roc_auc_score(y_true, y_pred_prob[:,1])
 
   
    
    for key, value in scores.items():
        print("{}: {}".format(key,value))
        
    return scores

def bet_value(bet_place, success, odds):
    """Calculate value of bet
    
    args: 
        bet_place (bool): Whether bet was placed
        success (bool): Whether bet was a succeess
        odds (float): Multiplier for successful bet
    return:
        value (float): value of bet
    """
    
    value = 0
    if bet_place == True:
        if success == True:
            value = odds
    
        else:
            value = -1.0
    
    return value

def calc_bets_col(df, place_bet_col, bet_success_col, odds_col, val_col_name='bet_value'):
    """Function to take a dataframe and calculate the value of all bets placed in simulating bets.
    
    
    args:
        df (pandas.DataFrame): dataframe containing betting information
        place_bet_col (string): name of column indicating whether to place bet
        bet_success_col (string): name of column indicating whether bet is successful
        odds_col (string): name of column indictating odds for result
        val_col_name (string): name of column to return with bet values
        
    return:
        new_df (pandas.DataFrame): dataframe with new bet value column
    """
    new_df = df.copy()
    new_df[val_col_name] = new_df.apply(lambda row: bet_value(row[place_bet_col], row[bet_success_col], row[odds_col]), axis=1)
    
    return new_df


if __name__ == "__main__":
    y_true = pd.Series([1, 0])
    print(y_true)
    y_pred_prob = np.array([[0.4, 0.6],
                           [0.3, 0.7]])
    print(y_pred_prob)
    
    score_model(y_true, y_pred_prob)
    print(bet_value(1, 1, 3))
    print(bet_value(1, 0, 3))
    print(bet_value(0, 1, 3))
    print(bet_value(0, 0, 3))