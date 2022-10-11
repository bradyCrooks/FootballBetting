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

def simulation(bets, odds, results):
    """Calculate profit over season and payout on a game basis
    
    All arrays passed must be of the same width
    """
    
    
    total_spent = bets.sum().sum()
    payout = bets * results * odds
    winnings = payout.sum().sum()
    
    return (winnings - total_spent, payout)

def bet_decider(preds, odds, margin=0):
    """
    

    Parameters
    ----------
    preds : TYPE
        DESCRIPTION.
    odds : TYPE
        DESCRIPTION.
    margin : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    
    odds_prob = 1 / odds
    return (preds > odds_prob + margin).astype(int)
    
    
    