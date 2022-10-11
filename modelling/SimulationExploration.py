# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 13:05:49 2021

Explore Building Simulation

@author: brady
"""

# %%
import sys
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats, get_multi_seasons
from scoring import EvalModel

season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'

# %%
E1_1718 = GetTeamStats('E1', '1718', season_dir, eos_stats_path, include_odds=True)
val_data = E1_1718.getFixAndStats().loc[:, ['B365H', 'B365D', 'B365A', 'HomeWin', 'AwayWin', 'Draw']]



# %%
import numpy as np
bets = (np.random.rand(val_data.shape[0], 3) >= (1/3.)).astype(int)

# %% 
total_spent = bets.sum().sum()
odds = val_data[["B365H", "B365D", "B365A"]].values
results = val_data[["HomeWin", "AwayWin", "Draw"]].values

# %%
# Odds listed can be considered as a multiply to bet placed when won
winnings = (bets * results * odds).sum().sum()

print(winnings - total_spent)



# %% 
def simulation(bets, odds, results):
    total_spent = bets.sum().sum()
    payout = bets * results * odds
    winnings = payout.sum().sum()
    
    return (winnings - total_spent, payout)


# %%
x = 1000000
total = 0
for i in range(x):
    bets = (np.random.rand(val_data.shape[0], 3) >= (1/3.)).astype(int)
    total += simulation(bets, odds, results)[0]
    
ave_winnings = total/x
print(ave_winnings)