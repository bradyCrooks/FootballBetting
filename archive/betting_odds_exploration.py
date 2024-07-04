# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:35:30 2019

Exploring betting odds

@author: brady
"""
# %%
import pandas as pd

season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results\\'

data = pd.read_csv(season_dir + "E1_1617.csv")
data = data.append(pd.read_csv(season_dir + "E1_1516.csv"))
data = data.append(pd.read_csv(season_dir + "E1_1415.csv"))

# %% columns

print (data.columns)

# %%
odds_columns = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD',
       'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA',
       'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
data = data.loc[:, ["FTR"] + odds_columns]

# %%
results = ['H','D', 'A']

for col in results:
    data[col] = data.FTR.apply(lambda x: int(x == col))



data = data[results + odds_columns]

# %% Convert odds to probabilities

for col in odds_columns:
    store_col = data[col].copy()
    data[col] = store_col.apply(lambda x: 1/x)

# %% Metrics
from sklearn.metrics import roc_auc_score, log_loss

score_dict = {}
data.dropna(inplace=True)
for col in odds_columns:
    score_dict[col] = {'log_loss': log_loss(data[col[-1]], data[col]),
                       'roc_auc_score': roc_auc_score(data[col[-1]], data[col])}
    


score_df = pd.DataFrame(score_dict)

# %% 

score_df_t = score_df.transpose()

score_df_t.reset_index(inplace=True)

score_df_t.columns = ['compPred', 'log_loss', 'roc_auc_score']

score_df_t['result'] = score_df_t.compPred.apply(lambda x: x[-1])
score_df_t['company'] = score_df_t.compPred.apply(lambda x: x[:-1])

df = score_df_t.iloc[:, 1:]

df = df[['company', 'result', 'log_loss', 'roc_auc_score']]

df['gini'] = df.roc_auc_score * 2 - 1

print(df.sort_values(['result', 'gini'], ascending=False))


# %%

