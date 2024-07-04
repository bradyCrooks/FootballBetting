# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:20:56 2019

basic model exploration

1 seasons worth of data
only past season chars

Some naive feature selection
@author: brady
"""
# %% Setup
import sys
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append("C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\processing")

from get_season_stats import GetTeamStats

season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'

E1_1516 = GetTeamStats('E1', '1516', season_dir, eos_stats_path)

data = E1_1516.getFixAndStats()


# %% 
data.info()

data.columns

# %%

# TODO: Look for ways to reduce collinearity

# TODO: Look at ExtraTrees Classifier

# TODO: Look at RFE CV

X = data.loc[:, ['Ht_Position', 'Ht_Points', 'Ht_Win', 'Ht_Draw', 'Ht_Loss', 'Ht_Scored',
       'Ht_Conceded', 'Ht_Shots_For', 'Ht_Shots_Ag', 'Ht_Shots_Tg_For',
       'Ht_Shots_Tg_Ag', 'Ht_Fouls_For', 'Ht_Fouls_Ag', 'Ht_Bk_Pts',
       'Ht_Win_H', 'Ht_Draw_H', 'Ht_Loss_H', 'Ht_Scored_H', 'Ht_Conceded_H',
       'Ht_Shots_For_H', 'Ht_Shots_Ag_H', 'Ht_Shots_Tg_For_H',
       'Ht_Shots_Tg_Ag_H', 'Ht_Fouls_For_H', 'Ht_Fouls_Ag_H', 'Ht_Bk_Pts_H',
       'Ht_Promoted', 'Ht_Relegated', 'At_Position', 'At_Points', 'At_Win',
       'At_Draw', 'At_Loss', 'At_Scored', 'At_Conceded', 'At_Shots_For',
       'At_Shots_Ag', 'At_Shots_Tg_For', 'At_Shots_Tg_Ag', 'At_Fouls_For',
       'At_Fouls_Ag', 'At_Bk_Pts', 'At_Win_A', 'At_Draw_A', 'At_Loss_A',
       'At_Scored_A', 'At_Conceded_A', 'At_Shots_For_A', 'At_Shots_Ag_A',
       'At_Shots_Tg_For_A', 'At_Shots_Tg_Ag_A', 'At_Fouls_For_A',
       'At_Fouls_Ag_A', 'At_Bk_Pts_A', 'At_Promoted', 'At_Relegated']]

y = data.loc[:, "HomeWin"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

# %%
from sklearn.feature_selection import RFE

logReg = LogisticRegression()

rfe = RFE(logReg, 20)

# TODO: Find optimum number of features

#rfeFit = rfe.fit(X_train, y_train.values.ravel())

print(rfeFit.get_params)
print(rfeFit.support_)
print(rfeFit.ranking_)

temp = pd.Series(rfeFit.support_, index=X_train.columns)
features = temp[temp==True].index

print(features)

# %% 
X_featRedTrain = X_train[features]
X_featRedTest = X_test.loc[:, features]
logreg = LogisticRegression()
logreg.fit(X_featRedTrain, y_train)
y_pred = logreg.predict(X_featRedTest)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_featRedTest, y_test)))


#%%
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_featRedTrain)
result=logit_model.fit()
print(result.summary2())

# %% Significant features
# These should be reduced one after the other
# Techincally Ht_Shots_For_H isn't signficant but it's close enough here
sigFeat = ['Ht_Shots_Tg_For', 'Ht_Shots_For_H', 'At_Shots_Ag', 'At_Shots_Tg_Ag', 'At_Shots_Ag_A', 'At_Shots_Tg_Ag_A']

X_SigTrain = X_featRedTrain.loc[:,sigFeat]

logit_model2 = sm.Logit(y_train, X_SigTrain)
result2 = logit_model2.fit()
print(result2.summary2())

# Need to revisit this


# %% 
logreg = LogisticRegression()
logreg.fit(X_SigTrain, y_train)

# %%
X_SigTest = X_test.loc[:, sigFeat]

y_pred = logreg.predict(X_SigTest)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_SigTest, y_test)))
# Lose accuracy after selecting for significant features
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# %%
y_PredProb = logreg.predict_proba(X_SigTest)

comp = pd.DataFrame({'PredProb':y_PredProb[:,1], 'Actual':y_test, 'Pred': y_pred})

# %% 
E1_1617 = GetTeamStats('E1', '1617', season_dir, eos_stats_path)
val_data = E1_1617.getFixAndStats()
y_val = val_data.loc[:,"HomeWin"]
X_val = val_data.loc[:,features]

print('Accuracy of logistic regression classifier on val set: {:.2f}'.format(logreg.score(X_val, y_val)))
