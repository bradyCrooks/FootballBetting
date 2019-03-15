# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:06:09 2019

@author: brady
"""

import pandas as pd

def GetSeasonSummary(league, season):
    file = league + '_' + season + ".csv"
    df = pd.read_csv(file)
    sel_cols = ["HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR"]
    df = df[sel_cols]
    
    df["HW"] = (df["FTR"] == 'H')
    df["HD"] = (df["FTR"] == 'D')
    df["HL"] = (df["FTR"] == 'A')
    df["AL"] = (df["FTR"] == 'H')
    df["AD"] = (df["FTR"] == 'D')
    df["AW"] = (df["FTR"] == 'A')
    df["HBP"] = df["HY"] * 10 + df["HR"] * 25 
    df["ABP"] = df["AY"] * 10 + df["AR"] * 25 
    
    home_teams = df.drop(["AwayTeam", 'FTR', 'HY', 'AY', 'AR', 'HR', 'AR', 'ABP', 'AL', 'AD', 'AW'], axis = 1).copy()
    home_teams.rename(columns={"HomeTeam": 'Team', 'HW': 'Win', 'HD':'Draw', 'HL': 'Loss', 'FTHG': 'Scored', 
                               'FTAG': 'Conceded','HS': 'Shots_For', 'HST': 'Shots_Tg_For', 'AS': 'Shots_Ag',
                               'AST':'Shots_Tg_Ag', 'HF': 'Fouls_For', 'AF': 'Fouls_Ag', 'HBP': 'Bk_Pts'}, inplace=True)
    home = home_teams.groupby('Team').sum()
    
    
    away_teams = df.drop(["HomeTeam", 'FTR', 'HY', 'AY', 'AR', 'HR', 'AR', 'HBP', 'HL', 'HD', 'HW'], axis = 1).copy()
    away_teams.rename(columns={"AwayTeam": 'Team', 'AW': 'Win', 'AD':'Draw', 'AL': 'Loss', 'FTAG': 'Scored', 
                               'FTHG': 'Conceded','AS': 'Shots_For', 'AST': 'Shots_Tg_For', 'HS': 'Shots_Ag',
                               'HST':'Shots_Tg_Ag', 'AF': 'Fouls_For', 'HF': 'Fouls_Ag', 'ABP': 'Bk_Pts'}, inplace=True)
    away = away_teams.groupby('Team').sum()
    
    results = home.merge(left_index=True, right_index=True, right=away, suffixes=('_H', '_A'))
    
    cols = list(results.columns)
    col_set = {col[:-2] for col in cols}
    
    for col in col_set:
        if col not in results.columns:
            results[col] = results[col + '_H'] + results[col + '_A']
    
    results["Points"] = results["Win"] * 3 + results["Draw"] * 1
    results[["Points"]].sort_values("Points", ascending=False)
    
    results.sort_values("Points", ascending=False, inplace=True)
    results['Position'] = range(1, len(results) + 1) 
    results['League'] = league
    results['Season'] = season
    results = results[['League', 'Season', 'Position', 
                       'Points', 'Win', 'Draw', 'Loss', 'Scored', 'Conceded', 'Shots_For', 'Shots_Ag', 
                       'Shots_Tg_For', 'Shots_Tg_Ag', 'Fouls_For', 'Fouls_Ag', 'Bk_Pts',
                       'Win_H', 'Draw_H', 'Loss_H', 'Scored_H', 'Conceded_H', 'Shots_For_H', 'Shots_Ag_H', 
                       'Shots_Tg_For_H', 'Shots_Tg_Ag_H', 'Fouls_For_H', 'Fouls_Ag_H', 'Bk_Pts_H',
                       'Win_A', 'Draw_A', 'Loss_A', 'Scored_A', 'Conceded_A', 'Shots_For_A', 'Shots_Ag_A', 
                       'Shots_Tg_For_A', 'Shots_Tg_Ag_A', 'Fouls_For_A', 'Fouls_Ag_A', 'Bk_Pts_A']]
    
    return results

seasons = ['1415', '1516', '1617', '1718']
summaries = {"E1_" + season: GetSeasonSummary("E1", season) for season in seasons}
df = pd.concat(summaries.values())
df.to_csv("SeasonSummaries.csv")