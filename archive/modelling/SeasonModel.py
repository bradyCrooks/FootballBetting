# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:17:05 2019

@author: brady
"""
import pandas as pd



class SeasonModel:
    def __init__(self, season):
        self.season = season
        self.seasonData = pd.read_csv("E1_%s.csv"%season)
        self.teams = self._getTeams()
    
    def _getTeams(self):
        return self.seasonData.HomeTeam.unique()

def seasonData_test():
    sm = SeasonModel('1516')
    assert sm.seasonData.Div.iloc[0] == 'E1'
    assert sm.seasonData.Data.iloc[0] == '07/08/15'
    print('Data looks ok')


def getTeams_test():
    sm = SeasonModel('1516')
    assert len(sm.teams) == 24
    print (sm.teams)


if __name__ == '__main__':
    getTeams_test()