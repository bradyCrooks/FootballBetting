# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:52:35 2019

Object to hold current season's teams and previous seasons stats for those stats including promoted/relegated

@author: brady
"""

import pandas as pd 

class TeamStats:
    """Class to obtain unique list of league's teams for a particular season
    and get the past season's stats for that team.
    
    Args:
        league (str): E followed by number indicating how many levels below prem
        season (str): Season listed as a 4 digit number i.e. 2010/2011 -> 1011
        season_dir (str): Filepath of directory containing raw season data
        eos_stats_path (str): Filepath for end of season stats data
        
    Attrs:
        league (str) = Store league arg
        season (str) = Store season arg
        season_path (str) = Path to the data for the selected league and season
        eos_stats (pandas.DataFrame) = Read in eos_stats
        teams (pandas.Series) = Unique list of teams for chosen league/season
    
    """
    
    
    def __init__(self, league, season, season_dir, eos_stats_path):
        
        self.league = league
        self.season = season
        self.season_path = "{}\\{}_{}.csv".format(season_dir, league, season)
        self.eos_stats = pd.read_csv(eos_stats_path)
        
        self.teams = self._teams()
        self.prev_season = self._prevSeason()
        
        
    def _teams(self):
        """Internal method to get unique list of teams
        
        returns: 
            teams (numpy.ndarray): unique series of teams for selected league 
                                    and season
        """
        
        # Get series of home team column from season data
        homeTeams = pd.read_csv(self.season_path, usecols=["HomeTeam"], 
                                  squeeze=True)
        
        teams = homeTeams.sort_values().unique()
        
        return teams
    
    def _prevSeason(self):
        """Internal method to get previous season
        
        returns:
            prev_season (str): previous season
    
        """
        
        # Subtracting 101 from the number version of current season gives
        # past season.
        curr_season = int(self.season)
        prev_season = str(curr_season - 101)
    
        return prev_season
    
    def prevSeasonFullStats(self):
        """Get stats for team previous season including promotion/relegation
        
        returns:
            stats (pandas.DataFrame): Stats from previous season
        """
        
        # Get correct season
        stats = self.eos_stats[self.eos_stats.Season == int(self.prev_season)]
        
        
        # Get correct teams
        stats = stats[stats.Team.isin(self.teams)]
        
        stats['Promoted'] = stats.League.apply(lambda x: promoted(self.league, x))
        stats['Relegated'] = stats.League.apply(lambda x: relegated(self.league, x))
        
        return stats
    
    def prevSeasonStatsHorA(self, at_home=True):
        """Get previous season stats for a team specific to them being home
        or away
        
        params
        ------
            at_home : bool
                Default is true. Select whether home or away
        
        returns
        -------
            stats : pandas.DataFrame
                General stats and home/away specifc stats from previous season
        """
        # Get full stats block
        stats = self.prevSeasonFullStats()
        
        # Remove home/away specific columns by looking for suffix
        if at_home:
            cols = [col for col in stats.columns if col[-2:] != "_A"]
        else:
            cols = [col for col in stats.columns if col[-2:] != "_H"]            
                          
        stats = stats[cols]
        stats = stats.drop(labels=["League", "Season"], axis=1)
        
        # Add prefix
        if at_home:
            new_cols = ["Ht_{}".format(col) for col in stats.columns]
        else:
            new_cols = ["At_{}".format(col) for col in stats.columns]
            
        stats.columns = new_cols
        
        return stats
        

def promoted(league, prev_league):
    """Given two league codes indicate a promotion
    
    args:
        league (str): Current league code
        prev_league (str): Previous league code
        
    returns:
        prom (int): indicator of promotion
    """
    
    prom = 0
    league_num = int(league[1])
    prev_league_num = int(prev_league[1])
    
    if league_num == prev_league_num - 1:
        prom = 1

    return prom        


def relegated(league, prev_league):
    """Given two league codes indicate a relagation
    
    args:
        league (str): Current league code
        prev_league (str): Previous league code
        
    returns:
        prom (int): indicator of relagation
    """
    
    rel = 0
    league_num = int(league[1])
    prev_league_num = int(prev_league[1])
    
    if league_num == prev_league_num + 1:
        rel = 1

    return rel       

class GetTeamStats(TeamStats):
    """Class to stick stats to fixtures along with results for fixtures in 
    order to prepare for modelling. 
    
    Inherits TeamStats class for convenience
    
    Args:
        league (str): E followed by number indicating how many levels below prem
        season (str): Season listed as a 4 digit number i.e. 2010/2011 -> 1011
        season_dir (str): Filepath of directory containing raw season data
        eos_stats_path (str): Filepath for end of season stats data
        include_odds (bool): Flag to indicate whether to include odds with fixtures
        
    Attrs:
        league (str) = Store league arg
        season (str) = Store season arg
        season_path (str) = Path to the data for the selected league and season
        eos_stats (pandas.DataFrame) = Read in eos_stats
        teams (pandas.Series) = Unique list of teams for chosen league/season
        fixtures (pandas.DataFrame) : Dataframe to contain fixtures and results\
        homeStats (pandas.DataFrame) : Get stats for home team
        awayStats (pandas.DataFrame) : Get stats for away team
        basicInfo (list) : basic info to get about fixtures
        oddsCols (list): betting odds to get about fixtures if requested
        include_odds (bool): Flag to indicate whether to include odds with fixtures
    """
    
    def __init__(self, league, season, season_dir, eos_stats_path, include_odds = False):
        TeamStats.__init__(self, league, season, season_dir, eos_stats_path)
         
        self.include_odds = include_odds
        self.basicInfo = ["Date", "HomeTeam", "AwayTeam", "FTR"] 
        self.oddsCols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD',
       'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA',
       'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
        self.fixtures = self._getFixtures(_include_odds = include_odds)
        self.homeStats = self.prevSeasonStatsHorA(at_home=True)
        self.awayStats = self.prevSeasonStatsHorA(at_home=False)
    
    
    def _getFixtures(self, _include_odds = False):
        """Internal method to get fixtures for requested season along with 
        formatted results.
        
        Args:
            _include_odds: Argument passed when building object
        """
        
        cols = self.basicInfo 
        
        if _include_odds:
            cols = cols + self.oddsCols
        
        fixtures = pd.read_csv(self.season_path, usecols=cols)
        
        fixtures["HomeWin"] = fixtures.FTR.apply(lambda result: 
                                                            int(result == "H"))
        fixtures["AwayWin"] = fixtures.FTR.apply(lambda result: 
                                                            int(result == "A"))
        fixtures["Draw"] = fixtures.FTR.apply(lambda result: 
                                                            int(result == "D"))
        
        fixtures = fixtures.drop("FTR", axis=1)
        
        return fixtures
    
    def getFixAndStats(self):
        """Method to return fixtures with previous season stats for both home
        team and away team.
        
        returns
        -------
            fixAndStats : pandas.DataFrame
                Fixtures, results and previous season stats for selected season
                ready for modelling
        """
        # Attach Home stats
        fixAndStats = self.fixtures.merge(self.homeStats, left_on="HomeTeam",
                                          right_on="Ht_Team")
        
        fixAndStats.drop("Ht_Team", axis=1, inplace=True)
        # Attach Away stats
        fixAndStats = fixAndStats.merge(self.awayStats, left_on="AwayTeam",
                                        right_on="At_Team")
        
        fixAndStats.drop("At_Team", axis=1, inplace=True)
        
        return fixAndStats
    
    def getFixAndBetProbs(self):
        if not(self.include_odds):
            self.fixtures = self._getFixtures(True)
            
        return self.fixtures
    
def get_multi_seasons(last_season, num_seasons, season_dir, eos_stats_path, league = 'E1'):
    """Create dataset containing fixtures and previous seasons for multiple 
    seasons for the purpose of model building.
    
    args:
        last_season (int): 4 digit number for latest season in dataset to create
        num_seasons (int): number of seasons to pull together
        season_dir (str): Directory to pull season data from 
        eos_stats_path (str): path to end of seasons stats table
        league (str): league code
        
    return:
        data (pd.DataFrame): dataframe containing multiple seasons worth of data
    """
    
    
    season = last_season
    data = pd.DataFrame()
    for i in range(num_seasons):
        
        teamStats = GetTeamStats(league, season, season_dir, eos_stats_path)
        teamData = teamStats.getFixAndStats()
        
        if season == str(last_season):
            data = teamData.copy()
        
        else: 
            data = pd.concat([data,teamData], axis=0)
            
        season -= 101
        
    data.reset_index(drop=True)
    
    data['Date'] = pd.to_datetime(data.Date)
    data.sort_values(['Date', 'HomeTeam']).reset_index(drop=True)
    
    return data

        
if __name__ == "__main__":
    season_dir='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\results'
    eos_stats_path='C:\\Users\\brady\\Documents\\FootballBetting\\FootballBetting\\data\\eos_stats\\SeasonSummaries.csv'
    test_stats = TeamStats('E1', '1516', season_dir, eos_stats_path)
    
    for attr, value in test_stats.__dict__.items():
        if attr == 'eos_stats':
            print(attr, value.head())
        else:
            print(attr, value)
            
#    print(promoted('E0', 'E1'))
#    print(promoted('E1', 'E1'))
#    print(promoted('E2', 'E1'))
#    
#    print(relegated('E0', 'E1'))
#    print(relegated('E1', 'E1'))
#    print(relegated('E2', 'E1'))
#    
#    print(test_stats.prevSeasonFullStats())
    print(test_stats.prevSeasonStatsHorA())
    print(test_stats.prevSeasonStatsHorA(False))
        
    test_GetStats = GetTeamStats('E1', '1516', season_dir, eos_stats_path, include_odds = True)
   # print(test_GetStats.fixtures)
    print(test_GetStats.getFixAndStats().head())
    print(test_GetStats.getFixAndStats().columns)
   

