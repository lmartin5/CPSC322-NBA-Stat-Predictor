"""CPSC 322 Final Project: NBA Team Success Predictor
@author L. Martin
@author E. Johnson
@date April 18, 2022

data_prep.py
Description:
    This python file contains the functions
    used to clean the player and team data and prepare
    it for classification tasks in the success predictor
    project.
"""

import os
from mysklearn.mypytable import MyPyTable

"""Globals (Change as data is added or removed)
    first_season(int): earliest season we have data for
        (i.e. 91 for 1990-1991 season)
    last_season(int): most recent season we have data for
        (i.e. 91 for 1990-1991 season)
    season_anomolies(dict of int, int pairs): keeps track of 
        seasons without the standard 82 games
        Note: in 2019-2020, non-bubble teams played around 63 games, 
            while bubble teams played 75
"""
first_season = 91
last_season = 21
season_anomolies = {21: 72, 20: 75, 12: 66, 99: 50}


def get_season_strings():
    """TODO
    """
    seasons = list(range(first_season, 100)) + list(range(0, last_season + 1))
    season_strings = []

    for season in seasons:
        begin = (season - 1) % 100
        if begin < 10:
            begin = "0" + str(begin)
        else:
            begin = str(begin)
        if season < 10:
            end = "0" + str(season)
        else:
            end = str(season)
        season_strings.append(begin + "_" + end)

    return season_strings


def get_team_data():
    """TODO
    """
    seasons = get_season_strings()
    data = []
    for season in seasons:
        file_loc = "teams_" + season + ".csv"
        file_loc = os.path.join("input_data", "team_stats", file_loc)
        season_data = MyPyTable().load_from_file(file_loc)
        data += season_data.data
    team_data = MyPyTable(season_data.column_names, data)
    return team_data


def get_player_data():
    """TODO
    """






def main():
    """Used to test validity of functions and to 
    call functions that store data in .csv files
    """
    team = get_team_data()
    print(team)


if __name__ == "__main__":
    main()