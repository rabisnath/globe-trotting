import pandas as pd
import numpy as np
import argparse
import os

"""['away_team', 'home_team', 'league_name', 'gym_name', 'away_record',
       'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_final', 'home_record',
       'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_final', 'away_poss',
       'away_ORtg', 'away_DRtg', 'home_poss', 'home_ORtg', 'home_DRtg',
       'away_eFGper', 'away_TOper', 'away_ORper', 'away_FTR', 'home_eFGper',
       'home_TOper', 'home_ORper', 'home_FTR', 'away_FIC', 'away_Off',
       'away_Def', 'away_Reb', 'away_Ast', 'away_PF', 'away_STL', 'away_TO',
       'away_BLK', 'away_FGM-A', 'away_3PM-A', 'away_FTM-A', 'home_FIC',
       'home_Off', 'home_Def', 'home_Reb', 'home_Ast', 'home_PF', 'home_STL',
       'home_TO', 'home_BLK', 'home_FGM-A', 'home_3PM-A', 'home_FTM-A',
       'date_played', 'staleness', 'home_minus_away', 'total']
"""

"""
Given stat suffixes like "Off, Def, FGM-A, 3PM-A"
Select only relevant cols from dataset
For each team, make a df:
    Grab all rows w that team as away or home
    For each stat
        Init stat vector, iterate and fill (check if away or home for each row)
        Call helper to make moving average vectors
    Bundle all the stat+moving avg vectors into a df and export
    ^the dfs for each team can be stacked horizontally and exported together

To make output dataset:
Each row has: 
    index/date/basic info
    away and home team
    populate stats row by row from team dfs made above
        for row i, away_team_def_30game = team_stats_df[$away_team_name_def_30game]
"""

to_keep = ['staleness', 'home_minus_away', 'total', 'away_team', 'home_team', 'date_played', 'index'] # non-stat features to keep
stat_suffixes = ['Off', 'Def', 'FGM-A', '3PM-A'] # want to grab features like "home_Off" and "away_Off" by suffix

def get_full_feature_list(base, stat_suffixes):
    # return base + home_suffix + away_suffix for suffix in stat_suffixes
    out = []
    for s in stat_suffixes:
        out.append('away_'+s)
        out.append('home_'+s)
    return out+base

def get_moving_avg(l, n):
    # takes a list of numbers and a range, returns a list of equal length
    # for i < n, new_list[i] is the average of l[:i], 
    # for i >= n, new_list[i] is the avg of l[i-n:i]
    out = [0] * len(l)
    for i in range(len(l)):
        if i == 0:
            out[i] = l[i]
        elif i < n:
            out[i] = np.mean(l[:i])
        else:
            out[i] = np.mean(l[i-n:i])
    return out


def get_team_df(box_scores, stat_suffixes, team, avgs=[2,5,10,50]):
    # takes box score data, list of stats, 
    # returns a df with cols like, 
    #   team_Off, team_Off_5day, team_Off_30day, team_Def, team_Def_5day, etc
    data = box_scores[(box_scores['away_team'] == team) | (box_scores['home_team'] == team)]
    data = data.sort_values(by='index')
    #print(data.head())
    home_rows = data[data['home_team'] == team]
    away_rows = data[data['away_team'] == team]
    stat_dfs = []
    for s in stat_suffixes:
        # these dfs have two cols: index (for ordering) and the stat
        s_from_home_rows = home_rows[['index', 'home_'+s]].rename(columns={'index':'index', 'home_'+s: team+'_'+s})
        s_from_away_rows = away_rows[['index', 'away_'+s]].rename(columns={'index':'index', 'away_'+s: team+'_'+s})
        #print(s_from_away_rows.head())
        stat_df = pd.concat([s_from_home_rows, s_from_away_rows], axis=0).sort_values(by='index')
        #print(stat_df)
        stat_vector = stat_df[team+'_'+s]
        #print(stat_vector[:5])
        # todo: call moving avg, add result to stat_df, add stat df to list, return horizontal?concat of all
        for n in avgs:
            stat_df[team+'_'+s+'_'+str(n)+'game_avg'] = get_moving_avg(stat_vector.values, n)
        #print(stat_df.head())
        stat_dfs.append(stat_df)
    team_df = pd.concat(stat_dfs, axis=1)

    return team_df

def clean_teamname(s):
    # replaces spaces with _
    return s.replace(' ', '_').replace('/', '')

def fix_names(df):
    # runs clean_teamname on each away_team and home_team
    cache = {}
    names = list(df.away_team.unique()) + list(df.home_team.unique())
    for n in names:
        cache[n] = clean_teamname(n)
    #print(df['away_team'][:5], type(df['away_team']))
    df['away_team'] = df['away_team'].map(lambda n: cache[n])
    df['home_team'] = df['home_team'].map(lambda n: cache[n])
    return df


def get_unique_teams(box_scores):
    # takes box score data, returns list of unique team names
    return [clean_teamname(s) for s in box_scores.away_team.unique()]

def get_all_team_dfs(box_scores, stat_suffixes):
    # calls get_team_df for each team represented in box_scores
    # returns dict of results
    team_dfs = {}
    for team in get_unique_teams(box_scores):
        team_dfs[team] = get_team_df(box_scores, stat_suffixes, team)
    #print("\nin get_all_team_dfs, keys: ", team_dfs.keys())
    return team_dfs

def export_team_data(path, team_dfs):
    # takes dict of dfs and saves them as csv under path
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)
    for team, df in team_dfs.items():
        df.to_csv(path+team+'_data.csv')

def import_team_data(path):
    # takes a dir and reads all csvs into a dict
    files = os.listdir(path)
    csvs = [f for f in files if '.csv' in f]
    team_dfs = {}
    for f in csvs:
        team_dfs[f[:-9]] = pd.read_csv(path+f)
    return team_dfs
    

def make_training_dataset(box_scores, base, stat_suffixes, team_data):
    # takes box score dataset, returns new dataset with
    # cols in base copied over
    # data from team_data for each stat represented in stat_suffixes
    training_data = box_scores[base]
    #print(training_data.head())
    # sort df by home team, fill in home stat columns, do same for away ###
    training_data = training_data.sort_values('home_team')
    #print(training_data.head())
    #print(team_data['2B_Control_Trapani'].head())
    


    return training_data

def tests():
    #print(get_full_feature_list(to_keep, stat_suffixes))
    #l = np.arange(10)
    #print(l, get_moving_avg(l, 3))
    #team_df = get_team_df(df, stat_suffixes, "Perth")
    #print(team_df.head())
    #print(get_unique_teams(df))

    team_data = get_all_team_dfs(df, stat_suffixes)
    #print(team_data["Perth"].head()) 
    export_team_data('team_data/', team_data)
    #team_data = import_team_data('team_data/')
    #print(team_data['Perth'].head())
    training_data = make_training_dataset(df, to_keep, stat_suffixes, team_data)
    

    pass

if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='.csv file to process')
    parser.add_argument('out', type=str, help='output file name, if none provided, overwrites target')

    args = parser.parse_args()
    target = args.target
    out = target if not args.out else args.out

    # importing data
    df = pd.read_csv(target, index_col=0)
    df = df.sort_values(by='staleness')
    #df = df.reset_index(drop=True)
    df = df.reset_index()
    df = fix_names(df)
    print(df.head())
    tests()

    # editing df
    # fn calls go here
    df.to_csv(out)