import pandas as pd

'''
Reading all the data from the CSVs into a dict of dfs
'''

csv_names = ['leagues', 'players', 'playerSeason', 'teams', 'teamSeason', 'teamSeasonLeague']
dfs = {}
for n in csv_names:
    dfs[n] = pd.read_csv(n+'.csv')




for n in csv_names:
    print("\n{}".format(n))
    print(dfs[n].columns)