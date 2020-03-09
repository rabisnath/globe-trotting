import pandas as pd
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import pickle

"""['staleness', 'away_team', 'home_team', 'index', 'away_3PM-A_50game_avg',
       'home_3PM-A_50game_avg', 'away_3PM-A_10game_avg',
       'home_3PM-A_10game_avg', 'away_3PM-A_5game_avg', 'home_3PM-A_5game_avg',
       'away_3PM-A_2game_avg', 'home_3PM-A_2game_avg', 'away_FGM-A_50game_avg',
       'home_FGM-A_50game_avg', 'away_FGM-A_10game_avg',
       'home_FGM-A_10game_avg', 'away_FGM-A_5game_avg', 'home_FGM-A_5game_avg',
       'away_FGM-A_2game_avg', 'home_FGM-A_2game_avg', 'away_Def_50game_avg',
       'home_Def_50game_avg', 'away_Def_10game_avg', 'home_Def_10game_avg',
       'away_Def_5game_avg', 'home_Def_5game_avg', 'away_Def_2game_avg',
       'home_Def_2game_avg', 'away_Off_50game_avg', 'home_Off_50game_avg',
       'away_Off_10game_avg', 'home_Off_10game_avg', 'away_Off_5game_avg',
       'home_Off_5game_avg', 'away_Off_2game_avg', 'home_Off_2game_avg',
       'home_minus_away']"""

example1 = [29189,189,42.345454545454544,39.058333333333344,42.37,38.06,42.52,41.88,42.85,39,46.73636363636364,45.46666666666667,47.08,44.92999999999999,46.08,45.24,52.4,46.65,26.45454545454545,25.83333333333333,26.4,25.6,27.8,24.4,26.5,21.5,10.181818181818182,9.916666666666666,9.9,9.9,11.0,10.6,11.0,9.0]
example2 = [29189,66,27.92,29.160000000000004,27.92,29.160000000000004,27.92,29.160000000000004,28.15,35.25,46.76000000000001,46,46.76000000000001,46.6,46.76000000000001,46.6,47.15000000000001,49.2,27.0,29.6,27.0,29.6,27.0,29.6,29.5,25.5,10.6,9.2,10.6,9.2,10.6,9.2,8.5,6.0]
#^copied from the same csv the model is trained on, with text based columns and the y column removed
examples = np.vstack([example1, example2])
ans = [155, 198]

def predict(model, x_test):
    print(x_test.shape)
    p = model.predict(x_test)
    print(p)
    return p


if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='model (pickle) to use')
    parser.add_argument('meta', type=str, help='pickle file w column names of the data the model was trained on')
    args = parser.parse_args()
    target = args.target

    #with open(args.target, 'rb') as f:
    #    model = keras.models.load_model(f)
    model = keras.models.load_model(args.target)
    with open(args.meta, 'rb') as f:
        cols = pickle.load(f)

    #print(type(model), cols)

    predict(model, examples)