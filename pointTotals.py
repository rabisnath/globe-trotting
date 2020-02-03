import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import pickle

""" features:
['away_team', 'home_team', 'league_name', 'gym_name', 'away_record',
       'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_final', 'home_record',
       'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_final', 'away_poss',
       'away_ORtg', 'away_DRtg', 'home_poss', 'home_ORtg', 'home_DRtg',
       'away_eFGper', 'away_TOper', 'away_ORper', 'away_FTR', 'home_eFGper',
       'home_TOper', 'home_ORper', 'home_FTR', 'date_played', 'staleness',
       'home_minus_away', 'total']
"""

class Sample(object):
    # right now, just going to give the network everything
    # even though you wouldn't know info like the final score
    # if you were placing a bet before the game
    # to do: think about what info we can give the network at
    # the time of inference
    # eventually build a more sophisticated model that incoporates
    # team and player histories (teamSeason, playerSeason) also
    def __init__(self, name, data):
        self.name = name
        
        n_cols = len(data.columns)
        values = data.values
        self.X = values[:, 0:n_cols-1]
        self.Y = values[:, n_cols-1]
        self.xwidth = n_cols-2
        self.ywidth = 1

class Model(object):
    def __init__(self, sample, model):
        self.name = model

        # X = inputs, Y = outputs
        self.X = sample.X
        self.Y = sample.Y
        #print("X: ", self.X.shape, "\nY:", self.Y.shape)
        self.n_inputs = sample.xwidth
        self.n_targets = sample.ywidth

        # t -> training, v -> validation
        #self.tX, self.vX, self.tY, self.vY = train_test_split(self.X, self.Y, test_size=0.2)

    def dense_model(self):
        #self.inputs = Input(shape=(int(self.n_inputs),), name='input')
        #h = self.inputs
        #h = Dense(int(self.n_inputs), activation='relu')(h)
        #h = BatchNormalization()(h)
        #h = Dense(int(self.n_inputs)//2, activation='relu')(h)
        #h = BatchNormalization()(h)
        #h = Dense(int(self.n_inputs)//4, activation='tanh')(h)
        #h = BatchNormalization()(h)
        #self.outputs = Dense(self.n_targets, activation='softmax', name='output')(h)
        #self.model = Model(inputs=self.inputs, outputs=self.outputs)
        
        self.model = Sequential()
        self.model.add(Dense(int(self.n_inputs), kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal', activation='relu'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model
        
    def train(self, path=""):
        estimator = KerasRegressor(build_fn=self.dense_model, epochs=5, batch_size=5)
        kfold = KFold(n_splits=10)
        results = cross_val_score(estimator, self.X, self.Y, cv=kfold)
        print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

        pickle.dump(estimator, open(self.name, 'wb'))



if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='.csv file to use in training')
    args = parser.parse_args()
    target = args.target

    data = pd.read_csv(target, index_col=0)
    data = data._get_numeric_data()
    #print(len(data.columns))
    s = Sample(target[:-4], data)
    dense = Model(s, "Dense")
    dense.train()
