import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import argparse
import pickle

class Sample(object):
    def __init__(self, name, data):
        self.name = name
        
        n_cols = len(data.columns)
        values = data.values
        self.X = values[:, 0:n_cols-1]
        self.Y = values[:, n_cols-1]
        self.xwidth = n_cols-1
        self.ywidth = 1

class Estimator(object):
    def __init__(self, sample, model_name):
        self.name = model_name

        # X = inputs, Y = outputs
        self.X = sample.X
        self.Y = sample.Y
        #print(self.X[0])
        #print(self.Y[0])
        #print("X: ", self.X.shape, "\nY:", self.Y.shape)
        self.n_inputs = sample.xwidth
        self.n_targets = sample.ywidth

        # t -> training, v -> validation
        self.tX, self.vX, self.tY, self.vY = train_test_split(self.X, self.Y, test_size=0.2)

        if "Dense" in self.name:       
            # self.model = Sequential()
            # self.model.add(Dense(int(self.n_inputs), kernel_initializer='normal', activation='relu'))
            # self.model.add(Dense(20, kernel_initializer='normal', activation='relu'))
            # self.model.add(Dense(10, kernel_initializer='normal', activation='relu'))
            # self.model.add(Dense(1, kernel_initializer='normal', activation='relu'))
            # self.model.compile(loss='mean_squared_error', optimizer='adam')

            #print("n_inputs: ", self.n_inputs)
            #self.inputs = [Input(shape=(self.tX[i].shape[1], self.tX[i].shape[2]), name='input_'+str(i)) for i in range(len(self.tX))]
            #inputs = Input(shape=(int(self.n_inputs),), name='input')
            #self.inputs = Input(shape=(self.tX.shape[0], self.tX.shape[1]), name='input')
            self.inputs = Input(shape=(self.tX.shape[1],))
            h = self.inputs
            h = Dense(32, activation='relu', name='dense_1')(h)
            h = BatchNormalization(momentum=0.6, name='bnorm_12')(h)
            h = Dense(32, activation='relu', name='dense_2')(h)
            h = BatchNormalization(momentum=0.6, name='bnorm_23')(h)
            h = Dense(16, activation='relu', name='dense_3')(h)
            h = Dense(8, activation='relu', name='dense_4')(h)
            self.outputs = Dense(1, activation='relu', name='prediction')(h)
            self.model = Model(inputs=self.inputs, outputs=self.outputs, name=self.name)
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.summary()
        
        
    def train(self, path="", name=""):

        history = self.model.fit(self.tX, self.tY, batch_size=100, epochs=100, shuffle=True,
            validation_data=(self.vX, self.vY),
            callbacks=[ModelCheckpoint(path+name+'_'+self.name+'.h5', monitor='val_loss', save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=15, mode='auto')])

        #self.model.save(path+name+'_from_Method_'+self.name+'.h5')

        # verifying that we get the same predictions here as we do in infer.py
        #example1 = [29189,189,42.345454545454544,39.058333333333344,42.37,38.06,42.52,41.88,42.85,39.05,46.73636363636364,45.46666666666667,47.08,44.92999999999999,46.08,45.24,52.4,46.65,26.45454545454545,25.83333333333333,26.4,25.6,27.8,24.4,26.5,21.5,10.181818181818182,9.916666666666666,9.9,9.9,11.0,10.6,11.0,9.0]
        #example2 = [29189,66,27.92,29.160000000000004,27.92,29.160000000000004,27.92,29.160000000000004,28.15,35.25,46.76000000000001,46.6,46.76000000000001,46.6,46.76000000000001,46.6,47.15000000000001,49.2,27.0,29.6,27.0,29.6,27.0,29.6,29.5,25.5,10.6,9.2,10.6,9.2,10.6,9.2,8.5,6.0]
        #examples = np.vstack([example1, example2])
        #print("predictions", model.predict(examples))

        #self.model.reset_metrics()
        #predictions = self.model.predict(self.vX)



if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='.csv file to use in training')
    parser.add_argument('training_name', type=str, help='what to name the saved model')
    parser.add_argument('--meta', action='store_true', help='enable to save list of columnnames in input dataset (to give to infer.py)')
    parser.add_argument('--dry', action='store_true', help='enable to run the script without actually training')
    args = parser.parse_args()
    target = args.target

    data = pd.read_csv(target, index_col=0)
    cols = data.columns
    #print("COLS: ", type(cols), '\n', cols)
    data = data._get_numeric_data()
    #print(len(data.columns))

    model_dir = 'models/'
    s = Sample(target[:-4], data)
    dense = Estimator(s, "Dense")
    if not args.dry:
        dense.train(path=model_dir, name=args.training_name)
    
    if args.meta:
        with open(model_dir+args.training_name+'_columns', 'wb') as f:
            pickle.dump(cols, f)
