import numpy as np
import pandas as pd


class cpunks10k:
    def __init__(self):
        self.punks_df = pd.read_pickle('../data/punks.pkl')
        self.labels = list(self.punks_df.dtypes[0:-1].index)

    def any_to_one(self, i):
        x = True
        if i['any']:
            x = False
        return int(x)

    def load_data(self, labels=None):
        if labels is None:
            labels = self.labels
        df = self.punks_df.copy()[self.labels + ['img']]
        df['any'] = df[self.labels].apply(np.any, axis=1)
        df['none'] = df.apply(lambda x: self.any_to_one(x), axis=1)
        df = df.drop(['any'], axis=1)
        X = df['img'].to_numpy()
        Y = df[self.labels + ['none']].to_numpy()
        X = np.array([row[0] for row in X])
        X_train = X[0:9000]
        Y_train = Y[0:9000]
        X_test = X[9000:10000]
        Y_test = Y[9000:10000]
        labels = self.labels + ['none']
        return (X_train, Y_train), (X_test, Y_test), (labels)
