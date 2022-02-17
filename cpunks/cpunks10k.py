import numpy as np
import pandas as pd
import pathlib


class cpunks10k:
    def __init__(self):
        module_root = pathlib.Path(__file__).parents[0]
        self.punks_df = pd.read_pickle(f"{module_root}/data/punks.pkl")

    def any_to_one(self, i):
        x = True
        if i['any']:
            x = False
        return int(x)

    def load_data(self, labels=None):
        '''todo: reasonably handle labels or punt to another function,
                 flexible insample/outsample scheme,
                 think through insample/outsample to mix and insure
                 reasonable distribution of types.
        '''
        if labels is None:
            labels = list(self.punks_df.dtypes[0:-1].index)
        df = self.punks_df.copy()[labels + ['img']]
        df['any'] = df[labels].apply(np.any, axis=1)
        df['none'] = df.apply(lambda x: self.any_to_one(x), axis=1)

        df = df.drop(['any'], axis=1)
        X = df['img'].to_numpy()
        Y = df[labels + ['none']].to_numpy()
        X = np.array([row[0] for row in X])
        X_train = X[0:9000]
        Y_train = Y[0:9000]
        X_test = X[9000:10000]
        Y_test = Y[9000:10000]
        return (X_train, Y_train), (X_test, Y_test), (labels)
