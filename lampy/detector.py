import numpy as np
import pandas as pd
from sklearn import linear_model

from lampy import data


DATA_FILE_2018 = 'data/636f6a2e-3a48-4801-afce-05c864adc60e-lampy_2019.csv'
DATA_FILE_2019 = 'data/2e6bbd16-8d1c-4efa-b0f0-06a74f40d8d3-lampysensory-2019-1pololeti.csv'
SID = 'y7e4onsytkb3ydonflz5kcbcigkh5ulo'


class Detector:
    def __init__(self, window=2):
        """Train the linear regression predictor and determin threshold."""
        data_2018 = pd.read_csv(DATA_FILE_2018, index_col='sid',
                                sep=';', parse_dates=['starttimestamp'])
        data_2019 = pd.read_csv(DATA_FILE_2019, index_col='sid',
                                parse_dates=['starttimestamp'])

        X_tr, y_tr = self.prepocess_data(data_2018, SID, window)
        X_te, y_te = self.prepocess_data(data_2019, SID, window)

        self.linear = linear_model.LinearRegression()
        self.linear.fit(X_tr, y_tr)

        y_pred = self.linear.predict(X_te)
        y_delta = np.abs(y_te - y_pred)

        _, bins = np.histogram(y_delta)
        self.threshold = bins[1]

    def prepocess_data(self, raw_data, sid, window):
        """Preprocess data to be suitable for training."""
        df_data = raw_data.loc[sid].set_index('starttimestamp')['pm10']\
                .sort_index()
        resampled_data = data.resample_data(df_data, period='15T')
        X = resampled_data.values[:-1].reshape(-1, 1)
        y = resampled_data.values[1:].reshape(-1, 1)
        X_trans, y_trans = data.transform_dataset(X, y, window)
        return X_trans, y_trans

    def detect(self, x, y_true):
        """Detect anomaly from history `x` and `y_true` is the true current
        value retrieved from API."""
        y_pred = self.linear.predict([x])[0, 0]
        y_delta = np.abs(y_true - y_pred)
        return y_delta > self.threshold, y_pred
