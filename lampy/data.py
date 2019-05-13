import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def resample_data(df, period='15T'):
    """Resample with `period`, fill forward and drop missing values. Default
    period is 15 minutes."""
    return df.resample(period).pad().dropna()


def create_dataset(df, target='pm10'):
    """Create dataset suitable for time series forecasting: based on previous
    value predict the next value."""
    return df.values[:-1], df[[target]].values[1:]


def train_validation_split(X, y):
    """Split data into train and validation set with the ration 0.5 (the same
    number of examples in both sets). Do not shuffle the date due to time
    dependency."""
    return train_test_split(X, y, test_size=0.5, shuffle=False)


def transform_dataset(X, y, window=2):
    """Transofrm dataset so the it is suitable for training of linear
    regression where `window` is the number of previous measurements used to
    predict the next value `y`."""
    n_rows, n_features = X.shape
    # prepare new design matrix
    new_X = np.zeros((n_rows, window * n_features))
    new_X[:, :n_features] = X
    for idx in range(1, window):
        start = n_features * idx
        end = n_features * (idx + 1)
        new_X[idx:, start:end] = X[:-idx]
    return new_X[(window - 1):], y[(window - 1):]
