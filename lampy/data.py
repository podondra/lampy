import pandas as pd
from sklearn.model_selection import train_test_split


def read_2018_data(data_file):
    return pd.read_csv(data_file, sep=';', parse_dates=['starttimestamp']) \
            .set_index(['sid', 'starttimestamp']).sort_index()


def read_2019_data(data_file):
    return pd.read_csv(data_file, parse_dates=['starttimestamp']) \
            .set_index(['sid', 'starttimestamp']).sort_index()


def resample_data(df, period='15T'):
    """Resample with `period`, fill forward and drop missing values. Default
    period is 15 minutes."""
    return df.resample(period).pad().dropna()


def create_dataset(df, target='pm10'):
    return df.values[:-1], df[[target]].values[1:]


def train_validation_split(X, y):
    return train_test_split(X, y, shuffle=False)
