import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)
