import numpy as np
from sklearn import metrics


def rmse(y_true, y_pred):
    """Compute the root mean square error."""
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
