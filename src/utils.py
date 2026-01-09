#Utils.py
import numpy as np

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(labels, predictions):
        return mean_squared_error(labels, predictions, squared=False)

#Calculate RMSE
def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))
