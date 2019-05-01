import matplotlib.pyplot as plt
import numpy as np


def plot_delta_hist(y_delta):
    ax = plt.axes()
    ax.hist(y_delta)
    ax.set_yscale('log')
    ax.set_ylabel('počet')
    ax.set_xlabel('$\mid\hat{y} - y\mid$')
    

def plot_anomalies(anomalies, y_trans, window=10):
    index = np.arange(anomalies.shape[0])
    fig, axs = plt.subplots(3, 3)
    for ax, anomaly in zip(axs.ravel(), index[anomalies.ravel()]):
        start, end = anomaly - window, anomaly + window + 1
        ax.plot(index[start:end], y_trans[start:end], '.--')
        ax.plot(index[anomaly], y_trans[anomaly], 'o', color='r')
        ax.set_xticks([])
    for ax in axs[:, 0]:
        ax.set_ylabel('PM$_{10}$')
    fig.tight_layout()