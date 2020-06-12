# This file contains several plotting functions for data visualization.


import numpy as np
import matplotlib.pyplot as plotter


def plot_signals(X, sample_freq):
    num_sig = X.shape[0]
    num_samples = X.shape[1]

    # sample points of raw signals:
    samples = np.arange(0, num_samples)
    # actual time points of raw signals:
    time = (1 / sample_freq) * samples

    # create and format subplot:
    fig, axes = plotter.subplots(num_sig, 1)
    plotter.subplots_adjust(hspace=1)

    for i in range(num_sig):
        axes[i].set_title('Observed Signal x{0}(t)'.format(i+1))
        axes[i].plot(time, X[i])
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('x{0}(t)'.format(i+1))


def scatter_plot_signals(X):
    # create scatter plot of first 2 features of original data:
    fig, axis = plotter.subplots()
    axis.scatter(X[0], X[1], marker=".")
    axis.set_title('Observed Data X')
    axis.set_xlabel('x1')
    axis.set_ylabel('x2')
