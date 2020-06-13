# This file contains several plotting functions for data visualization.


import numpy as np
import matplotlib.pyplot as plotter


# Function description: plots signals in time domain.
# Inputs:
#   X = data
#       size: (num_sig, num_samples)
#   sample_freq = sampling frequency
#   title = title for plot
# Outputs: None
def plot_signals(X, sample_freq, title):
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
        axes[i].set_title(title + str(i+1) + '(t)')
        axes[i].plot(time, X[i])
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Signal Value'.format(i+1))


# Function description: creates a scatter plot of the data.
# Inputs:
#   X = data
#       size: (num_sig, num_samples)
#   title = title for plot
#   label = label for axes (either 'x' or 's')
# Outputs: None
def scatter_plot_signals(X, title, label):
    # create scatter plot of first 2 features of original data:
    fig, axis = plotter.subplots()
    axis.scatter(X[0], X[1], marker=".")
    axis.set_title(title)
    axis.set_xlabel(label + '1')
    axis.set_ylabel(label + '2')
