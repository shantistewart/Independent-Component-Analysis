# This file is the main script to run the FastICA algorithm.


import numpy as np
import matplotlib.pyplot as plotter
import scipy.io.wavfile
from Code import ICA, plotting_functions as plotting

print("\n")

# audio file names:
file_1 = 'microphone_1.wav'
file_2 = 'microphone_2.wav'

"""
# play audio files:
wave_obj = sa.WaveObject.from_wave_file(file_1)
play_obj = wave_obj.play()
play_obj.wait_done()
wave_obj = sa.WaveObject.from_wave_file(file_2)
play_obj = wave_obj.play()
play_obj.wait_done()
"""

# load audio files:
sample_freq_1, x1 = scipy.io.wavfile.read(file_1)
sample_freq_2, x2 = scipy.io.wavfile.read(file_2)
num_samples = x1.shape[0]
# print("Size of x1: ", x1.shape)
# print("Size of x2: ", x2.shape)
# print("Sampling frequencies of audio files: {0}, {1}\n".format(sample_freq_1, sample_freq_2))

# concatenate audio files into a 2D array:
X = np.zeros((2, num_samples))
X[0] = x1
X[1] = x2
num_sig = X.shape[0]
print("\nSize of X: ", X.shape)

# plot raw audio signals:
plotting.plot_signals(X, sample_freq_1)
# create a scatter plot of raw audio signals:
plotting.scatter_plot_signals(X)


# --------------------ICA ALGORITHM--------------------

# number of iterations to run FastICA:
num_iters = 100

# center data:
X = ICA.center(X)
# plotting.plot_signals(X, sample_freq_1)

# whiten data:
X_whiten = ICA.whiten(X)
print("\nSize of X_whiten: ", X_whiten.shape)
# print("Mean of whitened data:")
# print(np.mean(X_whiten, axis=1))
# print("Variance of whitened data:")
# print(np.var(X_whiten, axis=1))
# print()
# plot whitened signals:
plotting.plot_signals(X_whiten, sample_freq_1)
# create a scatter plot of whitened signals:
plotting.scatter_plot_signals(X_whiten)

# run FastICA algorithm:
W = ICA.fastICA(X, num_sources=num_sig, num_iters=num_iters)
print("\nSize of W: ", W.shape)
print(W)


# plotter.show()

print("\n\nDone!\n")
