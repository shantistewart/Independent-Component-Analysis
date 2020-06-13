# This file is the main script to run the FastICA algorithm.


import numpy as np
import matplotlib.pyplot as plotter
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from sklearn.decomposition import FastICA
import simpleaudio as sa
from Code import ICA, plotting_functions as plotting


print("\n")

# audio file names:
file_x1 = 'music_x1.wav'
file_x2 = 'music_x2.wav'
file_s1 = "music_s1.wav"
file_s2 = "music_s2.wav"
# plot titles:
title_x = "Observed Data x"
title_s = "Estimated Source s"
title_X = "Observed Data X"
title_S = "Estimated Sources S"

# """
# play audio files:
print("Playing", file_x1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_x2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x2)
play_obj = wave_obj.play()
play_obj.wait_done()
# """

# load audio files:
sample_freq_1, x1 = read(file_x1)
sample_freq_2, x2 = read(file_x2)
num_samples = x1.shape[0]

# concatenate audio files into a 2D array:
X = np.zeros((2, num_samples))
X[0] = x1
X[1] = x2
num_sig = X.shape[0]

# plot raw audio signals:
plotting.plot_signals(X, sample_freq_1, title_x)
# create a scatter plot of raw audio signals:
plotting.scatter_plot_signals(X, title_X, 'x')


# --------------------ICA ALGORITHM--------------------

# number of iterations to run FastICA:
num_iters = 1000

# center data:
X_center = ICA.center(X)

# whiten data:
X_white, whiten_filter = ICA.whiten(X_center)
# print("Mean of whitened data:")
# print(np.mean(X_white, axis=1))
# print("Variance of whitened data:")
# print(np.var(X_white, axis=1))

# run FastICA algorithm:
V = ICA.fastICA(X_white, num_sources=num_sig, num_iters=num_iters)
print("\nFinal rotation matrix V: ")
print(V)

# recover source signals:
S = ICA.recover_sources(X_white, V, X, whiten_filter)
# print("\nSize of S: ", S.shape)
# plot estimated source signals:
plotting.plot_signals(S, sample_freq_1, title_s)
# create a scatter plot of estimated source signals:
plotting.scatter_plot_signals(S, title_S, 's')

"""
# sklearn's implementation to verify against:
transformer = FastICA(n_components=2, random_state=0)
S = transformer.fit_transform(X.T)
S = S.T
# print("\nSize of S: ", S.shape)
plotting.plot_signals(S, sample_freq_1, title_s)
plotting.scatter_plot_signals(S, title_s, 's')
"""

print("")


# --------------------PLAYING RESULTS--------------------

# convert source numpy arrays to .WAV files:
write(file_s1, sample_freq_1, S[0].astype(np.int16))
write(file_s2, sample_freq_2, S[1].astype(np.int16))

# """
# play audio files:
print("Playing", file_s1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_s2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s2)
play_obj = wave_obj.play()
play_obj.wait_done()
# """


# display plots:
plotter.show()

print("\n\nDone!\n")
