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
file_x1 = 'microphone_1.wav'
file_x2 = 'microphone_2.wav'
file_s1 = "source_1.wav"
file_s2 = "source_2.wav"

"""
# play audio files:
print("Playing", file_x1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_x2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x2)
play_obj = wave_obj.play()
play_obj.wait_done()
"""

# load audio files:
sample_freq_1, x1 = read(file_x1)
sample_freq_2, x2 = read(file_x2)
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
X_center = ICA.center(X)
# plotting.plot_signals(X_center, sample_freq_1)

# whiten data:
X_whiten = ICA.whiten(X_center)
print("\nSize of X_whiten: ", X_whiten.shape)
# print("Mean of whitened data:")
# print(np.mean(X_whiten, axis=1))
# print("Variance of whitened data:")
# print(np.var(X_whiten, axis=1))
# plot whitened signals:
# plotting.plot_signals(X_whiten, sample_freq_1)
# create a scatter plot of whitened signals:
# plotting.scatter_plot_signals(X_whiten)

# run FastICA algorithm:
W = ICA.fastICA(X_whiten, num_sources=num_sig, num_iters=num_iters)
print("\nSize of W: ", W.shape)
print(W)

# recover source signals:
S = ICA.recover_sources(X_whiten, W)
print("\nSize of S: ", S.shape)
# plot estimated source signals:
plotting.plot_signals(S, sample_freq_1)
# create a scatter plot of estimated source signals:
plotting.scatter_plot_signals(S)

"""
# sklearn's implementation to verify against:
transformer = FastICA(n_components=2, random_state=0)
S = transformer.fit_transform(X.T)
S = S.T
print(S.shape)
plotting.plot_signals(S, sample_freq_1)
plotting.scatter_plot_signals(S)
"""

print("\n")


# --------------------PLAYING RESULTS--------------------

# convert source numpy arrays to .WAV files:
write(file_s1, sample_freq_1, S[0].astype(np.int16))
write(file_s2, sample_freq_2, S[1].astype(np.int16))

# play audio files:
print("Playing", file_s1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_s2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s2)
play_obj = wave_obj.play()
play_obj.wait_done()


# display plots:
# plotter.show()

print("\n\nDone!\n")
