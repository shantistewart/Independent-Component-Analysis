# This file is the main script to run the FastICA algorithm.


import numpy as np
import scipy.io.wavfile
import simpleaudio as sa


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
# display sizes of data arrays:
print("Size of x1: ", end="")
print(x1.shape)
print("Size of x2: ", end="")
print(x2.shape)
# display sampling frequencies:
print("Sampling frequencies of audio files: {0}, {1}\n".format(sample_freq_1, sample_freq_2))
