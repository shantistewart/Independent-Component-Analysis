# This file contains functions to implement the fastICA algorithm.


import numpy as np


# Function description: centers data.
# Inputs:
#   X = (un-centered) data
#       size: (num_sig, num_samples)
# Outputs:
#   X = (un-centered) data
#       size: (num_sig, num_samples)
def center(X):
    # center data:
    X = X - np.mean(X, axis=1, keepdims=True)

    return X
