# This file contains functions to implement the FastICA algorithm.


import numpy as np
import numpy.linalg as linalg


# Function description: centers data.
# Inputs:
#   X = (un-centered) data
#       size: (num_sig, num_samples)
# Outputs:
#   X = centered data
#       size: (num_sig, num_samples)
def center(X):
    # center data:
    X = X - np.mean(X, axis=1, keepdims=True)

    return X


# Function description: whitens data.
# Inputs:
#   X = (centered) data
#       size: (num_sig, num_samples)
# Outputs:
#   X_white = whitened data
#       size: (num_sig, num_samples)
#   whiten_filter = whitening filter = D_sqrt_inv * E.T
#       size: (num_sig, num_sig)
def whiten(X):
    num_sig = X.shape[0]

    # compute eigendecomposition of covariance matrix:
    eigval, eigvec = linalg.eigh(np.cov(X))
    eigval = np.flip(eigval)
    E = np.flip(eigvec, axis=1)
    E = np.real(E)

    # gather eigenvalues into a diagonal matrix:
    D = np.zeros((num_sig, num_sig))
    for i in range(num_sig):
        for j in range(num_sig):
            if i == j:
                D[i, j] = eigval[i]

    # compute D^(-1/2):
    D_sqrt_inv = np.sqrt(linalg.inv(D))

    # compute whitening filter:
    whiten_filter = np.matmul(D_sqrt_inv, E.T)

    # whiten data:
    X_white = np.matmul(whiten_filter, X)

    return X_white, whiten_filter


# Function description: implements the FastICA algorithm to determine the final rotation matrix of ICA.
# Inputs:
#   X = whitened data
#       size: (num_sig, num_samples)
#   num_sources = number of desired sources
#   num_iters = number of iterations to run algorithm
# Outputs:
#   V = final rotation matrix
#       size: (num_sig, num_sources)
def fastICA(X, num_sources=None, num_iters=100):
    # dimensions of X:
    num_sig = X.shape[0]
    num_samples = X.shape[1]

    # default number of sources to number of observed signals:
    if num_sources is None:
        num_sources = num_sig

    # unmixing matrix:
    V = np.zeros((num_sig, num_sources))

    for source in range(num_sources):
        # randomly initialize weight vector:
        v = np.random.randn(num_sig, 1)

        for _ in range(num_iters):
            v = (1 / num_samples) * np.matmul(X, np.tanh(np.matmul(v.T, X)).T) -\
                (1 / num_samples) * np.sum((1 - np.square(np.tanh(np.matmul(v.T, X))))) * v
            for k in range(source):
                v = v - np.dot(np.squeeze(v), V[:, k]) * np.reshape(V[:, k], (num_sig, 1))
            v = v / linalg.norm(v)

        # save w:
        V[:, source] = np.squeeze(v)

    return V


# Function description: recovers sources with unmixing matrix.
# Inputs:
#   X_white = whitened data
#       size: (num_sig, num_samples)
#   V = final rotation matrix
#       size: (num_sig, num_sig)
#   X = un-centered raw data
#       size: (num_sig, num_samples)
#   whiten_filter = whitening filter = D_sqrt_inv * E.T
#       size: (num_sig, num_sig)
# Outputs:
#   S = sources
#       size: (num_sources, num_samples)
def recover_sources(X_white, V, X, whiten_filter):
    # project whitened data onto independent components:
    S = np.matmul(V.T, X_white)

    # compute unmixing matrix:
    W = np.matmul(V, whiten_filter)

    # estimate the mean and standard deviation of the sources:
    S_mean = np.matmul(W, np.mean(X, axis=1, keepdims=True))
    S_std = np.matmul(V, np.std(X, axis=1, keepdims=True))

    # add the mean and standard deviation of the sources back in:
    S = S_std * S + S_mean

    return S
