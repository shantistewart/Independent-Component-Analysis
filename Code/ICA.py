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
def whiten(X):
    num_sig = X.shape[0]

    # compute covariance matrix:
    covariance = np.cov(X)

    # compute eigendecomposition of covariance matrix:
    eigval, eigvec = linalg.eigh(covariance)
    eigval = np.flip(eigval)
    E = np.flip(eigvec, axis=1)
    E = np.real(E)
    # print("\n")
    # print('Eigenvalues: ', eigval)
    # print("Eigenvectors:")
    # print(E)

    # gather eigenvalues into a diagonal matrix:
    D = np.zeros((num_sig, num_sig))
    for i in range(num_sig):
        for j in range(num_sig):
            if i == j:
                D[i, j] = eigval[i]
    # print("D:")
    # print(D)

    # compute D^(-1/2):
    D_sqrt_inv = np.sqrt(linalg.inv(D))
    # print("D^(-1/2):")
    # print(D_sqrt_inv)
    # print("\n")

    # whiten data:
    X_white = np.matmul(np.matmul(D_sqrt_inv, np.transpose(E)), X)

    return X_white


# Function description: implements the FastICA algorithm.
# Inputs:
#   X = whitened data
#       size: (num_sig, num_samples)
#   num_sources = number of desired sources
#   num_iters = number of iterations to run algorithm
# Outputs:
#   W = unmixing matrix
#       size: (num_sig, num_sources)
def fastICA(X, num_sources=None, num_iters=100):
    # dimensions of X:
    num_sig = X.shape[0]
    num_samples = X.shape[1]

    # default number of sources to number of observed signals:
    if num_sources is None:
        num_sources = num_sig

    # unmixing matrix:
    W = np.zeros((num_sig, num_sources))

    for source in range(num_sources):
        # randomly initialize weight vector:
        w = np.random.randn(num_sig, 1)

        for _ in range(num_iters):
            w = (1 / num_samples) * np.matmul(X, np.tanh(np.matmul(w.T, X)).T) -\
                (1 / num_samples) * np.sum((1 - np.square(np.tanh(np.matmul(w.T, X))))) * w
            for k in range(source):
                w = w - np.dot(np.squeeze(w), W[:, k]) * np.reshape(W[:, k], (num_sig, 1))
            w = w / linalg.norm(w)

        # save w:
        W[:, source] = np.squeeze(w)

    return W


# Function description: recovers sources with unmixing matrix.
# Inputs:
#   X_white = whitened data
#       size: (num_sig, num_samples)
#   W = unmixing matrix
#       size: (num_sig, num_sources)
#   X = un-centered raw data
#       size: (num_sig, num_samples)
# Outputs:
#   S = sources
#       size: (num_sources, num_samples)
def recover_sources(X_white, W, X):
    num_sources = W.shape[1]

    # project whitened data onto independent components:
    S = np.matmul(W.T, X_white)

    # estimate the mean and standard deviation of the sources:
    S_mean = np.matmul(W, np.mean(X[0:num_sources], axis=1, keepdims=True))
    S_std = np.matmul(W, np.std(X[0:num_sources], axis=1, keepdims=True))

    # add the mean and standard deviation of the sources back in:
    S = S_std * S + S_mean

    return S
