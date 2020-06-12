# This file contains functions to implement the fastICA algorithm.


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
#   X = whitened data
#       size: (num_sig, num_samples)
def whiten(X):
    num_sig = X.shape[0]

    # compute covariance matrix:
    covariance = np.cov(X)

    # compute eigendecomposition of covariance matrix:
    eigval, eigvec = linalg.eigh(covariance)
    E = np.real(eigvec)
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
    X_whiten = np.matmul(np.matmul(D_sqrt_inv, np.transpose(E)), X)

    return X_whiten
