import numpy as np


def sensitivity(G, M):
    return np.linalg.solve(G.T @ M @ G, G.T @ M)


def asymptotic_variance(K, Omega):
    return K @ Omega @ K.T


def local_bias(K, tau):
    return K @ tau


def amse(K, Omega, tau):
    bias = local_bias(K, tau)
    V = asymptotic_variance(K, Omega)
    return np.sum(bias**2) + np.trace(V)