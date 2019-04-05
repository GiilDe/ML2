
import numpy as np
import scipy as sp
import math


def isNaN(v):
    return v != v


def minus(val1, val2):
    if isNaN(val1) or isNaN(val2):
        return 0
    if isinstance(val1, str):
        assert isinstance(val2, str)
        if val1 != val2:
            return 1
        else:
            return 0
    else:
        return val1-val2


def euclidean_dist(u, v):
    dist = 0
    for val1, val2 in zip(u, v):
        dist += (minus(val1, val2))**2
    return dist


def relief(S: np.ndarray, threshold, times):
    m = S.shape[0]  #num of samples
    n = S.shape[1]  #num of features

    weights = np.zeros(n)
    for i in range(times):
        instance_index = np.random.randint(0, m)
        chosen = S[instance_index, :]

        same = [instance for instance in S if instance is not chosen and instance[0] == chosen[0]]
        same_dists = [euclidean_dist(chosen, instance) for instance in same]
        different = [instance for instance in S if instance[0] != chosen[0]]
        different_dists = [euclidean_dist(chosen, instance) for instance in different]

        closest_same = same[np.argmin(same_dists)]
        closest_different = different[np.argmin(different_dists)]

        for j in range(n):
            weights[j] += (minus(chosen[j], closest_different[j]))**2 - (minus(chosen[j], closest_same[j]))**2

    chosen_features = [index for index in range(n) if weights[index] > threshold]
    return chosen_features




