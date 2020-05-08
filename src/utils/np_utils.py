from __future__ import absolute_import
import logging

import numpy as np


def unit_norm(A, axis=-1):
    A = np.array(A)
    norm = np.linalg.norm(A, axis=axis, keepdims=True)
    return A / np.maximum(norm, 1e-15)


def random_select(inputs, n_samples):
    return inputs[np.random.permutation(inputs.shape[0])[:n_samples]]


def shuffled_copy(inputs):
    return inputs[np.random.permutation(inputs.shape[0])]


def prob_dist(A, axis=-1):
    A = np.array(A, dtype=np.float32)
    assert all(A >= 0)
    return A / (A.sum(axis=axis, keepdims=True) + 1e-9)


