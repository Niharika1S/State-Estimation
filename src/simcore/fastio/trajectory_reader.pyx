# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp


def read_trajectory(values):
    cdef cnp.ndarray arr = np.asarray(values, dtype=np.float64)
    return arr


def parse_csv_rows(path):
    arr = np.loadtxt(path, delimiter=",", ndmin=2)
    return arr.astype(np.float64, copy=False)
