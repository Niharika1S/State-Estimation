# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp


cdef class RingBuffer:
    cdef Py_ssize_t capacity
    cdef Py_ssize_t ptr
    cdef Py_ssize_t size
    cdef Py_ssize_t dim
    cdef cnp.float64_t[:, :] buf

    def __cinit__(self, Py_ssize_t capacity, Py_ssize_t dim=1):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.dim = dim
        self.buf = np.zeros((capacity, dim), dtype=np.float64)

    cpdef void push(self, cnp.float64_t[:] row):
        cdef Py_ssize_t i
        if row.shape[0] != self.dim:
            raise ValueError(f"expected {self.dim} values, got {row.shape[0]}")
        for i in range(self.dim):
            self.buf[self.ptr, i] = row[i]
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    cpdef cnp.ndarray snapshot(self):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((self.size, self.dim), dtype=np.float64)
        if self.size == 0:
            return out
        if self.size < self.capacity:
            out[:, :] = self.buf[:self.size, :]
            return out
        out[:self.capacity - self.ptr, :] = self.buf[self.ptr:, :]
        out[self.capacity - self.ptr:, :] = self.buf[:self.ptr, :]
        return out

    def as_array(self):
        return np.asarray(self.snapshot())
