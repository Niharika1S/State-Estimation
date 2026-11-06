"""Benchmark the JAX hot path versus the limited Cython I/O glue layer."""

from __future__ import annotations

import time

import numpy as np

try:
    from simcore.fastio import RingBuffer
except Exception:  # pragma: no cover - only used when extension is not built.
    RingBuffer = None


class PythonRingBuffer:
    def __init__(self, capacity: int, dim: int = 1):
        self.capacity = capacity
        self.dim = dim
        self.buf = np.zeros((capacity, dim), dtype=np.float64)
        self.ptr = 0
        self.size = 0

    def push(self, row):
        row = np.asarray(row, dtype=np.float64)
        if row.shape[0] != self.dim:
            raise ValueError(f"expected {self.dim} values, got {row.shape[0]}")
        self.buf[self.ptr] = row
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def snapshot(self):
        if self.size < self.capacity:
            return self.buf[: self.size].copy()
        return np.concatenate((self.buf[self.ptr:], self.buf[: self.ptr]), axis=0)


def benchmark(buffer_cls, n_pushes: int = 200_000, dim: int = 4, capacity: int = 4096):
    payload = np.random.default_rng(0).standard_normal((dim,), dtype=np.float64)
    buf = buffer_cls(capacity=capacity, dim=dim)
    t0 = time.perf_counter()
    for _ in range(n_pushes):
        buf.push(payload)
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    py_time = benchmark(PythonRingBuffer, n_pushes=100_000)
    if RingBuffer is not None:
        cy_time = benchmark(RingBuffer, n_pushes=100_000)
        ratio = py_time / cy_time if cy_time > 0 else float("inf")
        print(f"python_ringbuffer: {py_time:.6f}s")
        print(f"cython_ringbuffer: {cy_time:.6f}s")
        print(f"speedup: {ratio:.2f}x")
    else:
        print("Cython ring buffer not available in this environment; benchmark skipped.")
        print(f"python_ringbuffer: {py_time:.6f}s")


if __name__ == "__main__":
    main()
