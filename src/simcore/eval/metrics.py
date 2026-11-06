from __future__ import annotations

import numpy as np


def mse(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean((pred - target) ** 2))


def rmse(pred, target):
    return float(np.sqrt(mse(pred, target)))


def predictive_nll(mean, cov, target):
    mean = np.asarray(mean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    dim = mean.shape[-1] if mean.ndim > 0 else 1
    if cov.ndim == 1:
        cov = np.diag(cov)
    if cov.shape == ():
        cov = np.array([[float(cov)]])
    diff = target - mean
    if diff.ndim == 0:
        diff = diff.reshape(1)
    eye = np.eye(cov.shape[-1], dtype=np.float64)
    cov_reg = cov + 1e-6 * eye
    sign, logdet = np.linalg.slogdet(cov_reg)
    if sign <= 0:
        cov_reg = cov_reg + 1e-6 * eye
        sign, logdet = np.linalg.slogdet(cov_reg)
    solved = np.linalg.solve(cov_reg, diff)
    return float(0.5 * (dim * np.log(2.0 * np.pi) + logdet + diff @ solved))


def empirical_coverage(prediction, interval_halfwidth, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    interval_halfwidth = np.asarray(interval_halfwidth, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(target - prediction) <= np.maximum(interval_halfwidth, 0.0)))


def mmd_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def drift_detection_latency(flags, change_point):
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return float("inf")
    hit = np.flatnonzero(flags)
    if hit.size == 0:
        return float("inf")
    first_hit = int(hit[0])
    return float(max(0, first_hit - int(change_point)))
