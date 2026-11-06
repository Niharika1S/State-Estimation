from __future__ import annotations

import jax.numpy as jnp


def rbf_kernel(x, y, lengthscale=1.0, amplitude=1.0):
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    diff = x - y
    sq = diff * diff
    return amplitude * jnp.exp(-(0.5 / (lengthscale**2)) * sq)
