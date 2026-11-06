from __future__ import annotations

import jax.numpy as jnp


def spectral_mixture_kernel(x, xp, t, tp, params):
    """A spectral-mixture kernel with a time-decay term for drifted regimes."""
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    dt = jnp.asarray(t) - jnp.asarray(tp)
    dx = x - xp
    weights = jnp.asarray(params.spectral_weights, dtype=x.dtype)
    means = jnp.asarray(params.spectral_means, dtype=x.dtype)
    scales = jnp.asarray(params.spectral_scales, dtype=x.dtype)
    decay = jnp.asarray(params.time_decay, dtype=x.dtype)
    amplitude = jnp.asarray(params.log_amplitude, dtype=x.dtype)
    if dx.ndim == 0:
        dx = jnp.asarray([dx], dtype=x.dtype)
    spatial = jnp.sum(
        weights
        * jnp.exp(-2.0 * jnp.pi**2 * (dx**2) * scales)
        * jnp.cos(2.0 * jnp.pi * dx * means),
        axis=-1,
    )
    temporal = jnp.exp(-decay * dt**2)
    return amplitude * spatial * temporal
