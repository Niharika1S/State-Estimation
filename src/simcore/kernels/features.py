from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_rff_frequencies(key, n_features, dim, scale=1.0):
    """Sample RFF frequencies for a shift-invariant kernel approximation."""
    return jax.random.normal(key, shape=(n_features, dim), dtype=jnp.float32) * scale


def rff_features(x, omega, b, lengthscale=1.0):
    """Random Fourier feature map for an RBF-like kernel approximation."""
    x = jnp.asarray(x)
    omega = jnp.asarray(omega)
    b = jnp.asarray(b)
    if x.ndim == 1:
        x = x[None, :]
    scale = jnp.sqrt(2.0 / x.shape[-1])
    z = scale * jnp.cos((x @ omega.T) / max(float(lengthscale), 1e-6) + b)
    return z
