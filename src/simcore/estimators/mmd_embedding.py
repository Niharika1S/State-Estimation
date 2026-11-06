from __future__ import annotations

import jax.numpy as jnp


def adaptive_threshold(config, buffer_size=32):
    """Return a threshold for the MMD embedding drift score.

    The fixed-threshold mode is the default for experiments; the online calibration is a simple
    scale-based threshold derived from the current residual buffer length.
    """
    if "mmd_threshold" in config:
        return jnp.asarray(config["mmd_threshold"], dtype=jnp.float32)
    scale = config.get("mmd_scale", 0.25)
    q = config.get("mmd_quantile", 0.95)
    return jnp.asarray(scale * (buffer_size ** 0.5) * (1.0 + (1.0 - q)), dtype=jnp.float32)


def _push_buffer(buffer, ptr, value):
    value = jnp.asarray(value, dtype=buffer.dtype)
    next_buffer = buffer.at[ptr].set(value)
    next_ptr = (ptr + 1) % buffer.shape[0]
    return next_buffer, next_ptr


def mmd_update(state, residual, kparams_noise, config):
    """Kernel mean embedding tracker with an adaptive MMD drift threshold."""
    del kparams_noise
    residual = jnp.asarray(residual, dtype=jnp.float32).ravel()
    buffer = state.residual_buffer
    if buffer is None:
        buffer = jnp.zeros(config.get("ring_buffer_size", 32), dtype=jnp.float32)
    next_buffer, next_ptr = _push_buffer(buffer, state.buffer_ptr, residual.mean())

    dim = config.get("mmd_dim", 16)
    omega = config.get("mmd_omega")
    if omega is None:
        omega = jnp.linspace(-2.0, 2.0, dim, dtype=jnp.float32)
    else:
        omega = jnp.asarray(omega, dtype=jnp.float32)
    phi = jnp.sqrt(2.0 / dim) * jnp.cos(next_buffer[:, None] * omega[None, :] + 0.0)
    embedding = phi.mean(axis=0)
    prev = state.embedding
    if prev is None or prev.shape != embedding.shape:
        prev = jnp.zeros_like(embedding)
    mmd2 = jnp.sum((embedding - prev) ** 2)
    threshold = adaptive_threshold(config, buffer_size=next_buffer.shape[0])
    drift_flag = jnp.asarray(mmd2 > threshold, dtype=jnp.bool_)
    alpha = config.get("mmd_alpha", 0.3)
    embedding_new = (1.0 - alpha) * prev + alpha * embedding
    updated = state.__class__(
        method=state.method,
        residual_buffer=next_buffer,
        buffer_ptr=next_ptr,
        embedding=embedding_new,
        autocov_estimate=state.autocov_estimate,
        conformal_scores=state.conformal_scores,
        drift_flag=drift_flag,
    )
    return updated
