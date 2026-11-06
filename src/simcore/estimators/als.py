from __future__ import annotations

import jax.numpy as jnp


def _ring_buffer_push(buffer, ptr, value):
    value = jnp.asarray(value, dtype=buffer.dtype)
    new_buffer = buffer.at[ptr].set(value)
    new_ptr = (ptr + 1) % buffer.shape[0]
    return new_buffer, new_ptr


def als_update(state, residual, config):
    """Autocovariance least squares baseline update."""
    residual = jnp.asarray(residual, dtype=jnp.float32).ravel()
    buffer = state.residual_buffer
    if buffer is None:
        buffer = jnp.zeros(config.get("ring_buffer_size", 32), dtype=jnp.float32)
    next_buffer, next_ptr = _ring_buffer_push(buffer, state.buffer_ptr, residual.mean())
    values = next_buffer
    max_lag = min(config.get("als_lags", 4), values.size - 1)
    if max_lag < 0:
        estimate = jnp.array([jnp.mean(values ** 2), jnp.mean(values ** 2)], dtype=jnp.float32)
    else:
        covs = jnp.stack([
            jnp.mean(values * values[jnp.maximum(0, jnp.arange(values.size) - lag)])
            if lag == 0
            else jnp.mean(values[:-lag] * values[lag:])
            for lag in range(max_lag + 1)
        ])
        q = jnp.maximum(covs[0] - covs[1] if covs.size > 1 else covs[0], 1e-6)
        r = jnp.maximum(covs[0] + (covs[1] if covs.size > 1 else 0.0), 1e-6)
        estimate = jnp.stack([q, r])
    updated = state.__class__(
        method=state.method,
        residual_buffer=next_buffer,
        buffer_ptr=next_ptr,
        embedding=state.embedding,
        autocov_estimate=estimate,
        conformal_scores=state.conformal_scores,
        drift_flag=state.drift_flag,
    )
    return updated
