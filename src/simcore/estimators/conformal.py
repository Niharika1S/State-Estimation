from __future__ import annotations

import jax.numpy as jnp


def _ring_buffer_push(buffer, ptr, value):
    value = jnp.asarray(value, dtype=buffer.dtype)
    new_buffer = buffer.at[ptr].set(value)
    new_ptr = (ptr + 1) % buffer.shape[0]
    return new_buffer, new_ptr


def conformal_update(state, residual, config):
    """Online adaptive conformal update: keep a calibration buffer and adapt alpha."""
    residual = jnp.asarray(residual, dtype=jnp.float32).ravel()
    score = jnp.abs(residual).mean()
    buffer = state.conformal_scores
    if buffer is None:
        buffer = jnp.zeros(config.get("conformal_window", 32), dtype=jnp.float32)
    new_scores, next_ptr = _ring_buffer_push(buffer, state.buffer_ptr, score)
    alpha = config.get("alpha", 0.1)
    gamma = config.get("aci_gamma", 0.05)
    q_level = config.get("conformal_quantile", 1.0 - alpha)
    q = jnp.quantile(new_scores, q_level)
    halfwidth = jnp.asarray(q * config.get("conformal_scale", 1.0), dtype=jnp.float32)
    coverage_fail = jnp.asarray(jnp.abs(residual).mean() > halfwidth, dtype=jnp.float32)
    alpha_next = jnp.clip(alpha + gamma * (coverage_fail - alpha), 0.01, 0.49)
    drift_flag = jnp.asarray((coverage_fail > 0.0) or (halfwidth > 0.0), dtype=jnp.bool_)
    updated = state.__class__(
        method=state.method,
        residual_buffer=state.residual_buffer,
        buffer_ptr=next_ptr,
        embedding=state.embedding,
        autocov_estimate=state.autocov_estimate,
        conformal_scores=new_scores,
        drift_flag=drift_flag,
    )
    if config.get("return_interval", False):
        return updated, halfwidth
    return updated
