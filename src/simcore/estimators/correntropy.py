from __future__ import annotations

import jax.numpy as jnp


def correntropy_update(state, residual, config):
    """Maximum correntropy update as a robust ablation."""
    residual = jnp.asarray(residual, dtype=jnp.float32).ravel()
    sigma = config.get("sigma_kernel", 1.0)
    corr = jnp.exp(-(residual**2) / (2.0 * sigma**2))
    drift_flag = jnp.asarray(corr.mean() < config.get("correntropy_threshold", 0.7), dtype=jnp.bool_)
    state = state.__class__(
        method=state.method,
        residual_buffer=state.residual_buffer,
        buffer_ptr=state.buffer_ptr,
        embedding=state.embedding,
        autocov_estimate=state.autocov_estimate,
        conformal_scores=state.conformal_scores,
        drift_flag=drift_flag,
    )
    return state


def mckf_step(state, kparams, residual, sigma_kernel, config):
    """A simple scalar/vector maximum correntropy Kalman filter step."""
    x_pred = jnp.asarray(state.get("x", 0.0), dtype=jnp.float32)
    P_pred = jnp.asarray(state.get("P", 1.0), dtype=jnp.float32)
    innovation = jnp.asarray(residual, dtype=jnp.float32)
    if x_pred.ndim == 0:
        x_pred = x_pred.reshape((1,))
        innovation = innovation.reshape((1,))
        if P_pred.ndim == 0:
            P_pred = P_pred.reshape((1, 1))
    H = jnp.asarray(state.get("H", jnp.eye(x_pred.size, dtype=x_pred.dtype)), dtype=x_pred.dtype)
    R = jnp.asarray(config.get("R", 1e-2), dtype=x_pred.dtype)
    sigma = jnp.asarray(sigma_kernel, dtype=x_pred.dtype)
    corr_weight = jnp.exp(-(innovation**2) / (2.0 * sigma**2))
    S = H @ P_pred @ H.T + (R / (corr_weight + 1e-8))
    K = P_pred @ H.T @ jnp.linalg.inv(S)
    x_update = x_pred + K @ innovation
    I = jnp.eye(x_pred.size, dtype=x_pred.dtype)
    P_update = (I - K @ H) @ P_pred
    return {"x": x_update, "P": P_update, "weight": corr_weight}
