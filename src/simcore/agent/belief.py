from __future__ import annotations

import jax.numpy as jnp

from simcore.types import BeliefState, NoiseEstimatorState


class BeliefBuilder:
    """Simple state-aware belief model."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def build(self, mean, cov, noise_summary, drift_flag):
        return {
            "mean": jnp.asarray(mean),
            "cov": jnp.asarray(cov),
            "noise_summary": jnp.asarray(noise_summary),
            "drift_flag": jnp.asarray(drift_flag),
        }


def build_belief(
    kernel_pred_mean,
    kernel_pred_cov,
    noise_state: NoiseEstimatorState | None,
    conformal_halfwidth,
    expose_noise_info: bool = True,
    noise_feature_dim: int = 8,
):
    """Construct the belief state used by the agent.

    When expose_noise_info is False, the downstream policy operates on the same interface but sees
    zeroed noise features. This makes state-aware vs naive-agent ablations easy to compare.
    """
    kernel_pred_mean = jnp.asarray(kernel_pred_mean)
    kernel_pred_cov = jnp.asarray(kernel_pred_cov)
    conformal_halfwidth = jnp.asarray(conformal_halfwidth)

    noise_summary = jnp.zeros((noise_feature_dim + 2,), dtype=kernel_pred_mean.dtype)
    drift_flag = jnp.asarray(False)

    if noise_state is not None and expose_noise_info:
        embedding = jnp.asarray(noise_state.embedding)
        if embedding.ndim > 0:
            trunc = embedding.reshape(-1)[:noise_feature_dim]
            pad = jnp.zeros((noise_feature_dim - trunc.size,), dtype=trunc.dtype)
            embedding_vec = jnp.concatenate([trunc, pad], axis=0)
        else:
            embedding_vec = jnp.asarray([embedding], dtype=kernel_pred_mean.dtype)
        autocov = jnp.asarray(noise_state.autocov_estimate).reshape(-1)
        if autocov.size == 0:
            autocov = jnp.zeros((2,), dtype=kernel_pred_mean.dtype)
        summary = jnp.concatenate([embedding_vec[:noise_feature_dim], autocov[:2], jnp.asarray([conformal_halfwidth]).reshape(-1)], axis=0)
        noise_summary = jnp.asarray(summary, dtype=kernel_pred_mean.dtype)
        drift_flag = jnp.asarray(noise_state.drift_flag, dtype=jnp.bool_)

    if not expose_noise_info:
        noise_summary = jnp.zeros_like(noise_summary)
        drift_flag = jnp.asarray(False)

    uncertainty_diag = jnp.asarray(conformal_halfwidth)
    if uncertainty_diag.ndim == 0:
        uncertainty_diag = jnp.full((kernel_pred_cov.shape[-1],), jnp.square(uncertainty_diag), dtype=kernel_pred_mean.dtype)
    else:
        uncertainty_diag = jnp.square(uncertainty_diag).reshape(-1)[: kernel_pred_cov.shape[-1]]
        uncertainty_diag = jnp.pad(uncertainty_diag, (0, max(0, kernel_pred_cov.shape[-1] - uncertainty_diag.size)))

    return BeliefState(
        mean=kernel_pred_mean,
        cov=kernel_pred_cov + jnp.diag(uncertainty_diag),
        noise_summary=noise_summary,
        drift_flag=drift_flag,
    )
