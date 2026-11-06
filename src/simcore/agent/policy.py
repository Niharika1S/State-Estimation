from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def sample_action_sequences(key, horizon, n_candidates, action_dim, scale=0.5):
    """Sample candidate action sequences for random-shooting MPC."""
    keys = jax.random.split(key, n_candidates)
    return jax.vmap(
        lambda k: jax.random.normal(k, (horizon, action_dim), dtype=jnp.float32) * scale
    )(keys)


def rollout_cost(belief, action_seq, kernel_model=None, config=None):
    """Simple quadratic rollout cost penalizing state error and control effort."""
    del kernel_model
    config = {} if config is None else config
    mean = jnp.asarray(belief.mean)
    cov = jnp.asarray(belief.cov)
    target = jnp.asarray(config.get("target_state", jnp.zeros_like(mean)))
    action_seq = jnp.asarray(action_seq)
    if action_seq.ndim == 1:
        action_seq = action_seq[:, None]
    predicted = mean + jnp.cumsum(action_seq[:, : mean.size], axis=0)
    tracking = jnp.sum((predicted - target) ** 2)
    effort = config.get("control_cost", 1e-3) * jnp.sum(action_seq ** 2)
    uncertainty = config.get("risk_weight", 0.1) * jnp.trace(cov)
    return tracking + effort + uncertainty


@partial(jax.jit, static_argnames=("horizon", "n_candidates"))
def mpc_policy(belief, kernel_model, horizon, n_candidates, key, config):
    """Jitted MPC policy using random shooting over candidate action sequences."""
    config = {} if config is None else config
    action_dim = belief.mean.size
    action_scale = float(config.get("action_scale", 0.5))
    candidates = sample_action_sequences(key, horizon, n_candidates, action_dim, action_scale)
    costs = jax.vmap(lambda seq: rollout_cost(belief, seq, kernel_model, config))(candidates)
    best_idx = jnp.argmin(costs)
    return candidates[best_idx, 0]


class ModelPredictiveController:
    """Thin wrapper around the jitted random-shooting MPC policy."""

    def __init__(self, horizon: int = 3, n_candidates: int = 16):
        self.horizon = horizon
        self.n_candidates = n_candidates

    def act(self, state, belief, key=None, config=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        return mpc_policy(belief, None, self.horizon, self.n_candidates, key, {**(config or {}), "action_dim": getattr(state, "x", jnp.zeros(1)).size})
