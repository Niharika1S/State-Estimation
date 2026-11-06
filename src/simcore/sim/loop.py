from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from simcore.agent.belief import build_belief
from simcore.agent.policy import mpc_policy
from simcore.estimators.als import als_update
from simcore.estimators.conformal import conformal_update
from simcore.estimators.correntropy import correntropy_update
from simcore.estimators.mmd_embedding import mmd_update
from simcore.models.kernel_gp import KernelGPModel
from simcore.plant.dynamics import plant_step
from simcore.types import AgentState, InducingState, KernelParams, NoiseEstimatorState


def _default_kernel_params(state_dim: int, dtype=jnp.float32):
    return KernelParams(
        log_lengthscale=jnp.array(jnp.log(1.0), dtype=dtype),
        log_amplitude=jnp.array(0.0, dtype=dtype),
        spectral_weights=jnp.array([1.0], dtype=dtype),
        spectral_means=jnp.array([0.1], dtype=dtype),
        spectral_scales=jnp.array([1.0], dtype=dtype),
        time_decay=jnp.array(0.1, dtype=dtype),
    )


def _default_inducing_state(state_dim: int, dtype=jnp.float32, budget: int = 8):
    return InducingState(
        Z=jnp.zeros((budget, state_dim), dtype=dtype),
        m=jnp.zeros((budget,), dtype=dtype),
        S=jnp.ones((budget,), dtype=dtype),
    )


def _default_noise_state(config, state_dim: int, dtype=jnp.float32):
    ring_size = int(config.get("ring_buffer_size", 32))
    return NoiseEstimatorState(
        method=config.get("noise_estimator", "mmd"),
        residual_buffer=jnp.zeros((ring_size,), dtype=dtype),
        buffer_ptr=jnp.array(0, dtype=jnp.int32),
        embedding=jnp.zeros((int(config.get("mmd_dim", 16)),), dtype=dtype),
        autocov_estimate=jnp.zeros((2,), dtype=dtype),
        conformal_scores=jnp.zeros((int(config.get("conformal_window", ring_size)),), dtype=dtype),
        drift_flag=jnp.array(False, dtype=jnp.bool_),
    )


def _make_initial_agent_state(initial_plant_state, config):
    state_dim = int(initial_plant_state.x.size)
    dtype = initial_plant_state.x.dtype
    noise_state = _default_noise_state(config, state_dim, dtype)
    mean = jnp.asarray(initial_plant_state.x, dtype=dtype)
    cov = jnp.eye(state_dim, dtype=dtype) * float(config.get("initial_cov_scale", 0.1))
    halfwidth = jnp.full((state_dim,), float(config.get("initial_halfwidth", 0.1)), dtype=dtype)
    belief = build_belief(
        mean,
        cov,
        noise_state,
        halfwidth,
        expose_noise_info=bool(config.get("expose_noise_info", True)),
        noise_feature_dim=int(config.get("noise_feature_dim", 8)),
    )
    return AgentState(
        belief=belief,
        policy_params={
            "horizon": int(config.get("mpc_horizon", 3)),
            "n_candidates": int(config.get("mpc_candidates", 16)),
        },
    )


def _estimate_noise_state(noise_state, residual, config):
    method = config.get("noise_estimator", "mmd")
    if method == "als":
        return als_update(noise_state, residual, config), jnp.asarray(config.get("initial_halfwidth", 0.1), dtype=residual.dtype)
    if method == "conformal":
        updated, halfwidth = conformal_update(noise_state, residual, {**config, "return_interval": True})
        return updated, halfwidth
    if method == "correntropy":
        updated = correntropy_update(noise_state, residual, config)
        return updated, jnp.asarray(config.get("sigma_kernel", 1.0), dtype=residual.dtype)
    updated = mmd_update(noise_state, residual, None, config)
    return updated, jnp.asarray(config.get("initial_halfwidth", 0.1), dtype=residual.dtype)


def _policy_config(config):
    allowed = {
        "action_scale",
        "control_cost",
        "risk_weight",
        "target_state",
        "mpc_horizon",
        "mpc_candidates",
        "action_dim",
        "initial_cov_scale",
    }
    return {k: v for k, v in config.items() if k in allowed and not isinstance(v, str)}


def rollout_system(initial_plant_state, initial_agent_state=None, n_steps=10, config=None, key=None):
    """JIT-compiled end-to-end rollout for the plant + kernel model + estimator + agent loop."""
    config = {} if config is None else dict(config)
    if key is None:
        key = jax.random.PRNGKey(0)

    state_dim = int(initial_plant_state.x.size)
    dtype = initial_plant_state.x.dtype
    kernel_params = config.get("kernel_params", _default_kernel_params(state_dim, dtype))
    kernel_model_state = config.get("kernel_model_state", _default_inducing_state(state_dim, dtype, budget=int(config.get("kernel_budget", 8))))
    noise_state = config.get("noise_state", _default_noise_state(config, state_dim, dtype))
    if initial_agent_state is None:
        initial_agent_state = _make_initial_agent_state(initial_plant_state, config)

    @partial(jax.jit, static_argnames=("n_steps",))
    def _scan_loop(plant_state, agent_state, kernel_state, noise_state, key, n_steps):
        def body(carry, _):
            plant_state, agent_state, kernel_state, noise_state, key = carry
            key, k_action, k_plant, k_update = jax.random.split(key, 4)

            action = mpc_policy(
                agent_state.belief,
                kernel_state,
                int(config.get("mpc_horizon", 3)),
                int(config.get("mpc_candidates", 16)),
                k_action,
                _policy_config(config),
            )

            plant_state, obs = plant_step(plant_state, action, k_plant, config)
            prediction = jnp.asarray(plant_state.x, dtype=obs.y.dtype)
            residual = obs.y - prediction

            noise_state, conformal_halfwidth = _estimate_noise_state(noise_state, residual, config)
            kernel_state = KernelGPModel.update_kernel_model(
                kernel_state,
                kernel_params,
                plant_state.x,
                obs.y,
                forgetting_factor=jnp.asarray(config.get("forgetting_factor", 0.98), dtype=plant_state.x.dtype),
                key=k_update,
                budget=int(config.get("kernel_budget", 8)),
                novelty_threshold=float(config.get("novelty_threshold", 0.1)),
            )

            kernel_pred_cov = jnp.eye(state_dim, dtype=plant_state.x.dtype) * float(config.get("kernel_cov_scale", 0.05))
            belief = build_belief(
                prediction,
                kernel_pred_cov,
                noise_state,
                conformal_halfwidth,
                expose_noise_info=bool(config.get("expose_noise_info", True)),
                noise_feature_dim=int(config.get("noise_feature_dim", 8)),
            )
            belief = belief.__class__(
                mean=belief.mean,
                cov=belief.cov,
                noise_summary=jnp.pad(
                    belief.noise_summary,
                    (0, max(0, 12 - belief.noise_summary.size)),
                    mode="constant",
                )[:12],
                drift_flag=belief.drift_flag,
            )
            agent_state = AgentState(belief=belief, policy_params=agent_state.policy_params)

            log_row = jnp.concatenate(
                [
                    plant_state.x.reshape(-1),
                    obs.y.reshape(-1),
                    action.reshape(-1),
                    jnp.asarray([noise_state.drift_flag], dtype=plant_state.x.dtype),
                    jnp.asarray([jnp.mean(residual**2)], dtype=plant_state.x.dtype),
                ]
            )
            return (plant_state, agent_state, kernel_state, noise_state, key), log_row

        (final_plant_state, final_agent_state, final_kernel_state, final_noise_state, _), logs = jax.lax.scan(
            body,
            (plant_state, agent_state, kernel_state, noise_state, key),
            xs=None,
            length=n_steps,
        )
        return final_plant_state, final_agent_state, final_kernel_state, final_noise_state, logs

    final_plant_state, final_agent_state, final_kernel_state, final_noise_state, logs = _scan_loop(
        initial_plant_state,
        initial_agent_state,
        kernel_model_state,
        noise_state,
        key,
        n_steps,
    )
    return final_plant_state, final_agent_state, final_kernel_state, final_noise_state, logs


def rollout(init_state, step_fn, horizon, actions=None, key=None, config=None):
    """Scan-based rollout. Supports either action sequences or zero actions."""
    if key is None:
        key = jax.random.PRNGKey(0)
    if actions is None:
        actions = jnp.zeros((horizon,), dtype=init_state.x.dtype)
    else:
        actions = jnp.asarray(actions, dtype=init_state.x.dtype)
        if actions.ndim == 0:
            actions = jnp.full((horizon,), actions, dtype=init_state.x.dtype)

    def body(carry, action):
        state, loop_key = carry
        step_key, loop_key = jax.random.split(loop_key)
        try:
            next_state, obs = step_fn(state, action, step_key, config or {})
        except TypeError:
            try:
                next_state, obs = step_fn(state, action, config or {})
            except TypeError:
                next_state, obs = step_fn(state, action)
        return (next_state, loop_key), obs

    (final_state, _), outputs = jax.lax.scan(body, (init_state, key), xs=actions)
    return final_state, outputs
