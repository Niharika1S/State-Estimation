import jax
import jax.numpy as jnp

from simcore.sim.loop import rollout, rollout_system
from simcore.types import PlantState


def step_fn(state, action=None, config=None):
    next_state = PlantState(x=state.x + 1.0, t=state.t + 1, regime=state.regime)
    obs = next_state.x
    return next_state, obs


def test_rollout_runs_and_return_shape():
    init = PlantState(x=jnp.array([0.0, 1.0], dtype=jnp.float32), t=jnp.array(0, dtype=jnp.int32), regime=jnp.array(0, dtype=jnp.int32))
    final_state, outputs = rollout(init, step_fn, 3)
    assert outputs.shape == (3, 2)
    assert final_state.t == 3


def test_system_rollout_runs():
    init = PlantState(x=jnp.array([0.0, 0.3], dtype=jnp.float32), t=jnp.array(0, dtype=jnp.int32), regime=jnp.array(0, dtype=jnp.int32))
    config = {
        "noise_distribution": "gaussian",
        "noise_scale": 0.05,
        "state_dim": 2,
        "mpc_horizon": 3,
        "mpc_candidates": 8,
        "expose_noise_info": True,
        "noise_estimator": "mmd",
        "kernel_budget": 8,
        "kernel_cov_scale": 0.1,
        "initial_cov_scale": 0.1,
    }
    final_state, _, _, _, logs = rollout_system(init, n_steps=5, config=config, key=jax.random.PRNGKey(0))
    assert logs.shape == (5, 8)
    assert final_state.t == 5
