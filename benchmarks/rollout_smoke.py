"""Smoke-test the end-to-end rollout loop and print a compact summary."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from simcore.sim.loop import rollout_system
from simcore.types import PlantState


def main():
    init = PlantState(
        x=jnp.array([0.2, -0.1], dtype=jnp.float32),
        t=jnp.array(0, dtype=jnp.int32),
        regime=jnp.array(0, dtype=jnp.int32),
    )
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
    final_state, agent_state, kernel_state, noise_state, logs = rollout_system(
        init,
        n_steps=25,
        config=config,
        key=jax.random.PRNGKey(0),
    )
    print(f"final_t={int(final_state.t)}")
    print(f"final_x={np.asarray(final_state.x)}")
    print(f"mean_log_mse={float(jnp.mean(logs[:, -1]))}")
    print(f"drift_flag_last={bool(noise_state.drift_flag)}")


if __name__ == "__main__":
    import numpy as np

    main()
