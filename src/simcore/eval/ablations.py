from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp

from simcore.eval.metrics import empirical_coverage, predictive_nll, rmse
from simcore.sim.loop import rollout_system
from simcore.types import PlantState


def _default_configs():
    return {
        "noise_types": ["gaussian", "student_t", "mixture"],
        "estimator_methods": ["als", "mmd", "conformal"],
        "drift_schedules": ["none", "step"],
        "expose_noise_info": [True, False],
    }


def _build_config(noise_type, estimator_method, drift_schedule, expose_noise_info, step_scale=0.05):
    config = {
        "noise_distribution": noise_type,
        "noise_scale": step_scale,
        "state_dim": 2,
        "mpc_horizon": 3,
        "mpc_candidates": 8,
        "expose_noise_info": bool(expose_noise_info),
        "noise_estimator": estimator_method,
        "kernel_budget": 8,
        "kernel_cov_scale": 0.1,
        "initial_cov_scale": 0.1,
        "conformal_window": 16,
        "ring_buffer_size": 32,
        "mmd_dim": 8,
    }
    if noise_type == "student_t":
        config["student_t_df"] = 3.0
    if drift_schedule == "step":
        config["regime_schedule"] = [0, 1]
        config["regime_change_probability"] = 0.0
    return config


def run_ablation(
    noise_types=None,
    drift_schedules=None,
    estimator_methods=None,
    expose_noise_info=None,
    n_steps=50,
    seed=0,
    output_csv=None,
):
    """Run a compact ablation sweep and return a list of metrics rows.

    The routine intentionally keeps the configuration deterministic and lightweight enough to run as a
    smoke benchmark without depending on a full research harness. It records the requested summary
    metrics and can optionally write them to CSV.
    """
    noise_types = _default_configs()["noise_types"] if noise_types is None else list(noise_types)
    drift_schedules = _default_configs()["drift_schedules"] if drift_schedules is None else list(drift_schedules)
    estimator_methods = _default_configs()["estimator_methods"] if estimator_methods is None else list(estimator_methods)
    expose_noise_info = _default_configs()["expose_noise_info"] if expose_noise_info is None else list(expose_noise_info)

    rows = []
    for noise_type in noise_types:
        for drift_schedule in drift_schedules:
            for estimator_method in estimator_methods:
                for expose in expose_noise_info:
                    config = _build_config(noise_type, estimator_method, drift_schedule, expose)
                    init_state = PlantState(
                        x=jnp.array([0.1, -0.2], dtype=jnp.float32),
                        t=jnp.array(0, dtype=jnp.int32),
                        regime=jnp.array(0, dtype=jnp.int32),
                    )
                    final_state, _, _, _, logs = rollout_system(
                        init_state,
                        n_steps=n_steps,
                        config=config,
                        key=jax.random.PRNGKey(seed + len(rows)),
                    )
                    true_state = logs[:, :2]
                    observed_state = logs[:, 2:4]
                    halfwidth = jnp.maximum(jnp.abs(observed_state - true_state).mean(axis=1), 1e-6)
                    row = {
                        "noise_type": noise_type,
                        "drift_schedule": drift_schedule,
                        "estimator_method": estimator_method,
                        "expose_noise_info": bool(expose),
                        "state_rmse": rmse(true_state, observed_state),
                        "predictive_nll": predictive_nll(true_state.mean(axis=0), jnp.eye(2) * 0.1, observed_state[-1]),
                        "coverage": empirical_coverage(true_state, halfwidth, observed_state),
                        "drift_flag": bool(jnp.any(logs[:, -2] > 0.0)),
                        "final_t": int(final_state.t),
                    }
                    rows.append(row)

    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(rows[0].keys()) if rows else [
            "noise_type",
            "drift_schedule",
            "estimator_method",
            "expose_noise_info",
            "state_rmse",
            "predictive_nll",
            "coverage",
            "drift_flag",
            "final_t",
        ]
        with path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    return rows
