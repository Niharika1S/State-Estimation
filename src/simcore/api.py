from __future__ import annotations

import math
import time
import uuid
from typing import Any

import jax
import jax.numpy as jnp
from fastapi import Body, FastAPI, HTTPException

from simcore.agent.belief import build_belief
from simcore.models.kernel_gp import KernelGPModel
from simcore.orchestration import build_ablation_grid, build_n8n_workflow_specs, build_tool_integration_matrix, module_registry
from simcore.plant.dynamics import plant_step
from simcore.sim.loop import _default_inducing_state, _default_kernel_params, _default_noise_state, _estimate_noise_state, _make_initial_agent_state
from simcore.types import PlantState

app = FastAPI(title="simcore orchestration API", version="0.1.0")

SESSIONS: dict[str, dict[str, Any]] = {}

DEFAULT_CONFIG: dict[str, Any] = {
    "noise_distribution": "gaussian",
    "measurement_noise_distribution": "gaussian",
    "noise_scale": 0.05,
    "measurement_noise_scale": 0.02,
    "state_dim": 2,
    "dynamics_model": "pendulum",
    "mpc_horizon": 3,
    "mpc_candidates": 8,
    "expose_noise_info": True,
    "noise_estimator": "mmd",
    "kernel_budget": 8,
    "kernel_cov_scale": 0.1,
    "initial_cov_scale": 0.1,
    "initial_halfwidth": 0.1,
    "control_cost": 1e-3,
    "risk_weight": 0.1,
    "action_scale": 0.5,
    "target_state": [0.0, 0.0],
    "drift_amplitude": 0.0,
    "drift_period": 20.0,
    "initial_state": [0.0, 0.3],
    "seed": 0,
}


def _recursive_to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return _recursive_to_python(value.tolist())
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(k): _recursive_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recursive_to_python(v) for v in value]
    if isinstance(value, (jnp.ndarray,)):
        return _recursive_to_python(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _recursive_to_python(value.item())
        except (AttributeError, ValueError, TypeError):
            pass
    return str(value)


def _coerce_config(payload: dict[str, Any] | None) -> dict[str, Any]:
    config = {**DEFAULT_CONFIG}
    if payload:
        config.update({k: v for k, v in payload.items() if v is not None})
    if config.get("target_state") is not None and not isinstance(config["target_state"], (list, tuple)):
        config["target_state"] = [float(config["target_state"])]
    return config


def _new_session(config: dict[str, Any]) -> dict[str, Any]:
    initial_state = list(config.get("initial_state", [0.0, 0.3]))
    state_dim = len(initial_state)
    initial_x = jnp.asarray(initial_state, dtype=jnp.float32)
    plant_state = PlantState(x=initial_x, t=jnp.asarray(0, dtype=jnp.int32), regime=jnp.asarray(0, dtype=jnp.int32))
    kernel_params = _default_kernel_params(state_dim, plant_state.x.dtype)
    kernel_state = _default_inducing_state(state_dim, plant_state.x.dtype, budget=int(config.get("kernel_budget", 8)))
    noise_state = _default_noise_state(config, state_dim, plant_state.x.dtype)
    agent_state = _make_initial_agent_state(plant_state, config)
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "config": config,
        "plant_state": plant_state,
        "agent_state": agent_state,
        "kernel_state": kernel_state,
        "kernel_params": kernel_params,
        "noise_state": noise_state,
        "step": 0,
        "logs": [],
        "metrics": {
            "rmse": 0.0,
            "coverage": 0.0,
            "drift_detection_latency": None,
            "drift_events": 0,
            "total_cost": 0.0,
            "step_count": 0,
        },
        "drift_event_step": None,
        "last_drift_flag": False,
        "ended": False,
        "created_at": time.time(),
    }
    SESSIONS[session_id] = session
    return session


def _get_session_or_404(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


def _safe_coverage(true_state: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray) -> float:
    true_state = jnp.asarray(true_state)
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    width = 2.0 * jnp.sqrt(jnp.maximum(jnp.diag(cov), 0.0))
    covered = jnp.abs(true_state - mean) <= width
    return float(jnp.mean(covered))


def _choose_action(belief: Any, kernel_state: Any, config: dict[str, Any], key: jax.Array) -> jax.Array:
    action_dim = int(belief.mean.size)
    horizon = int(config.get("mpc_horizon", 3))
    n_candidates = int(config.get("mpc_candidates", 8))
    action_scale = float(config.get("action_scale", 0.5))
    control_cost = float(config.get("control_cost", 1e-3))
    risk_weight = float(config.get("risk_weight", 0.1))
    target_state = jnp.asarray(config.get("target_state", [0.0] * action_dim), dtype=belief.mean.dtype)
    keys = jax.random.split(key, n_candidates)
    candidates = jax.vmap(
        lambda k: jax.random.normal(k, (horizon, action_dim), dtype=jnp.float32) * action_scale
    )(keys)

    def _candidate_cost(seq: jax.Array) -> jax.Array:
        seq = jnp.asarray(seq)
        if seq.ndim == 1:
            seq = seq[:, None]
        predicted = belief.mean + jnp.cumsum(seq[:, : belief.mean.size], axis=0)
        tracking = jnp.sum((predicted - target_state) ** 2)
        effort = control_cost * jnp.sum(seq ** 2)
        uncertainty = risk_weight * jnp.trace(jnp.asarray(belief.cov))
        return tracking + effort + uncertainty

    costs = jax.vmap(_candidate_cost)(candidates)
    return candidates[jnp.argmin(costs), 0]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "active_sessions": len(SESSIONS), "workflows": sorted(build_n8n_workflow_specs().keys())}


@app.get("/estimators")
def list_estimators() -> list[dict[str, str]]:
    return module_registry()["estimators"]


@app.get("/kernels")
def list_kernels() -> list[dict[str, str]]:
    return module_registry()["kernels"]


@app.get("/policies")
def list_policies() -> list[dict[str, str]]:
    return module_registry()["policies"]


@app.get("/tooling/matrix")
def list_tooling_matrix() -> list[dict[str, str]]:
    return build_tool_integration_matrix()


@app.get("/ablation/grid")
def list_ablation_grid() -> dict[str, Any]:
    grid = build_ablation_grid()
    return {
        "count": len(grid),
        "parameter_grid": grid,
    }


@app.get("/monitoring/summary")
def monitoring_summary() -> dict[str, Any]:
    return {
        "status": "healthy",
        "last_run_status": "unknown",
        "best_estimator_by_rmse": None,
        "tool_test_failures_last_24h": 0,
        "artifact_regression": False,
    }


@app.post("/session/start")
def start_session(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    config = _coerce_config(payload)
    if payload.get("estimator_method") is not None:
        config["noise_estimator"] = str(payload["estimator_method"]).lower()
    if payload.get("kernel_name") is not None:
        config["kernel_name"] = str(payload["kernel_name"]).lower()
    if payload.get("policy_name") is not None:
        config["policy_name"] = str(payload["policy_name"]).lower()
    session = _new_session(config)
    return {
        "session_id": session["session_id"],
        "config": _recursive_to_python(config),
        "selected_modules": {
            "estimator": config.get("noise_estimator"),
            "kernel": config.get("kernel_name"),
            "policy": config.get("policy_name"),
        },
        "created_at": session["created_at"],
    }


@app.post("/session/{session_id}/step")
def advance_step(session_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    del payload
    session = _get_session_or_404(session_id)
    if session["ended"]:
        raise HTTPException(status_code=400, detail=f"Session {session_id} has already ended.")

    config = session["config"]
    key = jax.random.PRNGKey(int(config.get("seed", 0)) + int(session["step"]) + 1)
    policy_config = {
        "action_scale": float(config.get("action_scale", 0.5)),
        "control_cost": float(config.get("control_cost", 1e-3)),
        "risk_weight": float(config.get("risk_weight", 0.1)),
        "target_state": list(config.get("target_state", [0.0, 0.0])),
    }
    action = _choose_action(session["agent_state"].belief, session["kernel_state"], config, key)

    plant_state, obs = plant_step(session["plant_state"], action, key, config)
    prediction = jnp.asarray(plant_state.x, dtype=obs.y.dtype)
    residual = obs.y - prediction
    noise_state, halfwidth = _estimate_noise_state(session["noise_state"], residual, config)
    kernel_state = KernelGPModel.update_kernel_model(
        session["kernel_state"],
        session["kernel_params"],
        plant_state.x,
        obs.y,
        forgetting_factor=jnp.asarray(config.get("forgetting_factor", 0.98), dtype=plant_state.x.dtype),
        key=jax.random.PRNGKey(int(config.get("seed", 0)) + int(session["step"]) + 2),
        budget=int(config.get("kernel_budget", 8)),
        novelty_threshold=float(config.get("novelty_threshold", 0.1)),
    )

    kernel_pred_cov = jnp.eye(plant_state.x.size, dtype=plant_state.x.dtype) * float(config.get("kernel_cov_scale", 0.05))
    belief = build_belief(
        prediction,
        kernel_pred_cov,
        noise_state,
        halfwidth,
        expose_noise_info=bool(config.get("expose_noise_info", True)),
        noise_feature_dim=int(config.get("noise_feature_dim", 8)),
    )
    session["agent_state"] = type(session["agent_state"])(belief=belief, policy_params=session["agent_state"].policy_params)
    session["plant_state"] = plant_state
    session["noise_state"] = noise_state
    session["kernel_state"] = kernel_state

    drift_flag = bool(belief.drift_flag)
    step_cost = float(jnp.mean(residual ** 2))
    rmse = float(jnp.sqrt(jnp.mean((plant_state.x - belief.mean) ** 2)))
    coverage = _safe_coverage(plant_state.x, belief.mean, belief.cov)

    if drift_flag and not session["last_drift_flag"]:
        session["drift_event_step"] = session["step"]
        session["drift_detection_latency"] = session["step"] - int(config.get("drift_event_step", session["step"]))
        session["metrics"]["drift_events"] += 1
    session["last_drift_flag"] = drift_flag

    session["metrics"]["step_count"] = int(session["step"]) + 1
    session["metrics"]["total_cost"] = float(session["metrics"].get("total_cost", 0.0)) + step_cost
    session["metrics"]["rmse"] = rmse
    session["metrics"]["coverage"] = coverage

    row = {
        "session_id": session_id,
        "step": int(session["step"]),
        "true_state": _recursive_to_python(plant_state.x),
        "observation": _recursive_to_python(obs.y),
        "belief_mean": _recursive_to_python(belief.mean),
        "belief_cov_trace": float(jnp.trace(belief.cov)),
        "noise_summary": _recursive_to_python(belief.noise_summary),
        "drift_flag": drift_flag,
        "action_taken": _recursive_to_python(action),
        "step_cost": step_cost,
        "rmse": rmse,
        "coverage": coverage,
        "timestamp": time.time(),
    }
    session["logs"].append(row)
    session["step"] += 1

    return {
        "session_id": session_id,
        "observation": row["observation"],
        "belief": {
            "mean": row["belief_mean"],
            "cov": _recursive_to_python(belief.cov),
            "noise_summary": row["noise_summary"],
            "drift_flag": drift_flag,
        },
        "noise_summary": row["noise_summary"],
        "drift_flag": drift_flag,
        "action_taken": row["action_taken"],
        "step_cost": step_cost,
        "true_state": row["true_state"],
        "coverage": coverage,
        "rmse": rmse,
        "step": row["step"],
    }


@app.post("/session/{session_id}/reconfigure")
def reconfigure_session(session_id: str, payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    config = session["config"]
    if payload.get("estimator_method"):
        estimator = str(payload["estimator_method"]).lower()
        valid = module_registry()["estimators"]
        if estimator not in {item["name"] for item in valid}:
            raise HTTPException(status_code=400, detail=f"Unknown estimator: {estimator}")
        config["noise_estimator"] = estimator
    if payload.get("expose_noise_info") is not None:
        config["expose_noise_info"] = bool(payload["expose_noise_info"])
    if payload.get("inject_drift"):
        config["drift_amplitude"] = float(payload.get("drift_amplitude", config.get("drift_amplitude", 0.5)))
        config["drift_period"] = float(payload.get("drift_period", config.get("drift_period", 8.0)))
        config["drift_event_step"] = int(session["step"])
        session["noise_state"] = session["noise_state"].__class__(
            method=session["noise_state"].method,
            residual_buffer=session["noise_state"].residual_buffer,
            buffer_ptr=session["noise_state"].buffer_ptr,
            embedding=session["noise_state"].embedding,
            autocov_estimate=session["noise_state"].autocov_estimate,
            conformal_scores=session["noise_state"].conformal_scores,
            drift_flag=jnp.asarray(True, dtype=jnp.bool_),
        )
        session["last_drift_flag"] = True
    return {
        "session_id": session_id,
        "config": _recursive_to_python(config),
        "drift_event_step": session.get("drift_event_step"),
        "status": "reconfigured",
    }


@app.get("/session/{session_id}/metrics")
def session_metrics(session_id: str) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    metrics = {
        "session_id": session_id,
        "step_count": int(session["metrics"].get("step_count", 0)),
        "rmse": float(session["metrics"].get("rmse", 0.0)),
        "coverage": float(session["metrics"].get("coverage", 0.0)),
        "drift_detection_latency": session["metrics"].get("drift_detection_latency"),
        "drift_events": int(session["metrics"].get("drift_events", 0)),
        "total_cost": float(session["metrics"].get("total_cost", 0.0)),
    }
    if session["drift_event_step"] is not None and session["metrics"].get("drift_detection_latency") is None:
        metrics["drift_detection_latency"] = 0
    return metrics


@app.post("/session/{session_id}/end")
def end_session(session_id: str) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    metrics = session["metrics"]
    summary = {
        "session_id": session_id,
        "status": "ended",
        "step_count": int(metrics.get("step_count", 0)),
        "rmse": float(metrics.get("rmse", 0.0)),
        "coverage": float(metrics.get("coverage", 0.0)),
        "drift_detection_latency": metrics.get("drift_detection_latency"),
        "drift_events": int(metrics.get("drift_events", 0)),
        "total_cost": float(metrics.get("total_cost", 0.0)),
        "config": _recursive_to_python(session["config"]),
        "log_rows": len(session["logs"]),
    }
    session["ended"] = True
    return summary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
