from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any


def module_registry() -> dict[str, list[dict[str, str]]]:
    return {
        "estimators": [
            {"name": "als", "type": "estimator"},
            {"name": "mmd", "type": "estimator"},
            {"name": "conformal", "type": "estimator"},
            {"name": "correntropy", "type": "estimator"},
        ],
        "kernels": [
            {"name": "rbf", "type": "kernel"},
            {"name": "spectral_mixture", "type": "kernel"},
        ],
        "policies": [
            {"name": "mpc_policy", "type": "policy"},
        ],
    }


def build_tool_integration_matrix(
    estimators: list[str] | None = None,
    kernels: list[str] | None = None,
    policies: list[str] | None = None,
    combinations_only: bool = False,
) -> list[dict[str, str]]:
    registry = module_registry()
    estimator_names = estimators if estimators is not None else [item["name"] for item in registry["estimators"]]
    kernel_names = kernels if kernels is not None else [item["name"] for item in registry["kernels"]]
    policy_names = policies if policies is not None else [item["name"] for item in registry["policies"]]

    matrix: list[dict[str, str]] = []
    if not combinations_only:
        for kind, name in [
            ("estimator", item) for item in estimator_names
        ] + [("kernel", item) for item in kernel_names] + [("policy", item) for item in policy_names]:
            matrix.append({"type": kind, "name": name})

    for estimator, kernel, policy in product(estimator_names, kernel_names, policy_names):
        matrix.append({
            "type": "combo",
            "estimator": estimator,
            "kernel": kernel,
            "policy": policy,
        })
    return matrix


def build_ablation_grid(
    noise_types: list[str] | None = None,
    drift_schedules: list[str] | None = None,
    estimators: list[str] | None = None,
    expose_noise_info: list[bool] | None = None,
    llm_supervisor: list[bool] | None = None,
) -> list[dict[str, Any]]:
    if noise_types is None:
        noise_types = ["gaussian", "laplace", "student_t", "mixture"]
    if drift_schedules is None:
        drift_schedules = ["none", "linear", "pulse"]
    if estimators is None:
        estimators = [item["name"] for item in module_registry()["estimators"]]
    if expose_noise_info is None:
        expose_noise_info = [False, True]
    if llm_supervisor is None:
        llm_supervisor = [False, True]

    grid: list[dict[str, Any]] = []
    for noise_type in noise_types:
        for drift_schedule in drift_schedules:
            for estimator_method in estimators:
                for expose_flag in expose_noise_info:
                    for llm_enabled in llm_supervisor:
                        grid.append({
                            "noise_type": noise_type,
                            "drift_schedule": drift_schedule,
                            "estimator_method": estimator_method,
                            "expose_noise_info": bool(expose_flag),
                            "llm_supervisor": bool(llm_enabled),
                        })
    return grid


def build_n8n_workflow_specs(base_url: str = "http://simcore:8000") -> dict[str, Any]:
    workflows: dict[str, Any] = {}

    workflows["workflow_1_core_agent_loop"] = {
        "name": "Workflow 1 — Core agentic simulation loop",
        "description": "Single episode loop with HTTP session lifecycle and drift-aware LLM override",
        "base_url": base_url,
        "run_order": [
            "Manual Trigger",
            "POST /session/start",
            "Set loop counters",
            "SplitInBatches",
            "POST /session/{id}/step",
            "IF drift_flag",
            "LLM supervisor decision",
            "POST /session/{id}/reconfigure",
            "Insert sim_runs row",
            "POST /session/{id}/end",
        ],
    }

    workflows["workflow_2_tool_integration_harness"] = {
        "name": "Workflow 2 — Tool / estimator integration test harness",
        "description": "Smoke-test each estimator/kernel/policy through its HTTP contract before live runs",
        "base_url": base_url,
        "module_matrix": build_tool_integration_matrix(),
        "smoke_test_strategy": {
            "single_module_passes_first": True,
            "combo_matrix": [
                {"estimator": "mmd", "kernel": "rbf", "policy": "mpc_policy"},
                {"estimator": "conformal", "kernel": "spectral_mixture", "policy": "mpc_policy"},
            ],
        },
        "run_order": [
            "Manual Trigger",
            "GET /estimators, /kernels, /policies",
            "Merge module registry",
            "SplitInBatches",
            "POST /session/start",
            "Smoke-test loop",
            "Assert invariants",
            "POST /session/{id}/end",
            "Aggregate test report",
        ],
    }

    workflows["workflow_3_ablation_runner"] = {
        "name": "Workflow 3 — Ablation / experiment runner",
        "description": "Grid search over drift/noise/estimator/expose_noise_info combinations with sub-workflow execution",
        "base_url": base_url,
        "parameter_grid": build_ablation_grid(),
        "expected_runs": len(build_ablation_grid()),
        "run_order": [
            "Manual or schedule trigger",
            "Build parameter grid",
            "SplitInBatches",
            "Execute Workflow 1",
            "Extract final metrics",
            "Insert ablation_results row",
            "Summarize report",
        ],
    }

    workflows["workflow_4_monitoring_dashboard"] = {
        "name": "Workflow 4 — Monitoring, dashboard, and alerting",
        "description": "Daily or periodic summary of sim_summaries, tool_test_results, and ablation_results",
        "base_url": base_url,
        "alert_rules": {
            "failures_in_last_24h": True,
            "metric_regression_vs_baseline": True,
            "notify": ["Slack", "Email"],
        },
        "dashboard_kpis": [
            "last_run_status",
            "best_estimator_by_rmse",
            "tool_test_failures_last_24h",
            "artifact_regression",
        ],
        "run_order": [
            "Schedule Trigger",
            "Query latest Postgres rows",
            "Compute headline metrics",
            "Alert if thresholds breached",
            "Publish dashboard snapshot",
        ],
    }

    return deepcopy(workflows)


__all__ = [
    "module_registry",
    "build_tool_integration_matrix",
    "build_ablation_grid",
    "build_n8n_workflow_specs",
]
