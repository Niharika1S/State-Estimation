from fastapi.testclient import TestClient

from simcore.api import app
from simcore.orchestration import build_n8n_workflow_specs

client = TestClient(app)


def test_health_and_module_lists():
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True

    estimators = client.get("/estimators")
    kernels = client.get("/kernels")
    policies = client.get("/policies")

    assert estimators.status_code == 200
    assert kernels.status_code == 200
    assert policies.status_code == 200
    assert {item["name"] for item in estimators.json()} >= {"mmd", "conformal"}
    assert {item["name"] for item in kernels.json()} >= {"rbf"}
    assert {item["name"] for item in policies.json()} >= {"mpc_policy"}


def test_session_lifecycle_reconfigure_and_metrics():
    start = client.post(
        "/session/start",
        json={
            "noise_estimator": "mmd",
            "expose_noise_info": True,
            "n_steps": 3,
            "initial_state": [0.0, 0.3],
        },
    )
    assert start.status_code == 200, start.text
    session_id = start.json()["session_id"]

    for _ in range(2):
        step = client.post(f"/session/{session_id}/step", json={})
        assert step.status_code == 200, step.text
        payload = step.json()
        assert payload["drift_flag"] in {True, False}
        assert "observation" in payload
        assert "action_taken" in payload

    reconfigure = client.post(
        f"/session/{session_id}/reconfigure",
        json={"estimator_method": "conformal", "expose_noise_info": False, "inject_drift": True},
    )
    assert reconfigure.status_code == 200, reconfigure.text

    metrics = client.get(f"/session/{session_id}/metrics")
    assert metrics.status_code == 200, metrics.text
    body = metrics.json()
    assert body["session_id"] == session_id
    assert body["step_count"] >= 2
    assert "rmse" in body
    assert "coverage" in body

    summary = client.post(f"/session/{session_id}/end")
    assert summary.status_code == 200, summary.text
    assert summary.json()["status"] == "ended"


def test_n8n_workflow_specs_exist():
    workflows = build_n8n_workflow_specs()
    assert set(workflows).issuperset({"workflow_1_core_agent_loop", "workflow_2_tool_integration_harness", "workflow_3_ablation_runner", "workflow_4_monitoring_dashboard"})
    for name, workflow in workflows.items():
        assert "name" in workflow
        assert "run_order" in workflow
        assert workflow["run_order"]
        assert "base_url" in workflow
