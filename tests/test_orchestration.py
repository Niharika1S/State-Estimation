from fastapi.testclient import TestClient

from simcore.api import app
from simcore.orchestration import build_ablation_grid, build_tool_integration_matrix

client = TestClient(app)


def test_tooling_matrix_contains_combinations():
    matrix = build_tool_integration_matrix()
    assert any(item.get("type") == "combo" for item in matrix)
    assert len(matrix) >= 3

    response = client.get("/tooling/matrix")
    assert response.status_code == 200
    payload = response.json()
    assert any(item.get("type") == "combo" for item in payload)


def test_ablation_grid_is_cartesian_and_exposed_via_api():
    grid = build_ablation_grid()
    assert len(grid) == 4 * 3 * 4 * 2 * 2
    assert all(set(item) == {"noise_type", "drift_schedule", "estimator_method", "expose_noise_info", "llm_supervisor"} for item in grid)

    response = client.get("/ablation/grid")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == len(grid)
    assert payload["parameter_grid"][0]["noise_type"] in {"gaussian", "laplace", "student_t", "mixture"}

    summary = client.get("/monitoring/summary")
    assert summary.status_code == 200
    body = summary.json()
    assert body["status"] == "healthy"
    assert "best_estimator_by_rmse" in body


def test_module_selection_in_session_start():
    response = client.post(
        "/session/start",
        json={
            "estimator_method": "conformal",
            "kernel_name": "spectral_mixture",
            "policy_name": "mpc_policy",
            "initial_state": [0.0, 0.2],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_modules"]["estimator"] == "conformal"
    assert body["selected_modules"]["kernel"] == "spectral_mixture"
    assert body["selected_modules"]["policy"] == "mpc_policy"
