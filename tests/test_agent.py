import jax
import jax.numpy as jnp

from simcore.agent.belief import BeliefBuilder, build_belief
from simcore.agent.policy import ModelPredictiveController, mpc_policy
from simcore.types import BeliefState, NoiseEstimatorState


def test_belief_builder_runs():
    builder = BeliefBuilder(dim=2)
    belief = builder.build(jnp.zeros((2,)), jnp.eye(2), jnp.zeros((2,)), False)
    assert belief["mean"].shape == (2,)

    noise_state = NoiseEstimatorState(
        method="mmd",
        residual_buffer=jnp.zeros((8,), dtype=jnp.float32),
        buffer_ptr=jnp.array(0, dtype=jnp.int32),
        embedding=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
        autocov_estimate=jnp.array([0.1, 0.2], dtype=jnp.float32),
        conformal_scores=jnp.zeros((8,), dtype=jnp.float32),
        drift_flag=jnp.array(True),
    )
    belief_state = build_belief(jnp.zeros((2,)), jnp.eye(2), noise_state, 0.1, expose_noise_info=True)
    assert isinstance(belief_state, BeliefState)
    assert belief_state.noise_summary.shape[0] > 0

    masked = build_belief(jnp.zeros((2,)), jnp.eye(2), noise_state, 0.1, expose_noise_info=False)
    assert jnp.all(masked.noise_summary == 0)
    assert not bool(masked.drift_flag)


def test_policy_returns_vector():
    policy = ModelPredictiveController(horizon=2, n_candidates=8)
    state = type("State", (), {"x": jnp.zeros((2,))})()
    action = policy.act(state, BeliefState(jnp.zeros((2,)), jnp.eye(2), jnp.zeros((2,)), jnp.array(False)), key=jax.random.PRNGKey(0))
    assert action.shape == (2,)

    direct = mpc_policy(
        BeliefState(jnp.zeros((2,)), jnp.eye(2), jnp.zeros((2,)), jnp.array(False)),
        None,
        2,
        8,
        jax.random.PRNGKey(1),
        {"action_dim": 2},
    )
    assert direct.shape == (2,)
