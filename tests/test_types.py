import jax
import jax.numpy as jnp

from simcore.types import AgentState, BeliefState, KernelParams, NoiseEstimatorState, Observation, PlantState


def test_dataclasses_roundtrip_through_jit():
    state = PlantState(x=jnp.array([1.0, 2.0], dtype=jnp.float32), t=jnp.array(0, dtype=jnp.int32), regime=jnp.array(0, dtype=jnp.int32))
    obs = Observation(y=jnp.array([0.5], dtype=jnp.float32), t=jnp.array(1, dtype=jnp.int32))
    kernel = KernelParams(
        log_lengthscale=jnp.array([0.1, 0.2], dtype=jnp.float32),
        log_amplitude=jnp.array([0.3], dtype=jnp.float32),
        spectral_weights=jnp.array([1.0], dtype=jnp.float32),
        spectral_means=jnp.array([0.5], dtype=jnp.float32),
        spectral_scales=jnp.array([0.1], dtype=jnp.float32),
        time_decay=jnp.array(0.01, dtype=jnp.float32),
    )
    noise_state = NoiseEstimatorState(
        method="als",
        residual_buffer=jnp.zeros((8,), dtype=jnp.float32),
        buffer_ptr=jnp.array(0, dtype=jnp.int32),
        embedding=jnp.array(0.0, dtype=jnp.float32),
        autocov_estimate=jnp.zeros((2,), dtype=jnp.float32),
        conformal_scores=jnp.zeros((1,), dtype=jnp.float32),
        drift_flag=jnp.array(False),
    )
    belief = BeliefState(
        mean=jnp.array([0.0, 0.0], dtype=jnp.float32),
        cov=jnp.eye(2, dtype=jnp.float32),
        noise_summary=jnp.array([0.0], dtype=jnp.float32),
        drift_flag=jnp.array(False),
    )
    agent = AgentState(belief=belief, policy_params={"lr": 1e-3})
    out_state = jax.jit(lambda x: x)(state)
    assert jnp.all(out_state.x == state.x)
    assert out_state.t == state.t
    assert out_state.regime == state.regime
    out_kernel = jax.tree_util.tree_map(lambda x: x, kernel)
    assert jnp.all(out_kernel.log_lengthscale == kernel.log_lengthscale)
    out_agent = jax.jit(lambda x: x)(agent)
    assert jnp.all(out_agent.belief.mean == agent.belief.mean)
    assert jnp.all(out_agent.belief.cov == agent.belief.cov)
    assert jax.jit(lambda x: x)(obs).t == obs.t
    out_noise = jax.jit(lambda x: x)(noise_state)
    assert jnp.all(out_noise.residual_buffer == noise_state.residual_buffer)
    assert out_noise.buffer_ptr == noise_state.buffer_ptr
