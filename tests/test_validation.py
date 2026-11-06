import numpy as np
import jax.numpy as jnp
from hypothesis import given, strategies as st

from simcore.estimators.conformal import conformal_update
from simcore.fastio import RingBuffer
from simcore.kernels.time_varying import spectral_mixture_kernel
from simcore.sim.loop import rollout_system
from simcore.types import KernelParams, NoiseEstimatorState, PlantState


@given(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
)
def test_kernel_params_remain_finite(log_lengthscale, weight, mean):
    params = KernelParams(
        log_lengthscale=jnp.array(log_lengthscale, dtype=jnp.float32),
        log_amplitude=jnp.array(0.5, dtype=jnp.float32),
        spectral_weights=jnp.array([weight], dtype=jnp.float32),
        spectral_means=jnp.array([mean], dtype=jnp.float32),
        spectral_scales=jnp.array([0.5], dtype=jnp.float32),
        time_decay=jnp.array(0.1, dtype=jnp.float32),
    )
    value = spectral_mixture_kernel(0.0, 0.5, 0.0, 0.0, params)
    assert jnp.isfinite(value)


@given(st.integers(min_value=1, max_value=32), st.integers(min_value=1, max_value=8))
def test_ring_buffer_never_exceeds_capacity(capacity, dim):
    rb = RingBuffer(capacity, dim)
    for i in range(capacity * 3):
        rb.push(np.full(dim, float(i), dtype=np.float64))
    out = rb.as_array()
    assert out.shape[0] <= capacity
    assert out.shape[1] == dim
    assert np.isfinite(out).all()


@given(
    st.lists(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=8),
)
def test_conformal_intervals_are_non_negative(values):
    residual = jnp.asarray(values, dtype=jnp.float32)
    state = NoiseEstimatorState(
        method="conformal",
        residual_buffer=jnp.zeros((8,), dtype=jnp.float32),
        buffer_ptr=jnp.array(0, dtype=jnp.int32),
        embedding=jnp.zeros((4,), dtype=jnp.float32),
        autocov_estimate=jnp.zeros((2,), dtype=jnp.float32),
        conformal_scores=jnp.zeros((8,), dtype=jnp.float32),
        drift_flag=jnp.array(False),
    )
    _, interval = conformal_update(state, residual, {"conformal_window": 8, "alpha": 0.1, "return_interval": True})
    assert float(interval) >= 0.0


def test_rollout_regression_has_no_nans_and_stable_covariance():
    init = PlantState(
        x=jnp.array([0.0, 0.3], dtype=jnp.float32),
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
    _, _, _, _, logs = rollout_system(init, n_steps=20, config=config, key=jnp.asarray([0, 0], dtype=jnp.uint32))
    assert jnp.isfinite(logs).all()
    assert logs.shape[1] >= 8
