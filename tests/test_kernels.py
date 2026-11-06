import jax.numpy as jnp

from simcore.kernels.features import rff_features
from simcore.kernels.stationary import rbf_kernel
from simcore.kernels.time_varying import spectral_mixture_kernel
from simcore.models.kernel_gp import KernelGPModel
from simcore.types import InducingState, KernelParams


def test_rbf_kernel_is_positive():
    x = jnp.array([0.0, 1.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0], dtype=jnp.float32)
    val = rbf_kernel(x, y, lengthscale=1.0, amplitude=1.0)
    assert val.shape == (2,)
    assert jnp.all(val >= 0.0)


def test_time_varying_kernel_runs():
    x = jnp.array([0.0, 0.5], dtype=jnp.float32)
    params = type("Params", (), {"spectral_weights": jnp.array([1.0], dtype=jnp.float32), "spectral_scales": jnp.array([1.0], dtype=jnp.float32), "spectral_means": jnp.array([1.0], dtype=jnp.float32), "time_decay": jnp.array(0.1, dtype=jnp.float32), "log_amplitude": jnp.array(1.0, dtype=jnp.float32)})()
    val = spectral_mixture_kernel(x[0], x[1], 1.0, 0.0, params)
    assert jnp.isfinite(val)


def test_rff_and_gp_predict_shapes():
    x = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    omega = jnp.ones((4, 2), dtype=jnp.float32)
    features = rff_features(x, omega, jnp.zeros((4,), dtype=jnp.float32), lengthscale=1.0)
    assert features.shape == (2, 4)

    params = KernelParams(
        log_lengthscale=jnp.array(0.0, dtype=jnp.float32),
        log_amplitude=jnp.array(1.0, dtype=jnp.float32),
        spectral_weights=jnp.array([1.0], dtype=jnp.float32),
        spectral_means=jnp.array([1.0], dtype=jnp.float32),
        spectral_scales=jnp.array([1.0], dtype=jnp.float32),
        time_decay=jnp.array(0.1, dtype=jnp.float32),
    )
    inducing = InducingState(
        Z=jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32),
        m=jnp.array([0.1, 0.2], dtype=jnp.float32),
        S=jnp.array([1.0, 1.0], dtype=jnp.float32),
    )
    pred = KernelGPModel.predict(x, inducing, params)
    assert pred.shape == (2,)

    updated = KernelGPModel.update_kernel_model(inducing, params, jnp.array([0.5, 0.0]), jnp.array(0.3, dtype=jnp.float32), forgetting_factor=0.9, budget=8)
    assert updated.Z.shape[0] >= 2
