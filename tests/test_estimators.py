import jax.numpy as jnp

from simcore.estimators.als import als_update
from simcore.estimators.conformal import conformal_update
from simcore.estimators.correntropy import correntropy_update, mckf_step
from simcore.estimators.mmd_embedding import adaptive_threshold, mmd_update
from simcore.types import NoiseEstimatorState


def make_state(buffer_size=8):
    return NoiseEstimatorState(
        method="test",
        residual_buffer=jnp.zeros((buffer_size,), dtype=jnp.float32),
        buffer_ptr=jnp.array(0, dtype=jnp.int32),
        embedding=jnp.array(0.0, dtype=jnp.float32),
        autocov_estimate=jnp.zeros((2,), dtype=jnp.float32),
        conformal_scores=jnp.zeros((buffer_size,), dtype=jnp.float32),
        drift_flag=jnp.array(False),
    )


def test_estimators_update_state():
    res = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    als = als_update(make_state(), res, {"ring_buffer_size": 8, "als_lags": 2})
    assert als.method == "test"
    assert als.autocov_estimate.shape == (2,)

    mmd = mmd_update(make_state(), res, None, {"mmd_dim": 8, "mmd_threshold": 0.1})
    assert mmd.method == "test"
    assert jnp.asarray(mmd.drift_flag).dtype == jnp.bool_

    conf = conformal_update(make_state(), res, {"conformal_window": 8, "alpha": 0.1, "return_interval": False})
    assert conf.method == "test"
    assert conf.conformal_scores.shape == (8,)

    corr = correntropy_update(make_state(), res, {"sigma_kernel": 1.0, "correntropy_threshold": 0.7})
    assert corr.method == "test"
    assert jnp.asarray(corr.drift_flag).dtype == jnp.bool_

    thresh = adaptive_threshold({"mmd_threshold": 0.5}, buffer_size=8)
    assert thresh > 0

    filt = mckf_step({"x": jnp.array([0.0], dtype=jnp.float32), "P": jnp.array([[1.0]], dtype=jnp.float32)}, None, jnp.array([0.4], dtype=jnp.float32), 0.5, {"R": 0.1})
    assert filt["x"].shape == (1,)
    assert filt["P"].shape == (1, 1)
