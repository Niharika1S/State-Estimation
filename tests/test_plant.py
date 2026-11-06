import jax
import jax.numpy as jnp

from simcore.plant.dynamics import true_dynamics
from simcore.plant.noise import sample_process_noise


def test_true_dynamics_shape():
    x = jnp.array([0.1, 0.2], dtype=jnp.float32)
    y = true_dynamics(x, jnp.array([0.0], dtype=jnp.float32), 0, {"gravity": 9.81, "length": 1.0, "damping": 0.05})
    assert y.shape == (2,)


def test_noise_sampling_shape():
    key = jax.random.PRNGKey(0)
    noise = sample_process_noise(key, 0, {"noise_distribution": "gaussian", "noise_scale": 0.1, "state_dim": 2})
    assert noise.shape == (2,)


def test_noise_moments_match_gaussian_expectations():
    key = jax.random.PRNGKey(9)
    keys = jax.random.split(key, 4000)
    noise = jax.vmap(
        lambda k: sample_process_noise(k, 0, {"noise_distribution": "gaussian", "noise_scale": 0.5, "state_dim": 2})
    )(keys)
    assert abs(noise.mean()) < 0.08
    assert abs(noise.var().mean() - 0.25) < 0.08
