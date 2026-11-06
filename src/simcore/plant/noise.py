from __future__ import annotations

import jax
import jax.numpy as jnp


def _make_noise_key(key):
    if hasattr(key, "dtype") and key.dtype != jnp.uint32:
        return jnp.asarray(key, dtype=jnp.uint32)
    return key


def sample_process_noise(key, regime, config):
    """Sample process noise from Gaussian, Laplace, Student-t, or a bimodal mixture."""
    key = _make_noise_key(key)
    dist_name = config.get("noise_distribution", "gaussian")
    scale = jnp.asarray(config.get("noise_scale", 0.05), dtype=jnp.float32)
    dim = int(config.get("state_dim", 2))
    if dist_name in {"gaussian", "normal"}:
        return jax.random.normal(key, shape=(dim,), dtype=jnp.float32) * scale
    if dist_name == "laplace":
        return jax.random.laplace(key, shape=(dim,), dtype=jnp.float32) * scale
    if dist_name == "student_t":
        df = float(config.get("student_t_df", 3.0))
        return jax.random.t(key, df=df, shape=(dim,), dtype=jnp.float32) * scale
    if dist_name == "mixture":
        key_a, key_b = jax.random.split(key)
        a = jax.random.normal(key_a, shape=(dim,), dtype=jnp.float32) * scale
        b = jax.random.normal(key_b, shape=(dim,), dtype=jnp.float32) * (scale * 2.0)
        mix = jnp.where(jnp.asarray(regime) > 0, a, b)
        return mix
    if dist_name == "regime_switch":
        mode = config.get("regime_noise_modes", ["gaussian", "laplace"])
        this_mode = mode[int(jnp.asarray(regime)) % len(mode)]
        return sample_process_noise(key, regime, {**config, "noise_distribution": this_mode})
    return jax.random.normal(key, shape=(dim,), dtype=jnp.float32) * scale


def regime_transition(state_regime, t, config, key=None):
    """Return the next regime index following a schedule or a simple Markov switch."""
    schedule = config.get("regime_schedule", [])
    if schedule:
        idx = int(jnp.asarray(t)) % len(schedule)
        return jnp.asarray(schedule[idx], dtype=jnp.asarray(state_regime).dtype)
    transition = config.get("regime_transition", "schedule")
    if transition == "markov":
        if key is None:
            key = jax.random.PRNGKey(0)
        p = float(config.get("regime_change_probability", 0.1))
        switch = jax.random.bernoulli(key, p=p)
        next_regime = jnp.where(switch, 1 - jnp.asarray(state_regime), jnp.asarray(state_regime))
        return next_regime.astype(jnp.asarray(state_regime).dtype)
    return jnp.asarray(state_regime)
