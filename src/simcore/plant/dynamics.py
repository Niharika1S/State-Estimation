from __future__ import annotations

import jax
import jax.numpy as jnp

from simcore.plant.noise import sample_process_noise, regime_transition


def pendulum_dynamics(x, action, t, config):
    """Pendulum dynamics with a time-varying effective length to create drift."""
    x = jnp.asarray(x, dtype=jnp.float32)
    theta = x[..., 0]
    omega = x[..., 1]
    gravity = config.get("gravity", 9.81)
    length = config.get("length", 1.0)
    drift_amp = config.get("drift_amplitude", 0.0)
    drift_period = max(float(config.get("drift_period", 20.0)), 1e-6)
    varying_length = length * (1.0 + drift_amp * jnp.sin(jnp.asarray(t) / drift_period))
    damping = config.get("damping", 0.05)
    mass = config.get("mass", 1.0)
    control = jnp.asarray(action, dtype=x.dtype)
    if control.shape == ():
        control = jnp.full_like(theta, control)
    elif control.shape == x.shape:
        control = control[..., 0]
    elif control.shape != theta.shape:
        if theta.shape == () and control.shape == (1,):
            control = control[0]
        else:
            control = jnp.broadcast_to(control, theta.shape)
    angular_accel = -(gravity / varying_length) * jnp.sin(theta) - damping * omega + control / mass
    return jnp.stack([omega, angular_accel], axis=-1)


def rc_dynamics(x, action, t, config):
    """A simple RC battery/thermal model used as a non-oscillatory alternative."""
    x = jnp.asarray(x, dtype=jnp.float32)
    voltage = x[..., 0]
    resistance = config.get("resistance", 1.0) * (1.0 + config.get("aging_rate", 0.0) * jnp.asarray(t))
    capacitance = config.get("capacitance", 1.0)
    current = jnp.asarray(action, dtype=x.dtype)
    if current.shape == ():
        current = jnp.full_like(voltage, current)
    elif current.shape == x.shape:
        current = current[..., 0]
    elif current.shape != voltage.shape:
        current = jnp.broadcast_to(current, voltage.shape)
    dv_dt = (current - voltage / resistance) / capacitance
    return jnp.stack([dv_dt], axis=-1)


def true_dynamics(x, action, t, config):
    model = config.get("dynamics_model", "pendulum")
    if model == "rc":
        return rc_dynamics(x, action, t, config)
    return pendulum_dynamics(x, action, t, config)


def observation_model(x, key, config):
    """Return a noisy observation with independent measurement noise."""
    noise_scale = config.get("measurement_noise_scale", 0.02)
    dist_name = config.get("measurement_noise_distribution", config.get("noise_distribution", "gaussian"))
    noise_config = {**config, "noise_distribution": dist_name, "noise_scale": noise_scale, "state_dim": x.shape[-1]}
    return x + sample_process_noise(key, 0, noise_config)


def plant_step(state, action, key, config):
    """Single-step plant update: deterministic dynamics + process noise + observation noise."""
    key_dyn, key_noise, key_obs = jax.random.split(key, 3)
    x_next_mean = true_dynamics(state.x, action, state.t, config)
    w = sample_process_noise(key_noise, state.regime, config)
    x_next = x_next_mean + w
    regime_next = regime_transition(state.regime, state.t, config, key_dyn)
    y = observation_model(x_next, key_obs, config)
    next_state = state.__class__(
        x=x_next,
        t=state.t + 1,
        regime=regime_next,
    )
    obs = type("Observation", (), {"y": y, "t": state.t + 1})()
    return next_state, obs


def pendulum_step(state, action, key, config):
    """Compatibility wrapper around the full plant update."""
    return plant_step(state, action, key, config)
