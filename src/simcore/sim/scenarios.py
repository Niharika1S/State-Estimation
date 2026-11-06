from __future__ import annotations

import jax.numpy as jnp


def drift_schedule(t, amplitude=0.2):
    return amplitude * jnp.sin(t / 10.0)


def regime_schedule(t, change_points):
    return jnp.asarray(change_points[int(t) % len(change_points)])
