from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class PlantState:
    x: jnp.ndarray
    t: jnp.ndarray
    regime: jnp.ndarray


@struct.dataclass
class Observation:
    y: jnp.ndarray
    t: jnp.ndarray


@struct.dataclass
class KernelParams:
    log_lengthscale: jnp.ndarray
    log_amplitude: jnp.ndarray
    spectral_weights: jnp.ndarray
    spectral_means: jnp.ndarray
    spectral_scales: jnp.ndarray
    time_decay: jnp.ndarray


@struct.dataclass
class InducingState:
    Z: jnp.ndarray
    m: jnp.ndarray
    S: jnp.ndarray


@struct.dataclass
class NoiseEstimatorState:
    method: str = struct.field(pytree_node=False)
    residual_buffer: jnp.ndarray = None
    buffer_ptr: jnp.ndarray = None
    embedding: jnp.ndarray = None
    autocov_estimate: jnp.ndarray = None
    conformal_scores: jnp.ndarray = None
    drift_flag: jnp.ndarray = None


@struct.dataclass
class BeliefState:
    mean: jnp.ndarray
    cov: jnp.ndarray
    noise_summary: jnp.ndarray
    drift_flag: jnp.ndarray


@struct.dataclass
class AgentState:
    belief: BeliefState
    policy_params: Dict[str, Any] = struct.field(pytree_node=False)

