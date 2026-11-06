from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from simcore.types import InducingState


class KernelGPModel:
    """A compact JAX-compiled kernel GP model using inducing points."""

    @staticmethod
    @jax.jit
    def predict(x_test, inducing, params):
        x_test = jnp.asarray(x_test)
        Z = inducing.Z
        if Z.ndim == 1:
            Z = Z[None, :]
        x_test = x_test.reshape((-1, Z.shape[-1]))
        distances = jnp.sum((x_test[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        lengthscale = jnp.maximum(jnp.exp(params.log_lengthscale).mean(), 1e-6)
        kernel = jnp.exp(-0.5 * distances / (lengthscale**2))
        denom = jnp.sum(kernel, axis=-1)
        return (kernel @ inducing.m) / (denom + 1e-6)

    @staticmethod
    @partial(jax.jit, static_argnames=("budget", "novelty_threshold"))
    def update_kernel_model(inducing, params, x_new, y_new, forgetting_factor=0.98, key=None, budget=8, novelty_threshold=0.1):
        del key
        x_new = jnp.asarray(x_new)
        y_new = jnp.asarray(y_new)
        if x_new.ndim == 1:
            x_new = x_new[None, :]
        if y_new.ndim == 0:
            y_new = y_new[None]

        if inducing.Z.shape[0] == 0:
            new_Z = x_new
            new_m = y_new
            new_S = jnp.ones((1,), dtype=inducing.S.dtype)
            return InducingState(Z=new_Z, m=new_m, S=new_S)

        base_m = forgetting_factor * inducing.m
        base_Z = inducing.Z
        base_S = inducing.S
        new_point = jnp.concatenate([base_Z, x_new], axis=0)
        new_m = jnp.concatenate([base_m, y_new], axis=0)
        new_S = jnp.concatenate([base_S, jnp.ones((1,), dtype=base_S.dtype)], axis=0)

        if new_point.shape[0] > budget:
            new_Z = new_point[-budget:]
            new_m = new_m[-budget:]
            new_S = new_S[-budget:]
        else:
            new_Z = new_point

        return InducingState(Z=new_Z, m=new_m, S=new_S)
