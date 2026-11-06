# Build Plan: Distribution-Free Noise Estimation + Time-Varying Kernel Dynamics + State-Aware Agent (JAX / Cython)

This document is a step-by-step algorithmic plan for a coding agent to implement the simulation
codebase described earlier. It specifies module boundaries, data structures, algorithms (as
pseudocode), numerical considerations, and a testing plan. Follow the phases in order; each phase
should produce a working, tested increment before moving to the next.

---

## 0. Design principles for the agent to follow

1. **JAX is the numerical core.** All hot-path math (kernel evaluation, filtering, gradients,
   rollouts) is written in JAX, using `jit`, `vmap`, and `lax.scan` — never Python `for` loops over
   timesteps in the hot path.
2. **Cython is only for glue, not math.** Use Cython exclusively for (a) fast typed data ingestion
   /serialization (e.g., parsing large logged trajectories, ring-buffer bookkeeping for online
   estimators), and (b) any CPU-bound pre/post-processing that isn't naturally vectorizable in JAX.
   Do not reimplement kernel or filtering math in Cython — JAX's XLA compilation will already
   outperform hand-written Cython for that.
3. **Everything stateful is an explicit pytree.** No hidden Python object state inside JIT'd
   functions. Model parameters, filter state, kernel state, and agent belief state are all
   registered JAX pytrees (via `flax.struct.dataclass` or `chex.dataclass`) so they can pass
   through `jit`/`scan`/`vmap` cleanly.
4. **Separate "true simulator" from "agent's model."** The plant (ground truth dynamics + injected
   noise) must be a separate module from the agent's kernel model and noise estimator, so the agent
   genuinely only sees noisy observations — this is what makes the noise estimation experiment valid.
5. **Determinism and seeding.** Every stochastic operation takes an explicit `jax.random.PRNGKey`,
   split explicitly, never a global RNG state. Log the seed for every run.

---

## Phase 1 — Repository scaffolding

```
project/
├── pyproject.toml
├── src/
│   └── simcore/
│       ├── __init__.py
│       ├── types.py          # pytree dataclasses (state, params, config)
│       ├── plant/
│       │   ├── dynamics.py   # ground-truth ODE/discrete dynamics
│       │   └── noise.py      # noise generators (Gaussian, Student-t, mixture, regime-switch)
│       ├── kernels/
│       │   ├── stationary.py
│       │   ├── time_varying.py   # spectral-mixture / product kernel with drifting params
│       │   └── features.py       # random Fourier features, Nystrom approx
│       ├── models/
│       │   ├── kernel_gp.py      # sparse/online GP-SSM dynamics model
│       │   └── koopman.py        # optional: kernel-DMD alternative
│       ├── estimators/
│       │   ├── als.py            # autocovariance least squares
│       │   ├── mmd_embedding.py  # kernel mean embedding + MMD noise tracker
│       │   ├── conformal.py      # split/online conformal prediction
│       │   └── correntropy.py    # maximum correntropy Kalman filter
│       ├── agent/
│       │   ├── belief.py         # belief-state construction
│       │   ├── policy.py         # MPC / model-based RL policy
│       │   └── controller_llm.py # optional: high-level supervisory logic
│       ├── sim/
│       │   ├── loop.py           # lax.scan-based rollout
│       │   └── scenarios.py      # drift schedules, change-point injection
│       ├── fastio/               # Cython package
│       │   ├── ring_buffer.pyx
│       │   ├── trajectory_reader.pyx
│       │   └── __init__.py
│       └── eval/
│           ├── metrics.py
│           └── ablations.py
├── tests/
│   ├── test_kernels.py
│   ├── test_estimators.py
│   ├── test_plant.py
│   ├── test_agent.py
│   └── test_integration.py
├── benchmarks/
│   └── bench_hotpath.py
└── configs/
    ├── pendulum_drift.yaml
    └── noise_regimes.yaml
```

**Tooling**
- `pyproject.toml` with `jax`, `jaxlib`, `flax`, `chex`, `optax`, `numpy`, `scipy`, `distrax`
  (for ground-truth noise sampling only, never inside the estimator), `pytest`, `hypothesis`.
- Cython build via `setuptools` + `Cython.Build.cythonize`, compiled as an extension module in
  `pyproject.toml` `[build-system]`.
- `pre-commit` with `ruff`/`black` and `mypy` (loose mode is fine for JAX code).

**Deliverable of Phase 1:** repo builds, `pip install -e .` succeeds, Cython extension compiles,
CI runs an empty test suite green.

---

## Phase 2 — Core types (pytrees)

Define once, reused everywhere. Pseudocode (using `flax.struct.dataclass`):

```python
@struct.dataclass
class PlantState:
    x: jnp.ndarray            # true latent state
    t: jnp.ndarray            # scalar time index
    regime: jnp.ndarray       # int, which noise/dynamics regime is active

@struct.dataclass
class Observation:
    y: jnp.ndarray
    t: jnp.ndarray

@struct.dataclass
class KernelParams:
    log_lengthscale: jnp.ndarray      # per-dimension, possibly time-indexed
    log_amplitude: jnp.ndarray
    spectral_weights: jnp.ndarray     # for spectral-mixture kernel
    spectral_means: jnp.ndarray
    spectral_scales: jnp.ndarray
    time_decay: jnp.ndarray           # controls how fast kernel forgets old data

@struct.dataclass
class InducingState:
    Z: jnp.ndarray             # inducing input locations
    m: jnp.ndarray             # variational mean
    S: jnp.ndarray             # variational covariance (or its Cholesky factor)

@struct.dataclass
class NoiseEstimatorState:
    method: str                        # static, not traced
    residual_buffer: jnp.ndarray       # fixed-size ring buffer, filled via Cython writer
    buffer_ptr: jnp.ndarray
    embedding: jnp.ndarray             # for MMD: kernel mean embedding coefficients
    autocov_estimate: jnp.ndarray      # for ALS
    conformal_scores: jnp.ndarray      # for conformal
    drift_flag: jnp.ndarray            # bool

@struct.dataclass
class BeliefState:
    mean: jnp.ndarray
    cov: jnp.ndarray
    noise_summary: jnp.ndarray   # compact features derived from NoiseEstimatorState
    drift_flag: jnp.ndarray

@struct.dataclass
class AgentState:
    belief: BeliefState
    policy_params: dict          # optax-compatible pytree
```

**Deliverable:** `types.py` with unit tests confirming every dataclass round-trips through
`jax.jit(lambda s: s)` and `jax.tree_util.tree_map`.

---

## Phase 3 — Plant (ground-truth simulator)

**Algorithm: `plant_step`**
```
function plant_step(state: PlantState, action, key, config) -> (PlantState, Observation):
    key_dyn, key_noise = split(key)
    x_next_mean = true_dynamics(state.x, action, state.t, config)   # e.g. pendulum ODE, RK4 step
    w = sample_process_noise(key_noise, state.regime, config)      # Gaussian / Student-t / mixture
    x_next = x_next_mean + w
    regime_next = regime_transition(state.regime, state.t, config) # scheduled or Markov switch
    y = observation_model(x_next, key_obs, config)                 # add measurement noise separately
    return PlantState(x_next, state.t + 1, regime_next), Observation(y, state.t + 1)
```

- Implement `sample_process_noise` for at least: Gaussian, Laplace, Student-t (low dof for heavy
  tails), Gaussian mixture (bimodal), and a regime-switching wrapper that swaps parameters at
  scheduled or randomly drawn change points.
- `true_dynamics` should be swappable: pendulum/cart-pole with drifting mass/length, or a
  battery/thermal RC model with aging resistance (better fit for "computer engineering").
- Use `lax.scan` to roll out a full trajectory given a policy or a canned action sequence; wrap the
  `plant_step` as the scan body.

**Testing:** verify statistical moments of `sample_process_noise` outputs match closed-form
expectations (mean, variance, kurtosis where defined) via Monte Carlo over many keys.

---

## Phase 4 — Time-varying kernel

**Algorithm: spectral-mixture kernel with drifting weights**
```
function kernel(x, x', t, t', params: KernelParams):
    dx = x - x'
    dt = t - t'
    spatial = sum_q params.spectral_weights[q] *
              exp(-2 pi^2 * dx^2 * params.spectral_scales[q]) *
              cos(2 pi * dx * params.spectral_means[q])
    temporal_decay = exp(-params.time_decay * dt^2)   # down-weights stale cross-time correlations
    return spatial * temporal_decay
```

**Random Fourier Feature approximation (for O(n) updates instead of O(n^2/n^3)):**
```
function rff_features(x, t, omega, b, params):
    # omega: sampled frequencies (fixed per drift-window, resampled on regime change)
    z = sqrt(2 / D) * cos(omega @ concat(x, t) * params.log_lengthscale_transform + b)
    return z    # feature map, so kernel(x1,x2) ≈ z(x1) . z(x2)
```

**Online kernel model update (sparse GP / kernel RLS with forgetting):**
```
function update_kernel_model(inducing: InducingState, kparams: KernelParams,
                              new_x, new_y, forgetting_factor, key) -> InducingState:
    # Option A: streaming sparse variational GP (SVGP) natural-gradient step
    #   - compute predictive mean/var at new_x using current (Z, m, S)
    #   - compute ELBO gradient w.r.t. (m, S) and kparams
    #   - take one optax step (e.g. Adam) — this is the "learns as it drifts" mechanism
    # Option B: kernel recursive least squares (KRLS) with exponential forgetting
    #   - maintain inverse Gram matrix K^-1 recursively (Sherman-Morrison update)
    #   - multiply old statistics by forgetting_factor < 1 each step to downweight stale data
    #   - periodically prune/replace inducing points with lowest "novelty" score
    return updated_inducing_state

- This buffer lives **outside** JAX's traced graph: it's fed by a plain Python/NumPy callback after
  each `lax.scan` chunk (e.g., scan in chunks of 100 steps, then push the chunk's residuals into
  the Cython buffer for logging/offline analysis), not inside the per-step JIT'd hot path.
- Similarly write a `trajecto
```

- Pick **Option A (SVGP with natural gradients)** as the primary path — it's the more defensible
  research contribution and composes cleanly with `optax`.
- Implement inducing point management: add a new inducing point when a novelty/leverage score
  exceeds a threshold; evict the lowest-utility one to keep a fixed budget (important for O(1)
  per-step cost).
- Resample RFF frequencies `omega` whenever `drift_flag` fires (bridges the kernel and noise
  modules — see Phase 6).

**Deliverable:** `kernels/time_varying.py`, `models/kernel_gp.py`, jitted `update_kernel_model` and
`predict` functions, benchmarked for per-step wall-clock time.

---

## Phase 5 — Distribution-free noise estimators

Implement all four; wire ALS as baseline, MMD-embedding as primary, conformal as the
uncertainty-quantification layer used by the agent, correntropy as a robustness ablation.

### 5.1 Autocovariance Least Squares (ALS) — baseline
```
function als_update(state: NoiseEstimatorState, residual, config):
    push residual into ring buffer (Cython-backed, see Phase 8)
    compute sample autocovariances C_0..C_L from buffer
    solve linear least-squares system A @ vec(Q, R) = vec(C) for process/measurement noise covariances
    return updated_state with new autocov_estimate
```

### 5.2 Kernel mean embedding + MMD tracker — primary
```
function mmd_update(state: NoiseEstimatorState, residual, kparams_noise, config):
    push residual into buffer
    mu_hat = mean_k( phi(residual_i) for residual_i in buffer )   # empirical kernel mean embedding
    mmd2 = || mu_hat - state.embedding ||_H^2                     # compare to previous embedding
    drift_flag = mmd2 > adaptive_threshold(config)                 # e.g. via permutation test or
                                                                     # Hoeffding-type bound on MMD
    embedding_new = (1 - alpha) * state.embedding + alpha * mu_hat  # exponential update
    return NoiseEstimatorState(embedding=embedding_new, drift_flag=drift_flag, ...)
```
- Use RFF or Nystrom features for `phi` so the embedding is a finite vector, not a kernel matrix —
  required for O(1) updates and JIT compatibility.
- `adaptive_threshold`: implement both (a) a fixed quantile from a bootstrap/permutation
  calibration run offline, and (b) an online version using Hoeffding/McDiarmid bound on MMD given
  buffer size — expose both, default to (a) for the main experiments, note (b) as a stretch goal.

### 5.3 Conformal prediction — uncertainty bands, feeds the agent
```
function conformal_update(state: NoiseEstimatorState, residual, config):
    nonconformity_score = |residual| or Mahalanobis-style score under current kernel model
    push score into state.conformal_scores (ring buffer, size = calibration window)
    quantile_level = ceil((1 - alpha) * (n + 1)) / n
    interval_halfwidth = quantile(state.conformal_scores, quantile_level)
    return updated_state, interval_halfwidth
```
- Implement the **online/adaptive conformal** variant (ACI — Adaptive Conformal Inference, Gibbs &
  Candès) so `alpha` (miscoverage target) is itself adapted based on recent coverage errors —
  this is what makes it work under non-stationary drift instead of plain split conformal.

### 5.4 Maximum Correntropy Kalman Filter — robustness ablation
```
function mckf_step(state, kparams, residual, sigma_kernel, config):
    # standard Kalman predict step using current kernel-model Jacobian/linearization
    innovation = residual
    correntropy_weight = exp(-innovation^2 / (2 * sigma_kernel^2))   # downweights outliers
    K_gain = P_pred @ H^T @ inv(H @ P_pred @ H^T + R / correntropy_weight)
    x_update = x_pred + K_gain @ innovation
    P_update = (I - K_gain @ H) @ P_pred
    return updated filter state
```

**Deliverable:** all four estimators pass a synthetic benchmark: generate residuals from known
Gaussian/Student-t/bimodal/regime-switching processes, confirm (a) ALS and MMD track the true
covariance/embedding within tolerance, (b) conformal empirical coverage is within a few points of
nominal `1-alpha` even across an injected change point, (c) MCKF has materially lower state RMSE
than a plain EKF under heavy-tailed/impulsive noise.

---

## Phase 6 — Belief-state construction (bridges kernel model + noise estimators)

```
function build_belief(kernel_pred_mean, kernel_pred_cov, noise_state: NoiseEstimatorState,
                       conformal_halfwidth) -> BeliefState:
    noise_summary = concat(
        flatten(noise_state.embedding)[:K],   # truncated/PCA'd embedding, fixed size
        noise_state.autocov_estimate.flatten(),
        conformal_halfwidth,
    )
    return BeliefState(
        mean=kernel_pred_mean,
        cov=kernel_pred_cov + diag(conformal_halfwidth**2),  # inflate belief by uncertainty
        noise_summary=noise_summary,
        drift_flag=noise_state.drift_flag,
    )
```

This is the single function that decides *what the agent is allowed to see*. For the ablation study
(state-aware vs naive), implement a flag `expose_noise_info: bool` that zeroes out `noise_summary`
and `drift_flag` when False, so the same downstream policy code can be run either way.

---

## Phase 7 — State-aware agent

**Policy architecture (pick MPC as primary, RSSM/latent policy as stretch goal):**

```
function mpc_policy(belief: BeliefState, kernel_model, horizon, n_candidates, key, config):
    # sample n_candidates action sequences (random shooting or CEM)
    candidates = sample_action_sequences(key, horizon, n_candidates, config)
    # roll out each candidate through the *kernel model* (not the true plant!), propagating belief
    costs = vmap(lambda seq: rollout_cost(belief, seq, kernel_model, config))(candidates)
    # penalize cost by predicted uncertainty growth (uncertainty-averse planning)
    costs_adj = costs + config.risk_weight * uncertainty_penalty(belief, candidates, kernel_model)
    best_seq = candidates[argmin(costs_adj)]
    return best_seq[0]   # first action, receding horizon
```

- `rollout_cost` uses `lax.scan` over the horizon, propagating belief mean/cov through the kernel
  model's predictive equations (moment matching) plus the task cost (e.g. LQR-style quadratic cost
  to a target state).
- **CEM (cross-entropy method)** refinement loop as an upgrade over pure random shooting: iterate
  resampling candidates from a Gaussian fit to the elite fraction, 3-5 iterations, still fully
  `vmap`-able.
- If pursuing the RSSM/Dreamer-style stretch goal: replace `kernel_model` rollouts with a learned
  latent recurrent model trained via `optax`, and the policy becomes an actor network trained via
  policy-gradient (REINFORCE with baseline, or a simple actor-critic) against **imagined** rollouts.
  This is a larger sub-project — implement it only after the MPC baseline is solid, and treat it as
  Phase 10 (stretch).

**Deliverable:** `agent/policy.py`, jitted `mpc_policy`, benchmarked to confirm it produces
reasonable actions on the pendulum/cart-pole testbed under nominal (no drift) conditions before
adding noise-awareness.

---

## Phase 8 — Cython fast I/O layer

Use Cython only where profiling in Phase 9 shows a genuine bottleneck outside JIT — typically:
ring-buffer writes for the noise estimator history, and reading/parsing large logged trajectory
files for offline analysis.

```
# ring_buffer.pyx
cimport cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class RingBuffer:
    cdef cnp.float64_t[:, :] buf
    cdef Py_ssize_t capacity, ptr, size, dim

    def __cinit__(self, Py_ssize_t capacity, Py_ssize_t dim):
        self.buf = np.zeros((capacity, dim), dtype=np.float64)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.dim = dim

    cpdef void push(self, cnp.float64_t[:] row):
        cdef Py_ssize_t i
        for i in range(self.dim):
            self.buf[self.ptr, i] = row[i]
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    cpdef cnp.ndarray[cnp.float64_t, ndim=2] snapshot(self):
        return np.asarray(self.buf[:self.size])
```

- This buffer lives **outside** JAX's traced graph: it's fed by a plain Python/NumPy callback after
  each `lax.scan` chunk (e.g., scan in chunks of 100 steps, then push the chunk's residuals into
  the Cython buffer for logging/offline analysis), not inside the per-step JIT'd hot path.
- Similarly write a `trajectory_reader.pyx` for fast parsing of large CSV/binary trajectory logs
  used in the ablation-runner (Phase 11), if NumPy's own loaders prove too slow at scale.
- Build with `cythonize(["src/simcore/fastio/*.pyx"], language_level=3)`, typed `.pxd` headers for
  anything imported cross-module.

**Deliverable:** Cython extension builds, has its own `pytest` suite (push/pop correctness, wraps
correctly at capacity), and a micro-benchmark showing it beats an equivalent pure-Python ring
buffer by a clear margin — only keep it if the benchmark justifies the complexity; otherwise drop
Cython entirely and use NumPy, per the "Cython only if necessary" instruction.

---

## Phase 9 — Full simulation loop + profiling

```
function rollout(initial_plant_state, initial_agent_state, n_steps, config, key):
    def scan_body(carry, _):
        plant_state, agent_state, kernel_model_state, noise_state, key = carry
        key, k1, k2 = split(key, 3)

        action = mpc_policy(agent_state.belief, kernel_model_state, ...)
        plant_state, obs = plant_step(plant_state, action, k1, config)

        residual = obs.y - kernel_predict_mean(kernel_model_state, plant_state.t)
        noise_state = mmd_update(noise_state, residual, ...)          # + als/conformal/mckf as configured
        kernel_model_state = update_kernel_model(kernel_model_state, obs, ..., key=k2)

        belief = build_belief(kernel_pred_mean, kernel_pred_cov, noise_state, conformal_halfwidth)
        agent_state = AgentState(belief=belief, policy_params=agent_state.policy_params)

        log_row = make_log_row(plant_state, obs, action, belief, noise_state)
        return (plant_state, agent_state, kernel_model_state, noise_state, key), log_row

    final_carry, logs = lax.scan(scan_body, init_carry, xs=None, length=n_steps)
    return logs
```

- Wrap the whole `rollout` in a single `jax.jit` where possible. If the Cython ring-buffer callback
  needs to run outside JIT, chunk the scan (e.g. `n_steps // chunk_size` calls to a jitted
  `rollout_chunk`, with the Python-level loop over chunks doing the Cython logging in between).
- Profile with `jax.profiler` / `tensorboard` trace viewer to find the actual bottleneck before
  writing any Cython — confirm the design principle in step 0.2.

**Deliverable:** end-to-end rollout runs on the pendulum/battery testbed, produces a log DataFrame,
and a smoke-test notebook/script plotting true state vs. estimated belief vs. noise-estimate error
over time.

---

## Phase 10 — Testing & validation plan

1. **Unit tests** (per Phase 2–8 deliverables above): kernel positive-definiteness, estimator
   convergence on synthetic data with known ground truth, pytree round-tripping through `jit`.
2. **Property-based tests** (via `hypothesis`): random valid `KernelParams` never produce NaNs;
   `RingBuffer` never exceeds capacity; conformal intervals are always non-negative.
3. **Integration test:** run a short (e.g. 200-step) rollout end-to-end and assert no NaNs, belief
   covariance stays positive-definite, and drift-flag fires within N steps of an injected change
   point (regression test against a fixed seed).
4. **Ablation study (the actual research experiments), via `eval/ablations.py`:**
   - Grid over `{noise_type} x {drift_schedule} x {estimator_method} x {expose_noise_info: bool}`.
   - Metrics collected per run: state RMSE, predictive NLL, conformal empirical coverage, MMD
     between estimated and true noise embedding (computed against an oracle using known true noise
     samples, for evaluation only — never fed to the agent), cumulative task cost, and
     drift-detection latency (steps between true change point and `drift_flag` firing).
   - Output a results table/CSV plus plots (RMSE vs. time with drift markers, coverage vs. nominal).
5. **Baselines to implement for comparison:** standard Kalman filter with fixed Q/R, EKF/UKF,
   particle filter, GP-SSM without time-varying kernel (fixed kernel), LSTM-based one-step
   predictor. These share the same `plant` and `rollout` scaffolding — only swap the model/estimator
   module — so implement them as alternative `kernel_model`/`noise_state` backends conforming to the
   same interface.

---

## Phase 11 — Stretch goals (only after Phases 1–10 are solid)

- RSSM/Dreamer-style latent world model + actor-critic policy trained on imagined rollouts.
- Theoretical regret/tracking bound for the online kernel update under bounded drift rate.
- Hardware deployment: export the sparse kernel model + MCKF to run on a Jetson/RPi, profile
  latency/memory, quantify the JAX→embedded gap (e.g. via `jax2tf` + TFLite, or a hand-written
  Cython inference path for the deployed model only).

---

## Suggested order of work for the agent (checklist)

- [ ] Phase 1: scaffolding, empty CI green
- [ ] Phase 2: pytree types + tests
- [ ] Phase 3: plant simulator + noise generators + tests
- [ ] Phase 4: time-varying kernel + online update + benchmark
- [ ] Phase 5: four noise estimators + synthetic validation
- [ ] Phase 6: belief-state construction + ablation flag
- [ ] Phase 7: MPC policy + sanity rollout
- [ ] Phase 9: full `lax.scan` loop, profiled
- [ ] Phase 8: Cython ring buffer, **only if profiling in Phase 9 justifies it**
- [ ] Phase 10: full ablation study + baselines + plots
- [ ] Phase 11: stretch goals, time permitting

Hand this document to the coding agent one phase at a time, requiring a passing test suite before
moving to the next phase — this keeps each increment reviewable and avoids a large, hard-to-debug
integration at the end.
