# simcore

A compact simulation core for JAX-first plant, kernel, and estimator experiments.

## Overview

`simcore` is a research-oriented simulation package for exploring how different plant models, noise regimes, estimators, kernels, and control policies behave under uncertainty and drift. It is designed for rapid experimentation with state-estimation workflows in a JAX-first environment.

The project includes:

- plant dynamics and noise models
- estimator implementations for comparison
- kernel-based modeling components
- agent-style belief/control logic
- orchestration utilities for ablations and workflow automation
- a FastAPI-based API for session management and experiment execution

## Project structure

- `src/simcore/plant/` — plant and dynamics models
- `src/simcore/estimators/` — estimator implementations
- `src/simcore/kernels/` — stationary and time-varying kernels
- `src/simcore/agent/` — belief and policy logic
- `src/simcore/orchestration.py` — registry, ablation grid, and workflow specs
- `src/simcore/api.py` — FastAPI application and session API
- `tests/` — validation and integration tests

## Installation

```bash
pip install -e .
```

## Quick usage

```bash
pytest
```

You can also use the API layer from `src/simcore/api.py` to start sessions and run experiment flows through the HTTP interface.

## Typical use cases

- compare estimator robustness under different noise distributions
- evaluate drift detection and recovery behavior
- benchmark kernel choices against a shared plant model
- automate ablations across noise regimes and estimator configurations

## Status

This repo is best understood as a research prototype and simulation experimentation toolkit rather than a polished production application. It is a strong foundation for experiments, benchmarking, and portfolio/demo work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
