# n8n Orchestration Plan: Agentic Layer + Simulation Runner + Tool-Integration Testing

This assumes the Phase 1–10 Python/JAX codebase (`simcore`) is built and running as a service.
n8n sits on top as the **orchestration and agentic decision layer** — it never does the math, it
calls `simcore`'s HTTP API, reasons over the results, and drives experiments. This plan has four
workflows: (1) the core agent loop, (2) the tool/estimator integration test harness, (3) the
ablation/experiment runner, and (4) monitoring & alerting.

---

## 0. Prerequisite: expose `simcore` as an API

Before touching n8n, wrap the Phase 9 rollout into a stateful FastAPI service (one Python process,
not restarted per call — state persists in memory or Redis between ticks):

| Endpoint | Purpose | Called by |
|---|---|---|
| `POST /session/start` | Create a session_id, init plant + agent + kernel + estimator states, return session_id | Workflow 1, step 1 |
| `POST /session/{id}/step` | Advance one tick: plant_step → estimate → update_kernel → policy → belief. Returns full log row | Workflow 1, main loop |
| `POST /session/{id}/reconfigure` | Swap estimator method, toggle `expose_noise_info`, inject a drift event | Workflow 1 & 2 |
| `GET /session/{id}/metrics` | Cumulative RMSE, coverage, drift-detection latency so far | Workflow 1, 3 |
| `POST /session/{id}/end` | Finalize, flush logs, return summary | Workflow 1, 3 |
| `GET /health` and `GET /estimators` / `GET /kernels` / `GET /policies` | List available pluggable modules (for the tool-integration tests) | Workflow 2 |

This is the contract n8n's HTTP Request nodes will call against. Keep it stable — n8n workflows
should not need to change when you tweak internals of `simcore`.

---

## Workflow 1 — Core agentic simulation loop

**Purpose:** run one full episode, with an LLM-based supervisory agent making high-level decisions
(estimator/kernel/policy choices, drift response) on top of the numeric control loop.

### Nodes, in order

1. **Manual Trigger / Webhook** — start a run; accepts a JSON config (noise type, drift schedule,
   estimator method, `n_steps`, `expose_noise_info`).
2. **HTTP Request → `POST /session/start`** — pass the config, capture `session_id`.
3. **Set node** — initialize loop counters (`step = 0`, `max_steps`, `drift_events_seen = 0`) in
   workflow static data or a small Redis/Postgres row keyed by `session_id` (n8n executions don't
   share memory across iterations by default, so persist counters externally, not just in-node).
4. **Loop start — Split in Batches node** (or a `SplitInBatches` configured with `n_steps` dummy
   items) — this is the tick driver; one batch item = one simulation step.
5. **HTTP Request → `POST /session/{id}/step`** — advance the plant/agent one tick. Returns:
   `{observation, belief, noise_summary, drift_flag, action_taken, step_cost}`.
6. **IF node — `drift_flag == true`:**
   - **True branch:**
     a. **Set node** — increment `drift_events_seen`.
     b. **LLM node (Anthropic/OpenAI)** — the supervisory agent. System prompt gives it the
        current belief summary, noise_summary, recent cost trend, and the list of available
        estimator/kernel modules (from `GET /estimators`, fetched once at workflow start and
        cached). Ask it to choose one of a **fixed action set**: `{keep_estimator,
        switch_to_conformal_only, widen_risk_weight, request_kernel_retrain, no_op}`. Force
        structured JSON output (tool-call style) so the next node can branch on it reliably.
     c. **Switch node** on the LLM's chosen action → routes to the matching
        `POST /session/{id}/reconfigure` HTTP Request node variant.
     d. **NoOp/Merge back** into the main loop.
   - **False branch:** pass straight through (no reconfiguration).
7. **Function/Code node** — flatten the step's response + any reconfiguration event into one log
   row.
8. **Postgres (or Google Sheets) node — Insert** — append the log row (table: `sim_runs`, columns:
   `session_id, step, true_state, obs, belief_mean, belief_cov_trace, noise_summary_hash,
   drift_flag, action, llm_decision, step_cost, timestamp`).
9. **Loop back** to node 4 until batches exhausted (n8n's Split-in-Batches loop-back wiring).
10. **HTTP Request → `POST /session/{id}/end`** — get final summary metrics.
11. **Set + Postgres Insert** — write the run summary to a `sim_summaries` table (one row per
    episode: config used, final RMSE, coverage, drift-detection latency, total cost, LLM
    intervention count).
12. **Respond to Webhook / Slack node** — notify with a short summary and a link to the dashboard
    (Workflow 4).

**Key design point:** the LLM node is deliberately constrained to a fixed, small action vocabulary
rather than free-form text — this keeps it testable and keeps the "agentic" claim honest (it's
making discrete supervisory decisions over a well-defined tool set, not hallucinating control
signals into the physical loop).

---

## Workflow 2 — Tool / estimator integration test harness

**Purpose:** before trusting Workflow 1 for real experiments, verify every pluggable module
(estimator, kernel, policy) is correctly callable, returns well-formed data, and the LLM's tool
selection logic dispatches correctly to each. This is your "test the architecture" workflow.

### Nodes, in order

1. **Manual Trigger.**
2. **HTTP Request → `GET /estimators`, `GET /kernels`, `GET /policies`** — three parallel branches,
   discover everything registered in `simcore`.
3. **Merge node** — combine into one list of `{type: "estimator"|"kernel"|"policy", name}` items.
4. **Split in Batches** — iterate one module at a time.
5. **HTTP Request → `POST /session/start`** with a short fixed synthetic config (e.g., 20 steps,
   fixed seed, Gaussian noise) and this batch item's module selected.
6. **Split in Batches (inner loop, 20 items)** → **HTTP Request → `/step`** ×20, same pattern as
   Workflow 1 but no LLM node — this is a fast, deterministic smoke test.
7. **Function/Code node — Assertions:**
   - no NaN/Inf in `belief_mean`, `belief_cov`
   - `drift_flag` is boolean
   - response latency under a threshold (log it regardless)
   - if `type == "estimator"`: noise_summary vector has expected fixed length
   - if `type == "policy"`: action within declared action bounds
8. **IF node** — assertions pass?
   - **Fail branch:** **Set node** to build a failure record → **Postgres Insert** into
     `tool_test_failures` (module name, step, error detail) → **Slack/Email node** alert.
   - **Pass branch:** **Postgres Insert** into `tool_test_results` with `status = pass`, latency
     stats.
9. **HTTP Request → `POST /session/{id}/end`** — clean up session.
10. **Loop back** to node 4 until all modules tested.
11. **Aggregate node** — pull all rows from `tool_test_results` / `tool_test_failures` for this
    run, build a pass/fail matrix.
12. **Postgres/Sheets Insert** — write a summary "integration test report" row (timestamp, modules
    tested, pass count, fail count).
13. **Slack node** — post the pass/fail matrix so failures are visible before any real experiment
    runs on top of a broken module.

**Cross-tool integration check (important, add as steps 4b–4c inside the batch loop):** also test
*combinations*, not just individual modules in isolation — e.g., MMD estimator + time-varying
kernel + MPC policy together, since a module can pass alone but fail when composed (shape
mismatches in `noise_summary` feeding `build_belief`, etc.). Do this by making the batch list in
step 4 a **cartesian product** of a small representative subset (e.g., 2 estimators × 2 kernels ×
1 policy) rather than single modules, once single-module smoke tests are green.

---

## Workflow 3 — Ablation / experiment runner

**Purpose:** the actual research-grade batch runner — sweeps the full experiment grid described in
Phase 10 of the build plan, using Workflow 1's session API but without the interactive LLM step
(or with it, as a further ablation axis: `llm_supervisor: on/off`).

### Nodes, in order

1. **Manual Trigger**, or **Schedule Trigger** for periodic reruns as the codebase evolves.
2. **Code node** — build the parameter grid in JS/Python: cartesian product of
   `{noise_type} × {drift_schedule} × {estimator_method} × {expose_noise_info} × {llm_supervisor}`.
   Output as an array of config objects.
3. **Split in Batches** — one config per iteration.
4. **Execute Workflow node → Workflow 1** (called as a sub-workflow, passing the config in and
   waiting for its output) **or**, for speed, a slimmed inline version of Workflow 1's step loop
   without the Slack notification node.
5. **Function/Code node** — extract the final summary metrics from the sub-workflow's return.
6. **Postgres Insert** — append to `ablation_results` (all config fields + all metrics as columns,
   so it's directly queryable/pivotable later).
7. **Loop back** to node 3 until grid exhausted.
8. **Postgres query node** — pull the full `ablation_results` table for this batch run.
9. **Code node** — compute summary statistics needed for the report (mean/std RMSE per config
   group, best estimator per noise type, effect of `expose_noise_info`, etc.).
10. **Google Sheets node (or HTML/Markdown generation via Code node)** — write a results table,
    one sheet/section per comparison axis.
11. **Email/Slack node** — send the finished report link.

**Practical notes:**
- Grid size can get large fast (5 noise types × 3 drift schedules × 4 estimators × 2 × 2 = 240
  runs). Each run being ~20–200 steps keeps this tractable, but consider running Workflow 3 with
  n8n's queue mode / multiple workers if wall-clock time matters, since HTTP calls to `simcore`
  dominate cost, not n8n itself.
- Add a **rate/concurrency limit** (n8n's "Split in Batches" batch size, or a `Wait` node) if
  `simcore` isn't built to handle many concurrent sessions — check whether your FastAPI service is
  single-worker (likely, given in-memory JAX state) before firing 240 sessions in parallel.

---

## Workflow 4 — Monitoring, dashboard, and alerting

**Purpose:** lightweight observability so failures in Workflow 1/2/3 are caught quickly, and so
you (or your advisor) can check status without reading raw Postgres tables.

### Nodes, in order

1. **Schedule Trigger** — every N minutes, or **Webhook** triggered at the end of each workflow run.
2. **Postgres query node** — pull latest rows from `sim_summaries`, `tool_test_results`,
   `ablation_results`.
3. **Code node** — compute a small set of headline numbers (last run status, current best
   estimator by RMSE, any tool-test failures in the last 24h).
4. **IF node** — any failures or metric regressions vs. a stored baseline?
   - **True:** Slack/Email alert with details.
   - **False:** no-op.
5. **HTML/Markdown generation (Code node)** → write to a static file or a Notion/Confluence page
   via its connector, if available — a simple always-current status page.

---

## How the four workflows relate

```
Workflow 2 (tool/integration tests)
   └── must be green before →
Workflow 1 (single agentic run)  ←── called as sub-workflow by ──┐
   └── individual runs feed →                                    │
Workflow 3 (ablation/experiment grid) ───────────────────────────┘
   └── results feed →
Workflow 4 (monitoring/dashboard)
```

Run order in practice:
1. First time, and after any change to `simcore`'s API surface: run **Workflow 2**. Don't proceed
   if anything fails.
2. Sanity-check with a single **Workflow 1** run (small `n_steps`, look at the log table by eye).
3. Once both are clean, launch the full **Workflow 3** grid.
4. Leave **Workflow 4** on a schedule throughout.

---

## Suggested n8n setup checklist

- [ ] `simcore` FastAPI service running and reachable from n8n (same Docker network or public URL)
- [ ] Postgres (or equivalent) instance with tables: `sim_runs`, `sim_summaries`,
      `tool_test_results`, `tool_test_failures`, `ablation_results`
- [ ] Credentials configured in n8n for: Postgres, Slack/Email, LLM provider (Anthropic/OpenAI)
- [ ] Workflow 2 built and passing on all currently implemented modules
- [ ] Workflow 1 built, tested on one config end-to-end, LLM action vocabulary fixed and validated
- [ ] Workflow 3 built, grid size sanity-checked against `simcore`'s concurrency limits
- [ ] Workflow 4 scheduled and alert channel confirmed working (send a test alert)
