# Anton MVP

This folder documents the MVP service build that turns Anton into a connector-native conversational analytics backend.

## Delivered Scope

1. Sessioned API (`/sessions`, `/turn`, `/events`, `/runs`, `/approvals`, `/metrics`)
2. Connector tool bridge (`list`, `schema`, `query`, `sample`, `write`)
3. Governance baseline (policy gates, approval workflow, budgets, audit events)
4. Artifact and run trace endpoints for reproducibility
5. Benchmark harness (`anton eval-mvp`) with p50/p95 and pass-rate reporting
6. Versioned skill registry (`/skills`) with parameterized run execution
7. Scheduled run definitions (`/scheduled-runs`) with trigger/pause/resume controls
8. Optional scheduler worker for automatic due-run triggering

## Key Paths

- Service API: `anton/service/app.py`
- Runtime manager: `anton/service/runtime.py`
- Store/audit layer: `anton/service/store.py`
- Connector adapters: `anton/connectors/`
- Governance modules: `anton/governance/`
- Eval harness: `anton/service/eval.py`
