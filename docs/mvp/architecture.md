# Architecture

## Control Plane

- `FastAPI` service owns sessions, turns, runs, approvals, and metrics.
- `ServiceStore` persists state in SQLite (`.anton/service/service.db`) and writes immutable audit lines to `.anton/service/audit.log.jsonl`.
- `RuntimeManager` controls in-memory session runtimes and enforces per-turn locks.

## Data Plane

- `ChatSession` remains the orchestration engine for LLM planning and tool execution.
- Scratchpad execution is still per-session and persistent across turns.
- Connector access is exposed as a first-class tool through `ConnectorHub`.

## Connector Layer

- `HTTPConnectorClient`: bridges to an existing connector/auth service (`ANTON_CONNECTOR_API_BASE_URL`).
- `LocalSQLiteConnectorClient`: local development/testing backend.

## Governance Layer

- `PolicyEngine` enforces tool-level constraints.
- `RunGovernance` creates approval records for destructive actions.
- `BudgetTracker` enforces token and tool-call budgets.

## Reproducibility

- Per-run traces include prompt/response, audit events, token usage, approvals, and generated artifacts.
- Run traces are retrievable via `/runs/{run_id}/trace`.
