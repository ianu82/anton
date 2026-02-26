# Runbook

## Local Startup

1. Configure provider credentials in `.anton/.env`.
2. (Optional) Configure connector bridge via env vars:
   - `ANTON_CONNECTOR_MODE=http`
   - `ANTON_CONNECTOR_API_BASE_URL=https://connector.internal`
   - `ANTON_CONNECTOR_API_TOKEN=...`
3. Start service:

```bash
anton serve --host 127.0.0.1 --port 8000
```

## Health Checks

- `GET /health` must return `{ "status": "ok" }`.
- `GET /metrics` should show non-zero runs after smoke tests.

## Smoke Test

1. `POST /sessions`
2. `POST /sessions/{id}/turn`
3. `GET /sessions/{id}/events`
4. `GET /runs/{run_id}/trace`
5. `POST /runs/{run_id}/cancel` (optional cancellation check)
6. `POST /skills` + `POST /skills/{skill_id}/run` (optional reusable workflow check)
7. `POST /scheduled-runs` + `POST /scheduled-runs/{id}/trigger` (optional scheduling check)

## CLI Equivalents

With `anton serve` running, you can use:

- `anton skills list`
- `anton skills create <name> "<prompt_template>"`
- `anton skills run <skill_id> --session-id <session_id> --params '{"metric":"churn"}'`
- `anton schedules list`
- `anton schedules create --session-id <session_id> --skill-id <skill_id> --params '{"metric":"churn"}' --interval-seconds 3600`
- `anton schedules trigger <schedule_id>`

In interactive chat (`anton`), slash commands are also available:

- `/skills list|show|create|version|run`
- `/schedules list|show|create|trigger|pause|resume`

## Approval Handling

1. Query `GET /approvals` for `pending` items.
2. Approve/reject via `POST /approvals/{id}/decision`.
3. Re-run turn after approval if the run ended with `approval_required`.

## Budget and Policy Tuning

Tune via env/settings:

- `ANTON_MAX_TOKENS_PER_RUN`
- `ANTON_MAX_TOOL_CALLS_PER_RUN`
- `ANTON_MAX_ESTIMATED_SECONDS_WITHOUT_APPROVAL`
- `ANTON_CONNECTOR_MAX_QUERY_LIMIT`
- `ANTON_MEMORY_ENABLED`
- `ANTON_MEMORY_RECALL_MAX_ITEMS`
- `ANTON_MEMORY_RECALL_MAX_CHARS`
- `ANTON_SERVICE_WORKER_MODE` (`local` or `queue`)
- `ANTON_SERVICE_QUEUE_WORKER_COUNT`
- `ANTON_SERVICE_SCHEDULER_ENABLED`
- `ANTON_SERVICE_SCHEDULER_POLL_SECONDS`
- `ANTON_SERVICE_SCHEDULER_BATCH_SIZE`
