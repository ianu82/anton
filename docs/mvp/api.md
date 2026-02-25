# API Contract

## Session Lifecycle

### `POST /sessions`

Create a session runtime.

Request:

```json
{
  "session_id": "optional-id",
  "workspace_path": "/abs/path/to/workspace",
  "metadata": {
    "team": "analytics",
    "auth_context": {
      "user_id": "u-123",
      "org_id": "org-9",
      "roles": ["analyst"],
      "attributes": {"region": "us"}
    }
  }
}
```

Response:

```json
{
  "session_id": "abc123",
  "workspace_path": "/abs/path/to/workspace",
  "metadata": {"team": "analytics"}
}
```

### `POST /sessions/{session_id}/turn`

Execute one conversational turn.

Request:

```json
{
  "message": "show churn by cohort",
  "idempotency_key": "optional-client-key",
  "wait_for_completion": true,
  "wait_timeout_seconds": 300
}
```

Response fields:

- `run_id`
- `status` (`queued` | `running` | `completed` | `approval_required` | `cancelled` | `failed`)
- `reply`
- `error`
- `pending_approval_ids`

If `idempotency_key` is repeated for the same session, the existing run is returned instead of creating a duplicate.

### `GET /sessions/{session_id}/runs`

List recent runs for a session.
- `total_tokens`
- `tool_calls`

### `GET /sessions/{session_id}/events`

Return session events, optionally incremental by `since_id`.

Query params:

- `since_id` (default `0`)
- `limit` (default `500`)

Notable event types:

- `memory_retrieval`: emitted at run start with payload:
  - `retrieved_count`
  - `provenance` (`session_summary`/`learning` identifiers)

## Run and Artifacts

### `GET /runs/{run_id}`

Returns run summary and final status.

### `POST /runs/{run_id}/cancel`

Requests cancellation for an in-flight run (best effort).

### `GET /runs/{run_id}/artifacts`

Returns run artifacts discovered under `.anton/output`.

### `GET /runs/{run_id}/trace`

Returns complete reproducibility bundle:

- run summary
- run events
- artifacts

## Approvals

### `GET /approvals`

List approvals by `status` (default `pending`) and optional `session_id` filter.

### `POST /approvals/{approval_id}/decision`

Request body:

```json
{
  "approved": true,
  "note": "approved by analyst"
}
```

## Skills

### `POST /skills`

Create a reusable, parameterized prompt skill.

Request:

```json
{
  "name": "cohort_churn",
  "description": "Analyze churn by cohort",
  "prompt_template": "Show churn by {dimension} for {period}.",
  "metadata": {"team": "analytics"}
}
```

### `GET /skills`

List skills with latest version metadata.

### `GET /skills/{skill_id}`

Fetch one skill and its latest template/version details.

### `POST /skills/{skill_id}/versions`

Create a new version for an existing skill.

### `POST /skills/{skill_id}/run`

Render a skill template with params and execute it as a normal session turn.

Request:

```json
{
  "session_id": "abc123",
  "version": 2,
  "params": {
    "dimension": "cohort",
    "period": "Q1"
  }
}
```

## Scheduled Runs

### `POST /scheduled-runs`

Create a saved schedule definition backed by a skill + params.

```json
{
  "name": "daily-kpi",
  "session_id": "abc123",
  "skill_id": "skill-1",
  "skill_version": 2,
  "params": {"dimension": "cohort", "period": "Q1"},
  "interval_seconds": 3600,
  "start_in_seconds": 0,
  "active": true
}
```

### `GET /scheduled-runs`

List schedules (optional `status=active|paused` filter).

### `GET /scheduled-runs/{schedule_id}`

Fetch one schedule.

### `POST /scheduled-runs/{schedule_id}/trigger`

Trigger a schedule immediately and execute its configured skill run.

### `POST /scheduled-runs/{schedule_id}/pause`

Pause a schedule.

### `POST /scheduled-runs/{schedule_id}/resume`

Resume a paused schedule.

When `ANTON_SERVICE_SCHEDULER_ENABLED=true`, active schedules whose `next_run_at` is due are automatically triggered by the service worker.
Automatic and manual triggers emit `schedule_triggered` events (`payload.mode` is `automatic` or `manual`).

## Metrics

### `GET /metrics`

Returns run counts, success rates, and latency percentiles.
