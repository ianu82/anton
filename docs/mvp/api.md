# API Contract

## Session Lifecycle

### `POST /sessions`

Create a session runtime.

Request:

```json
{
  "session_id": "optional-id",
  "workspace_path": "/abs/path/to/workspace",
  "metadata": {"team": "analytics"}
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
  "message": "show churn by cohort"
}
```

Response fields:

- `run_id`
- `status` (`completed` | `approval_required` | `failed`)
- `reply`
- `error`
- `pending_approval_ids`
- `total_tokens`
- `tool_calls`

### `GET /sessions/{session_id}/events`

Return session events, optionally incremental by `since_id`.

Query params:

- `since_id` (default `0`)
- `limit` (default `500`)

## Run and Artifacts

### `GET /runs/{run_id}`

Returns run summary and final status.

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

## Metrics

### `GET /metrics`

Returns run counts, success rates, and latency percentiles.
