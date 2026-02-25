# Phase 1: Public Read API — Design

Date: 2026-02-24
Status: Approved
Builds on: Phase 0 WorldKernel extraction

## Goal

Expose the WorldKernel's state via a read-only HTTP API so external tools, dashboards, and future agent SDKs can observe the simulation without coupling to Python internals.

## Decisions Made

- **World management**: Single world per server instance. Multi-world comes later.
- **Storage**: In-memory only. No SQLite in Phase 1; receipts and events live in Python lists.
- **Event feed**: Polling via `GET /events?since_round=N`. No SSE streaming in Phase 1.
- **Architecture**: Thin API layer (Approach A). `api/` only imports from `kernel/`, never from `MetaIsland/` directly.
- **Upgrade path**: URL uses `/v1/world` (singular). Phase 2 adds `/v1/worlds/{id}/...` alongside — no breaking change.
- **Framework**: FastAPI (already installed). Pydantic response models. Auto-generated OpenAPI docs.

## Directory Structure

```
api/
  __init__.py
  app.py            # FastAPI application factory
  deps.py           # Dependency injection (get_kernel, get_event_log)
  models.py         # Pydantic response models + EventEnvelope
  routes/
    __init__.py
    world.py        # /v1/world/* endpoints
    discovery.py    # /.well-known/leviathan-agent.json
scripts/
  run_server.py     # Creates WorldKernel + starts uvicorn + runs sim loop
test_api.py         # At repo root (matching existing test layout)
```

## API Endpoints

| Method | Path | Response Model | Purpose |
|--------|------|----------------|---------|
| `GET` | `/v1/world` | `WorldInfo` | World metadata (id, round_id, member_count, state_hash) |
| `GET` | `/v1/world/snapshot` | `WorldSnapshotResponse` | Full current state |
| `GET` | `/v1/world/rounds/current` | `RoundInfo` | Current round_id + last receipt |
| `GET` | `/v1/world/rounds/{round_id}` | `RoundReceiptResponse` | Historical receipt by round_id |
| `GET` | `/v1/world/events?since_round=N` | `List[EventEnvelope]` | Polling event feed |
| `GET` | `/.well-known/leviathan-agent.json` | `AgentDiscovery` | Agent discovery manifest |
| `GET` | `/health` | `{"status": "ok"}` | Health check |

## Response Models (Pydantic)

### WorldInfo
```python
class WorldInfo(BaseModel):
    world_id: str
    round_id: int
    member_count: int
    state_hash: str
```

### RoundInfo
```python
class RoundInfo(BaseModel):
    round_id: int
    last_receipt: Optional[RoundReceiptResponse]
```

### EventEnvelope
```python
class EventEnvelope(BaseModel):
    event_id: int           # monotonic sequence number
    event_type: str         # "round_settled", etc.
    round_id: int
    timestamp: str
    payload: Dict[str, Any] # the receipt dict
```

### AgentDiscovery
```json
{
  "name": "Leviathan",
  "version": "0.1.0",
  "api_version": "v1",
  "capabilities": ["read_snapshot", "read_events", "read_receipts"],
  "endpoints": { "base": "/v1/world" }
}
```

## Dependency Injection (deps.py)

The app factory accepts a `WorldKernel` instance and an event log (list). These are injected via FastAPI's `Depends()` mechanism. This makes testing trivial — `TestClient` overrides the dependency to inject a test kernel.

```python
# Singleton holders, set by create_app()
_kernel: Optional[WorldKernel] = None
_event_log: Optional[List[EventEnvelope]] = None

def get_kernel() -> WorldKernel: ...
def get_event_log() -> List[EventEnvelope]: ...
```

## Server Runner (scripts/run_server.py)

CLI arguments:
- `--members N` (default: 10)
- `--land WxH` (default: 20x20)
- `--seed S` (default: 42)
- `--port P` (default: 8000)
- `--rounds N` (default: 0 = infinite)
- `--pace SECONDS` (default: 2.0, time between rounds)

Behavior:
1. Creates `WorldKernel(WorldConfig(...), save_path=tmpdir)`
2. Creates FastAPI app via `create_app(kernel, event_log)`
3. Starts simulation loop in a daemon background thread
4. Runs uvicorn on main thread
5. Graceful shutdown on SIGINT (stops sim loop, then uvicorn)

The simulation loop:
```
while running and (max_rounds == 0 or round < max_rounds):
    kernel.begin_round()
    kernel.settle_round(seed=round)
    append EventEnvelope to event_log
    sleep(pace)
```

## Testing Strategy

All tests in `test_api.py` at repo root. Uses FastAPI TestClient (httpx-based, no real server).

Tests:
1. `test_health` — 200, `{"status": "ok"}`
2. `test_world_info` — correct world_id, round_id, member_count
3. `test_snapshot` — returns valid snapshot with members list
4. `test_rounds_current_before_any_round` — round_id=0, no receipt
5. `test_rounds_current_after_settle` — receipt appears after round
6. `test_events_polling` — settle 3 rounds, `?since_round=1` returns 2 events
7. `test_events_empty` — `?since_round=999` returns empty list
8. `test_round_by_id` — GET /v1/world/rounds/1 returns correct receipt
9. `test_round_not_found` — GET /v1/world/rounds/999 returns 404
10. `test_discovery` — valid manifest with required fields

Fixture: WorldKernel with seed=42, 5 members, 10x10 land. Helper function to settle N rounds and populate event log.

## Files Changed

New files:
- `api/__init__.py`
- `api/app.py`
- `api/deps.py`
- `api/models.py`
- `api/routes/__init__.py`
- `api/routes/world.py`
- `api/routes/discovery.py`
- `scripts/run_server.py`
- `test_api.py`

Modified files:
- `requirements.txt` (add fastapi, uvicorn, httpx)

Existing files unchanged:
- `kernel/` — untouched
- `MetaIsland/` — untouched
- All existing tests — must keep passing
