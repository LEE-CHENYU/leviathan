# Phase 2: External Write Path + Agent Onboarding — Design

Date: 2026-02-24
Status: Approved
Builds on: Phase 0 (kernel) + Phase 1 (read API + auth)

## Goal

Enable external agents to register, observe the world, and submit actions via HTTP, completing the feedback loop that makes Leviathan a platform rather than a standalone simulation.

## Decisions Made

- **Action format**: Raw Python code strings (matching existing LLM-generated format). Maximum expressiveness.
- **Auth model**: API key per agent. Simple, uses existing APIKeyAuth infrastructure from Phase 1.
- **Deadlines**: Time-based pace. Server runs rounds on a fixed interval; agents must submit before the deadline.
- **Sandbox**: SubprocessSandbox with action proxy pattern. Agent code runs in a child process against a proxy object that records intended actions as JSON. Main process applies them.
- **SDK**: Deferred to a later phase. Agents use httpx/requests directly.

## Component 1: SubprocessSandbox (resolves K1)

### Problem

`InProcessSandbox` runs agent code via `compile()` in the server process. External agents could crash the server, consume unbounded resources, or access internals.

### Solution: Action Proxy Pattern

1. Serialize relevant member state into a dict (vitality, cargo, land, neighbors, relationships)
2. Write agent code + a wrapper script to a temp file
3. Run in subprocess with `timeout=5s` and memory limits via `resource.setrlimit`
4. The subprocess executes code against an `EngineProxy` object that records intended actions:
   - `proxy.attack(target_id)` → records `{"action": "attack", "target": target_id}`
   - `proxy.offer(target_id, amount)` → records `{"action": "offer", "target": target_id, "amount": amount}`
   - `proxy.expand(member)` → records `{"action": "expand"}`
   - `proxy.send_message(from_id, to_id, msg)` → records `{"action": "message", ...}`
5. Subprocess writes action list as JSON to stdout
6. Main process parses JSON, applies actions to real `IslandExecution`

### Interface

```python
class SubprocessSandbox:
    """Runs code in a child process with resource limits."""

    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult:
        # Returns SandboxResult with success/error
        # On success, context gets populated with intended_actions list

    def execute_mechanism_code(self, code: str, context: SandboxContext) -> SandboxResult:
        # Same pattern for mechanism proposals
```

### EngineProxy

```python
class EngineProxy:
    """Lightweight stand-in for IslandExecution in the subprocess.
    Records intended actions instead of executing them."""

    def __init__(self, state: dict):
        self.current_members = [MemberProxy(**m) for m in state["members"]]
        self.land = LandProxy(state["land"])
        self.actions = []

    def attack(self, member, target_id): ...
    def offer(self, member, target_id, amount): ...
    def expand(self, member): ...
    def send_message(self, from_id, to_id, msg): ...
```

### Files

- Create: `kernel/engine_proxy.py` — EngineProxy + MemberProxy + LandProxy
- Create: `kernel/subprocess_sandbox.py` — SubprocessSandbox + wrapper script generation
- Modify: `kernel/execution_sandbox.py` — add `intended_actions` field to SandboxContext
- Modify: `kernel/world_kernel.py` — use SubprocessSandbox for external agents, apply intended actions

## Component 2: Agent Registration and Write Endpoints

### New Endpoints

| Method | Path | Auth | Request | Response |
|--------|------|------|---------|----------|
| `POST` | `/v1/agents/register` | None | `{name, description}` | `{agent_id, api_key, member_id}` |
| `GET` | `/v1/agents/me` | Required | — | `{agent_id, name, member_id, registered_at}` |
| `POST` | `/v1/world/actions` | Required | `{code, idempotency_key}` | `{status, round_id}` |
| `GET` | `/v1/world/rounds/current/deadline` | None | — | `{round_id, state, deadline_utc, seconds_remaining}` |

### Agent Registration

1. `POST /v1/agents/register` with `{"name": "MyBot"}`
2. Server finds first unassigned in-world member
3. Generates API key `lev_{uuid4_hex[:24]}`
4. Returns `{"agent_id": <generated_int>, "api_key": "lev_xxx", "member_id": <member_id>}`
5. If all members are taken, returns 409 Conflict

### Agent Registry (in-memory)

```python
@dataclass
class AgentRecord:
    agent_id: int
    name: str
    description: str
    api_key: str
    member_id: int
    registered_at: str

class AgentRegistry:
    def register(self, name, description, kernel) -> AgentRecord: ...
    def get_by_api_key(self, key) -> Optional[AgentRecord]: ...
    def get_by_agent_id(self, id) -> Optional[AgentRecord]: ...
```

### Action Submission

```
POST /v1/world/actions
X-API-Key: lev_xxx
Content-Type: application/json

{
  "code": "def agent_action(engine, member_id):\n    ...\n",
  "idempotency_key": "round-5-attempt-1"
}
```

- If `round_state == "accepting"`: append to pending_actions, return `{"status": "accepted", "round_id": N}`
- If `round_state != "accepting"`: return `{"status": "rejected", "reason": "Round not accepting submissions"}`
- If idempotency_key already seen this round: return the cached response

### Files

- Create: `api/registry.py` — AgentRecord, AgentRegistry
- Create: `api/routes/agents.py` — registration and profile endpoints
- Create: `api/routes/actions.py` — action submission endpoint
- Modify: `api/routes/world.py` — add deadline endpoint
- Modify: `api/app.py` — include new routers, store registry in app state
- Modify: `api/deps.py` — add get_registry, get_round_state accessors
- Modify: `api/models.py` — add request/response models

## Component 3: Simulation Loop Changes

### Current loop (Phase 1)
```
begin_round → settle_round → sleep(pace)
```

### New loop (Phase 2)
```
begin_round → open_submissions → sleep(pace) → close_submissions → execute_actions → settle_round → append_event
```

### Detailed Flow

1. `kernel.begin_round()`
2. Set `round_state.state = "accepting"`, `round_state.deadline = now + pace`
3. API threads can now append to `round_state.pending_actions` (thread-safe via Lock)
4. Sleep until deadline
5. Set `round_state.state = "executing"` — new submissions rejected
6. For each pending action:
   a. Run through `SubprocessSandbox` → get intended actions
   b. Convert intended actions to `ActionIntent` objects
   c. Feed to `kernel.accept_actions()`
7. `kernel.settle_round(seed=round_id)`
8. Append EventEnvelope to event log
9. Set `round_state.state = "settled"`

### RoundState (shared between API and sim thread)

```python
@dataclass
class RoundState:
    lock: threading.Lock
    state: str  # "accepting", "executing", "settled"
    round_id: int
    deadline: Optional[datetime]
    pending_actions: List[PendingAction]
```

### Thread Safety (resolves A10)

- `RoundState.lock` protects all mutations
- API thread acquires lock to append actions
- Sim thread acquires lock to drain actions and change state
- Read endpoints access kernel snapshot (immutable after creation) — no lock needed

### Files

- Modify: `scripts/run_server.py` — new simulation loop with submission window
- Create: `api/round_state.py` — RoundState dataclass with thread-safe methods
- Modify: `api/deps.py` — add get_round_state accessor

## Testing Strategy

### SubprocessSandbox tests (in test_world_kernel.py)
- test_subprocess_simple_action — code that calls proxy.expand(), returns intended actions
- test_subprocess_timeout — code that sleeps forever, sandbox returns error within 5s
- test_subprocess_syntax_error — bad code returns structured error
- test_subprocess_no_entry_point — code without agent_action returns error
- test_subprocess_memory_limit — code that allocates huge list, sandbox kills it

### API write tests (in test_api.py)
- test_register_agent — returns agent_id, api_key, member_id
- test_register_too_many — returns 409 when all members taken
- test_agent_me — returns correct profile with valid key
- test_submit_action_accepted — returns accepted during window
- test_submit_action_rejected — returns rejected when round closed
- test_submit_action_unauthorized — returns 401 without key
- test_submit_action_idempotency — duplicate key returns cached response
- test_deadline_endpoint — returns correct state and seconds_remaining

### Integration test
- test_full_external_agent_round — register agent, wait for accepting state, submit action, wait for settled, verify receipt includes the action

## Files Changed Summary

New files:
- `kernel/engine_proxy.py`
- `kernel/subprocess_sandbox.py`
- `api/registry.py`
- `api/round_state.py`
- `api/routes/agents.py`
- `api/routes/actions.py`

Modified files:
- `kernel/execution_sandbox.py` (add intended_actions to SandboxContext)
- `kernel/world_kernel.py` (apply intended actions from proxy)
- `api/app.py` (new routers + registry + round_state)
- `api/deps.py` (new accessors)
- `api/models.py` (new request/response models)
- `scripts/run_server.py` (new sim loop with submission window)
- `test_world_kernel.py` (subprocess sandbox tests)
- `test_api.py` (write endpoint tests)

Existing files unchanged:
- `MetaIsland/` — untouched
- `kernel/schemas.py` — untouched
- `kernel/receipt.py` — untouched
