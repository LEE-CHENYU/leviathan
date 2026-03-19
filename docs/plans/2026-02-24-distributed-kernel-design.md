# Distributed Kernel Design

Date: 2026-02-24
Status: Approved
Branch: `distributed-kernel`

## Goal

Extract a pure `WorldKernel` facade from the existing `IslandExecution` runtime to establish typed, serializable interfaces for deterministic state transitions. This is Phase 0 of the distributed architecture migration — the keystone that all later phases build on.

## Decisions Made

- **Scope**: Full blueprint documented, Phase 0 is the concrete implementation target, Phases 1-4 subject to change.
- **Code execution**: Keep current in-process execution for now, wrap behind swappable `ExecutionSandbox` interface.
- **Refactor style**: Extract pure kernel alongside existing code (facade pattern). `IslandExecution` keeps working; `WorldKernel` delegates to it internally.
- **Testing**: Determinism golden tests (same seed + same inputs = same state hash).

## Phase 0: WorldKernel Extraction

### New Directory Structure

```
kernel/
  __init__.py
  world_kernel.py       # WorldKernel facade class
  schemas.py            # Dataclasses for all typed interfaces
  execution_sandbox.py  # Swappable code execution interface
  receipt.py            # Deterministic hashing and receipt generation
```

### WorldKernel Interface

```python
class WorldKernel:
    """Pure facade over IslandExecution.
    No LLM calls, no network, no filesystem side effects."""

    def __init__(self, config: WorldConfig): ...
    def get_snapshot(self) -> WorldSnapshot: ...
    def accept_actions(self, actions: List[ActionIntent]) -> List[ActionResult]: ...
    def accept_mechanisms(self, mechanisms: List[MechanismProposal]) -> List[MechanismResult]: ...
    def settle_round(self, seed: int) -> RoundReceipt: ...
    def get_round_receipt(self) -> RoundReceipt: ...
```

**Design principle**: The kernel has no LLM calls. Agent decision generation (analyze, propose, decide) stays outside the kernel. The kernel handles: validate inputs, run deterministic state transitions, emit receipts.

### Schemas

```python
@dataclass
class WorldConfig:
    init_member_number: int
    land_shape: Tuple[int, int]
    random_seed: Optional[int] = None

@dataclass
class ActionIntent:
    agent_id: int
    round_id: int
    code: str
    idempotency_key: str

@dataclass
class ActionResult:
    agent_id: int
    success: bool
    old_stats: Dict[str, float]
    new_stats: Dict[str, float]
    performance_change: float
    messages_sent: List[Tuple[int, str]]
    error: Optional[str] = None
    signature: Optional[Dict] = None

@dataclass
class MechanismProposal:
    proposal_id: str
    agent_id: int
    code: str
    round_id: int

@dataclass
class MechanismResult:
    proposal_id: str
    executed: bool
    error: Optional[str] = None

@dataclass
class WorldSnapshot:
    world_id: str
    round_id: int
    members: List[Dict]
    land: Dict
    active_mechanisms: List[Dict]
    active_contracts: List[Dict]
    physics_constraints: List[Dict]
    state_hash: str

@dataclass
class RoundReceipt:
    round_id: int
    seed: int
    snapshot_hash_before: str
    snapshot_hash_after: str
    accepted_action_ids: List[str]
    rejected_action_ids: List[str]
    activated_mechanism_ids: List[str]
    judge_results: List[Dict]
    round_metrics: Dict[str, float]
    timestamp: str
```

### Execution Sandbox Interface

```python
class ExecutionSandbox(Protocol):
    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult: ...
    def execute_mechanism_code(self, code: str, context: SandboxContext) -> SandboxResult: ...

class InProcessSandbox(ExecutionSandbox):
    """Phase 0: wraps existing in-process execution with restricted namespace."""
```

### Receipt Hashing

- Canonical JSON: sorted keys, no whitespace, UTF-8 encoding.
- SHA-256 for all hashes.
- Kernel must not use nondeterministic sources during settlement (no wall clock, no unseeded random, no network).

```python
def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str).encode('utf-8')

def state_hash(snapshot: WorldSnapshot) -> str:
    return hashlib.sha256(canonical_json(snapshot_to_dict(snapshot))).hexdigest()
```

### Data Flow Through a Round

```
Agent LLM (outside kernel)
    | produces ActionIntent[]
WorldKernel.accept_actions(actions)
    | returns ActionResult[]
Agent LLM (outside kernel)
    | produces MechanismProposal[]
Judge (outside kernel)
    | produces approved/rejected list
WorldKernel.accept_mechanisms(approved)
    | returns MechanismResult[]
WorldKernel.settle_round(seed)
    | runs produce/consume/environment
    | returns RoundReceipt
```

### Testing Strategy

1. **Determinism golden tests**: Create WorldKernel with fixed seed, feed identical inputs, assert identical `snapshot_hash_after` across runs.
2. **Schema unit tests**: Serialization round-trip (dataclass -> dict -> JSON -> dict -> dataclass). Canonical JSON produces identical bytes.
3. **Integration test**: Run full round via WorldKernel facade, run same round via direct IslandExecution, assert member states match.
4. **Existing tests**: `test_graph_system.py` and `test_eval_metrics.py` must keep passing.

## Phase 1: Public Read API + Event Feed (subject to change)

- FastAPI app in new `api/` directory.
- Read-only endpoints: `/v1/worlds`, `/v1/worlds/{id}/rounds/current`, `/v1/worlds/{id}/events`, `/v1/worlds/{id}/snapshot`.
- `/.well-known/leviathan-agent.json` discovery endpoint.
- Event envelope schema wrapping kernel's `RoundReceipt`.
- Storage: SQLite initially (upgrade to Postgres when needed).

## Phase 2: External Write Path + Agent Onboarding (subject to change)

- Auth: ephemeral keypair registration (`POST /v1/agents/register`).
- Action submission: `POST /v1/worlds/{id}/actions` with deadline enforcement.
- Idempotency keys, rate limits.
- Minimal Python SDK in `sdk/python/`.

## Phase 3: Mechanism Governance Service (subject to change)

- Separate judge worker (process boundary).
- Mechanism proposal API with lifecycle states: `submitted -> lint_failed | judge_failed | canary_active -> active -> rolled_back`.
- Judge receipts stored for audit.
- Constrained mechanism DSL as primary format.

## Phase 4: Moderator + Multi-Tenant (subject to change)

- Moderator APIs: pause, rollback, ban, quota adjustment.
- Per-world and per-agent isolation.
- Constitution endpoint: `GET /v1/worlds/{id}/constitution`.

## Cross-World Federation (deferred, post-Phase 4)

- No-regret prep in Phase 0: `world_id` in all schemas, origin fields reserved in receipts.
- Three aggregation models documented in `docs/system-design/06-united-leviathan.md`.
- Federation model (sovereign worlds + bridge events) recommended over single-world sharding.

## Files Changed by Phase 0

New files:
- `kernel/__init__.py`
- `kernel/world_kernel.py`
- `kernel/schemas.py`
- `kernel/execution_sandbox.py`
- `kernel/receipt.py`
- `test_world_kernel.py` (golden + unit + integration tests)

Existing files unchanged:
- `MetaIsland/metaIsland.py` — untouched
- `MetaIsland/graph_engine.py` — untouched
- `MetaIsland/nodes/` — untouched
- All existing tests — must keep passing
