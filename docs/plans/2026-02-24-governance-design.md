# Phase 3: Governance — DAG Engine, Mechanisms, Judge & Metrics — Design

Date: 2026-02-24
Status: Approved
Builds on: Phase 0 (kernel) + Phase 1 (read API) + Phase 2 (write API + agent onboarding)

## Goal

Enable agents to propose and vote on world rules (mechanisms), validated by an LLM judge, executed through the existing DAG graph engine, with round-level observability metrics — completing the governance feedback loop.

## Decisions Made

- **Settlement**: Integrate the existing DAG graph engine (`ExecutionGraph`) rather than rebuilding phases manually (resolves K4)
- **Judge execution**: Subprocess worker with timeout (reuses existing `Judge` class via adapter)
- **Mechanism lifecycle**: Simple — submitted → approved/rejected → active (no canary mode)
- **Scope**: Full governance — DAG integration + mechanism proposals + judge + metrics

## Component 1: DAG Engine Integration (resolves K4)

### Problem

`WorldKernel.settle_round()` only calls `produce()` and `consume()`, missing fight, trade, reproduce, environment, contracts, and mechanism execution. The full simulation loop lives in `ExecutionGraph` (12 DAG nodes in topological layers).

### Solution

Wire `WorldKernel.settle_round()` to call the existing `ExecutionGraph` infrastructure nodes instead of manually calling individual phases.

### Which Nodes to Run

The 12 DAG nodes in `ExecutionGraph`:

| Node | Type | Phase 3 Action |
|------|------|----------------|
| `new_round` | infrastructure | **Run** — round bookkeeping |
| `analyze` | LLM-driven | **Skip** — requires live LLM, not kernel concern |
| `propose_mechanisms` | LLM-driven | **Replace** — use new MechanismRegistry (proposals come via API) |
| `judge` | LLM-driven | **Replace** — use JudgeAdapter (subprocess worker) |
| `execute_mechanisms` | infrastructure | **Run** — apply approved mechanisms |
| `agent_decisions` | LLM-driven | **Skip** — agents submit via Phase 2 write path |
| `execute_actions` | infrastructure | **Run** — apply agent actions |
| `contracts` | infrastructure | **Run** — settle contracts |
| `produce` | infrastructure | **Run** — resource production |
| `consume` | infrastructure | **Run** — resource consumption |
| `environment` | infrastructure | **Run** — environmental effects |
| `log_status` | infrastructure | **Run** — round logging |

### Implementation Approach

Create a `KernelDAGRunner` that:
1. Takes the existing `IslandExecution` instance from `WorldKernel`
2. Calls infrastructure phase methods in topological order
3. Skips LLM-driven nodes (analyze, agent_decisions)
4. Replaces propose_mechanisms and judge with kernel-managed equivalents
5. Passes `MechanismRegistry` and `JudgeAdapter` as dependencies

The runner does NOT use `ExecutionGraph` directly (that class is tightly coupled to MetaIsland's LLM pipeline). Instead, it calls the same underlying `IslandExecution` methods in the same order.

### Phase Order (matching DAG topology)

```
new_round → [propose via API] → judge → execute_mechanisms → [actions via API] → execute_actions → contracts → produce → consume → environment → log_status
```

### Files

- Create: `kernel/dag_runner.py` — KernelDAGRunner
- Modify: `kernel/world_kernel.py` — `settle_round()` delegates to KernelDAGRunner

## Component 2: Mechanism Proposals API + Registry (resolves K3)

### MechanismRegistry

Tracks all mechanism proposals and their lifecycle.

```python
@dataclass
class MechanismRecord:
    mechanism_id: str          # uuid4
    proposer_id: int           # agent member_id
    code: str                  # Python code string
    description: str           # human-readable summary
    status: str                # "submitted" | "approved" | "rejected" | "active"
    submitted_round: int
    judged_round: Optional[int]
    judge_reason: Optional[str]
    activated_round: Optional[int]

class MechanismRegistry:
    def submit(self, proposer_id, code, description) -> MechanismRecord
    def get_pending(self) -> List[MechanismRecord]
    def mark_approved(self, mechanism_id, round_id, reason) -> None
    def mark_rejected(self, mechanism_id, round_id, reason) -> None
    def activate(self, mechanism_id, round_id) -> None
    def get_active(self) -> List[MechanismRecord]
    def get_all(self) -> List[MechanismRecord]
    def get_by_id(self, mechanism_id) -> Optional[MechanismRecord]
```

### Lifecycle

```
submitted ──judge approves──→ approved ──execute_mechanisms──→ active
    │
    └──judge rejects──→ rejected
```

- **submitted**: Agent proposed via API, awaiting judge review
- **approved**: Judge approved, will be activated in execute_mechanisms phase
- **rejected**: Judge rejected with reason
- **active**: Code has been applied to the world

### New Endpoints

| Method | Path | Auth | Request | Response |
|--------|------|------|---------|----------|
| `POST` | `/v1/world/mechanisms/propose` | Required | `{code, description, idempotency_key}` | `{mechanism_id, status}` |
| `GET` | `/v1/world/mechanisms` | None | `?status=active` | `[{mechanism_id, proposer_id, description, status, ...}]` |
| `GET` | `/v1/world/mechanisms/{id}` | None | — | `{mechanism_id, code, status, judge_reason, ...}` |

### Proposal Submission Rules

- Only during "accepting" round state (same window as action submissions)
- One proposal per agent per round (enforced by registry)
- Code must define `propose_modification(execution_engine)` function (validated before storing)
- Idempotency key prevents duplicate submissions

### Files

- Create: `kernel/mechanism_registry.py` — MechanismRecord, MechanismRegistry
- Create: `api/routes/mechanisms.py` — proposal + listing endpoints
- Modify: `api/models.py` — MechanismProposeRequest, MechanismProposeResponse, MechanismResponse
- Modify: `api/app.py` — include mechanisms router
- Modify: `api/deps.py` — add get_mechanism_registry accessor

## Component 3: Judge Integration + Metrics (resolves K5)

### JudgeAdapter

A thin wrapper that runs the existing `Judge` class in a subprocess worker.

**Why subprocess?** The Judge calls an LLM via litellm which can hang, crash, or take unbounded time. Running it in a subprocess with a timeout (default 30s) prevents the sim thread from blocking.

```python
@dataclass
class JudgmentResult:
    approved: bool
    reason: str
    latency_ms: float
    error: Optional[str] = None

class JudgeAdapter:
    def __init__(self, timeout: float = 30.0, use_dummy: bool = False):
        self.timeout = timeout
        self.use_dummy = use_dummy

    def evaluate(self, code: str, proposer_id: int,
                 proposal_type: str, context: dict = None) -> JudgmentResult:
        """Run judge in subprocess. Returns JudgmentResult.
        On timeout/crash: returns rejected (fail-closed)."""

class DummyJudge:
    """Always approves. For testing and fast iteration."""
    def evaluate(self, code, proposer_id, proposal_type, context=None):
        return JudgmentResult(approved=True, reason="dummy", latency_ms=0.0)
```

### Where Judge Fits in the Loop

1. `propose_mechanisms` phase → agents submit proposals via API → stored in MechanismRegistry as "submitted"
2. `judge` phase → `JudgeAdapter.evaluate()` each pending proposal → approved proposals move to "approved", rejected get "rejected" status with reason
3. `execute_mechanisms` phase → runs code of "approved" mechanisms, moves them to "active"

Agent actions submitted via the Phase 2 write path are **not** judged (they run through SubprocessSandbox which already constrains them). Judging actions is opt-in for Phase 4.

### Round Metrics

Computed at the end of `settle_round()` from execution state:

```python
round_metrics = {
    "total_vitality": float,      # sum of all member vitality
    "gini_coefficient": float,    # wealth inequality (0=equal, 1=max)
    "trade_volume": int,          # number of offers this round
    "conflict_count": int,        # number of attacks this round
    "mechanism_proposals": int,   # proposals submitted this round
    "mechanism_approvals": int,   # proposals approved this round
    "population": int,            # active members
}
```

Gini coefficient uses the standard formula over member vitality values.

### Judge Results in Receipts

```python
judge_results = [
    {
        "proposal_id": str,
        "proposer_id": int,
        "approved": bool,
        "reason": str,
        "latency_ms": float
    }
]
```

### New Endpoints

| Method | Path | Auth | Response |
|--------|------|------|----------|
| `GET` | `/v1/world/metrics` | None | Latest round metrics |
| `GET` | `/v1/world/metrics/history` | None | Metrics for last N rounds |
| `GET` | `/v1/world/judge/stats` | None | Judge approval rate, recent rejections |

### Files

- Create: `kernel/judge_adapter.py` — JudgeAdapter, DummyJudge, JudgmentResult
- Create: `kernel/round_metrics.py` — compute_round_metrics()
- Create: `api/routes/metrics.py` — metrics + judge stats endpoints
- Modify: `kernel/world_kernel.py` — wire judge results + metrics into RoundReceipt
- Modify: `kernel/schemas.py` — add mechanism + metrics models if needed
- Modify: `api/models.py` — new response models for metrics/judge
- Modify: `api/app.py` — include new routers
- Modify: `api/deps.py` — add judge accessor

## Component 4: Tech Debt Documentation

Update `docs/plans/2026-02-24-implementation-compromises.md`:

- Mark **K3** (mechanism registry) as RESOLVED
- Mark **K4** (full settlement) as RESOLVED
- Mark **K5** (metrics + judge) as RESOLVED
- Document new Phase 3 compromises:
  - Judge subprocess timeout hardcoded at 30s (no dynamic tuning)
  - No LLM cost tracking per judgment
  - Gini coefficient only over vitality, not cargo or total wealth
  - MechanismRegistry is in-memory only (no persistence across restarts)
  - No mechanism rollback/deactivation (mechanisms are permanent once active)
  - Judge uses single LLM call (no multi-judge consensus or appeal)
  - No mechanism versioning (can't update an active mechanism, only propose new ones)

## Testing Strategy

### KernelDAGRunner tests
- test_dag_runner_phase_order — verify phases run in correct topological order
- test_dag_runner_skips_llm_nodes — verify analyze and agent_decisions are skipped
- test_dag_runner_multi_round — verify 3-round deterministic equivalence

### MechanismRegistry tests
- test_submit_mechanism — creates record with "submitted" status
- test_approve_reject — lifecycle transitions
- test_activate — approved → active
- test_get_pending_and_active — filtered queries
- test_one_proposal_per_agent_per_round — enforcement

### JudgeAdapter tests
- test_dummy_judge_approves_all — DummyJudge always returns approved
- test_judge_adapter_timeout — simulated slow judge returns rejected
- test_judge_adapter_success — mocked successful judgment
- test_judgment_result_in_receipt — judge results appear in RoundReceipt

### Round Metrics tests
- test_compute_metrics — verify all fields computed correctly
- test_gini_equal — equal distribution → gini = 0
- test_gini_unequal — one member has everything → gini ≈ 1
- test_metrics_in_receipt — metrics appear in RoundReceipt

### API endpoint tests
- test_propose_mechanism — accepted during window, returns mechanism_id
- test_propose_mechanism_rejected_outside_window — 400 when not accepting
- test_list_mechanisms — returns all with optional status filter
- test_get_mechanism_by_id — returns full detail
- test_metrics_endpoint — returns current round metrics
- test_metrics_history — returns N rounds of metrics
- test_judge_stats — returns approval rate and recent rejections

### Integration test
- test_full_governance_round — register agent, propose mechanism, judge approves, mechanism activates, verify in receipt with metrics

## Files Changed Summary

New files:
- `kernel/dag_runner.py`
- `kernel/mechanism_registry.py`
- `kernel/judge_adapter.py`
- `kernel/round_metrics.py`
- `api/routes/mechanisms.py`
- `api/routes/metrics.py`

Modified files:
- `kernel/world_kernel.py` (settle_round delegates to DAG runner, wire judge + metrics into receipt)
- `kernel/schemas.py` (mechanism + metrics types if needed)
- `api/models.py` (new request/response models)
- `api/app.py` (include mechanisms + metrics routers)
- `api/deps.py` (add mechanism_registry, judge accessors)
- `scripts/run_server.py` (pass judge + mechanism_registry to kernel, integrate into sim loop)
- `test_world_kernel.py` (DAG runner, judge adapter, metrics, mechanism registry tests)
- `test_api.py` (mechanism endpoint, metrics endpoint, judge stats, integration tests)
- `docs/plans/2026-02-24-implementation-compromises.md` (mark K3, K4, K5 resolved + new compromises)

Existing files unchanged:
- `MetaIsland/judge.py` — untouched (wrapped, not modified)
- `MetaIsland/execution_graph.py` — untouched (topology referenced, not imported)
- `kernel/engine_proxy.py` — untouched
- `kernel/subprocess_sandbox.py` — untouched
