# Implementation Blueprint

Date: 2026-02-24
Purpose: concrete migration plan from current MetaIsland runtime to distributed agent-native architecture.

## 1. Guiding Principles

1. Preserve simulation semantics first, then distribute.
2. Keep deterministic replay as a non-negotiable property.
3. Never execute untrusted player code in the oracle process.
4. Ship thin vertical slices that are testable end-to-end.

## 2. Phased Delivery Plan

## Phase 0: Stabilize and isolate core runtime (1-2 weeks)

Deliverables:

- Extract a pure `WorldKernel` service from current `IslandExecution` mutation paths.
- Define explicit phase interfaces:
  - `analyze_inputs`
  - `mechanism_inputs`
  - `action_inputs`
  - `settlement_outputs`
- Add deterministic round receipt object with:
  - world id
  - round id
  - seed
  - action hashes
  - mechanism ids
  - state hash
- Freeze and document the current action schema.

Code hotspots to refactor first:

- `MetaIsland/metaIsland.py` (`execute_code_actions`, `execute_mechanism_modifications`)
- `MetaIsland/nodes/system_nodes.py`
- `MetaIsland/contracts.py`

## Phase 1: Public read API and event feed (2-3 weeks)

Deliverables:

- FastAPI service wrapper around current runtime.
- Read-only endpoints:
  - `GET /v1/worlds`
  - `GET /v1/worlds/{id}`
  - `GET /v1/worlds/{id}/rounds/current`
  - `GET /v1/worlds/{id}/events`
  - `GET /v1/worlds/{id}/snapshot`
- `/.well-known/leviathan-agent.json` discovery endpoint.
- Event envelope schema v0.

Storage:

- Start with Postgres tables:
  - `world_events`
  - `world_round_receipts`
  - `world_snapshots`
  - `agent_sessions`

## Phase 2: External write path and agent onboarding (2-4 weeks)

Deliverables:

- Auth and registration:
  - `POST /v1/agents/register`
  - `POST /v1/worlds/{id}/join`
- Action and message submission:
  - `POST /v1/worlds/{id}/actions`
  - `POST /v1/worlds/{id}/messages`
- Deadline-aware round acceptance:
  - reject late submissions
  - idempotency keys for retries
- Capability negotiation:
  - supported action tags
  - max payload size
  - mechanism support level

SDK:

- Minimal Python SDK in `sdk/python/` with:
  - auto-discovery from `/.well-known`
  - typed request objects
  - round poll/submit helper

## Phase 3: Mechanism governance service (3-5 weeks)

Deliverables:

- Separate judge worker service:
  - policy checks
  - deterministic lint checks
  - LLM reasoning checks
- Mechanism proposal API:
  - `POST /v1/worlds/{id}/mechanisms/proposals`
  - `GET /v1/worlds/{id}/mechanisms/{proposal_id}`
- Activation pipeline states:
  - `submitted`
  - `lint_failed`
  - `judge_failed`
  - `canary_active`
  - `active`
  - `rolled_back`

Safety model:

- For v1, accept only constrained mechanism DSL or vetted AST subset.
- Defer arbitrary Python to isolated sandbox workers with strict limits.

## Phase 4: Moderator and multi-tenant controls (2-3 weeks)

Deliverables:

- Moderator APIs:
  - pause world
  - force rollback
  - ban session
  - adjust quotas
- Tenant isolation:
  - per-world quotas
  - per-agent rate limits
  - abuse detection signals

## 3. Protocol and Schema Details (v0)

Core write payloads:

- `ActionIntent`
  - `agent_id`
  - `round_id`
  - `phase`
  - `actions[]` (typed tags and args)
  - `signature`
  - `idempotency_key`

- `MechanismProposal`
  - `proposal_id`
  - `agent_id`
  - `world_id`
  - `scope`
  - `code_or_dsl`
  - `declared_resources`
  - `expected_invariants[]`
  - `rollback_plan`

- `RoundReceipt`
  - `round_id`
  - `snapshot_hash_before`
  - `snapshot_hash_after`
  - `accepted_action_ids[]`
  - `rejected_action_ids[]`
  - `activated_mechanism_ids[]`
  - `judge_receipt_ids[]`

## 4. Mapping Existing Modules to Target Services

- `MetaIsland/base_island.py` and domain logic -> `world-kernel` package.
- `MetaIsland/graph_engine.py` -> deterministic phase orchestrator in oracle service.
- `MetaIsland/judge.py` -> judge worker adapter (service boundary).
- `MetaIsland/contracts.py` -> contract subsystem under oracle settlement.
- `utils/eval_metrics.py` and smoke scripts -> evaluation/observability pipeline.

## 5. Security Baseline (must-have before open beta)

1. Signed agent requests and replay protection.
2. Strong payload validation with JSON Schema.
3. No direct Python `exec` from external submissions in oracle process.
4. Sandbox resource limits for optional code execution.
5. Full append-only audit log for moderation and postmortem.

## 6. Testing Plan by Phase

Phase-gated tests:

1. Determinism tests: same seed + same events => same state hash.
2. Protocol tests: join, action submit, deadline enforcement, idempotency.
3. Governance tests: mechanism rejection reasons and rollback correctness.
4. Load tests: N agents submitting concurrently by deadline windows.
5. Replay tests: rebuild snapshot from event log without divergence.

Keep current tests and add:

- API contract tests
- event replay golden tests
- judge policy regression suite

Note on current environment:

- `pytest -q test_graph_system.py` passes.
- `pytest -q test_eval_metrics.py` passes.
- `test_mechanism_execution.py` currently fails in this environment due `numpy`/`matplotlib` binary incompatibility. Fix environment lockfiles before using this as CI baseline.

## 7. Recommended Decision Points for the Team

Decide early:

1. Mechanism language strategy:
   - constrained DSL first, or sandboxed Python first.
2. Identity strategy:
   - API keys only, or signed keypairs from day one.
3. Fairness strategy:
   - simple lockstep deadlines now, commit-reveal now, or commit-reveal later.
4. Judge strategy:
   - single model + policy checks now, or multi-judge committee now.

## 8. First Build Slice (practical next sprint)

Implement this first:

1. `/.well-known/leviathan-agent.json`
2. `GET /v1/worlds/{id}/rounds/current`
3. `POST /v1/worlds/{id}/actions` with schema validation and receipts
4. deterministic event log + replay check for one world

This gives immediate external-agent interoperability without waiting for full mechanism sandboxing.

