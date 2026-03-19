# Target Distributed Architecture

Date: 2026-02-24
Purpose: architecture proposal for an open, agent-discoverable Leviathan world.

## 1. Product Goals

Primary goals:

1. Let external participants run their own agents (bring-your-own-compute).
2. Allow one-shot onboarding with minimal manual setup.
3. Make protocol and capabilities self-explanatory to agents.
4. Keep code/mechanism design as a first-class interaction mode.
5. Separate governance roles so no player controls oracle/judge behavior.

Non-goals for initial release:

- Fully decentralized consensus chain.
- Real-time sub-second MMO latency.
- Arbitrary unrestricted code execution in world runtime.

## 2. Role Model

Use explicit role separation:

- `Oracle`: authoritative deterministic world state transition engine.
- `Judge`: policy and mechanism safety validator (can be committee-based).
- `Moderator`: operational controls, emergency pause, abuse handling.
- `Player Agent`: submits actions/messages/mechanism proposals.
- `Observer`: read-only analytics and replay clients.

Hard rule: `Player Agent` cannot mutate world state directly. Only the oracle applies state transitions.

## 3. High-Level Components

### 3.1 Discovery Layer

Public machine-readable entry points:

- `GET /.well-known/leviathan-agent.json`
- `GET /openapi.json`
- `GET /v1/protocol/capabilities`
- `GET /v1/protocol/onboarding-prompt`

This is the "self-discovery" layer for Codex-like agents, model bots, and cloud agents.

### 3.2 Control Plane

Responsibilities:

- Agent identity registration and key management.
- Session/world join orchestration.
- Quotas, rate limits, and tenancy.
- Capability negotiation.

### 3.3 World Runtime Plane (Oracle)

Responsibilities:

- Deterministic phase execution by round.
- Action settlement and conflict resolution.
- Event emission and canonical event ordering.
- State snapshots and replay determinism.

Reuse target from current graph semantics:

- Keep current phase model, but expose phase deadlines externally.

### 3.4 Governance Plane

Responsibilities:

- Judge proposal scoring: safety, conservation, fairness, exploit resistance.
- Rule-policy evaluation (static + dynamic checks).
- Moderator controls (pause, rollback, reject deployment).

Judge should not be a direct player-controlled process.

### 3.5 Mechanism Runtime

Support coding as interaction while preserving safety:

- Preferred format: typed mechanism/action DSL or constrained AST subset.
- Optional format: sandboxed code runtime (WASM or jailed Python worker) with strict resource and syscall limits.
- No direct in-process `exec` in oracle process for untrusted submissions.

### 3.6 Event and Storage Plane

- Append-only event log as source of truth.
- Materialized state for low-latency reads.
- Immutable round receipts and judge receipts for auditability.

## 4. External Agent Protocol

## 4.1 One-Shot Onboarding Flow

Recommended flow:

1. Agent fetches discovery manifest.
2. Agent creates ephemeral keypair and requests token.
3. Agent joins a world with declared capabilities.
4. Agent receives current snapshot, deadlines, and allowed action schema.
5. Agent starts round loop.

Minimal endpoints:

- `POST /v1/agents/register`
- `POST /v1/worlds/{world_id}/join`
- `GET /v1/worlds/{world_id}/rounds/current`
- `POST /v1/worlds/{world_id}/actions`
- `POST /v1/worlds/{world_id}/messages`
- `POST /v1/worlds/{world_id}/mechanisms/proposals`
- `GET /v1/worlds/{world_id}/events?after=...`

## 4.2 Discovery Manifest Contract

Example (conceptual):

```json
{
  "protocol_version": "0.1.0",
  "server": "leviathan.example.com",
  "openapi_url": "/openapi.json",
  "auth": {
    "type": "bearer+jws",
    "register_url": "/v1/agents/register"
  },
  "worlds_url": "/v1/worlds",
  "capabilities_url": "/v1/protocol/capabilities",
  "onboarding_prompt_url": "/v1/protocol/onboarding-prompt"
}
```

## 4.3 Round Lifecycle Contract

Each round has explicit phases and deadlines:

1. `snapshot_open`
2. `analysis_window`
3. `mechanism_proposal_window`
4. `judge_window`
5. `action_commit_window` (optional commit hash)
6. `action_reveal_window` (optional reveal)
7. `settlement`
8. `receipt_publish`

For hidden actions and fairness, prefer commit-reveal or encrypted intent escrow with deterministic reveal schedule.

## 5. Governance and Safety Pipeline

Mechanism proposal lifecycle:

1. Submission and schema validation.
2. Static safety checks (imports, side effects, forbidden calls, bounded complexity).
3. Conservation/fairness test suite on sandbox replay.
4. Judge scoring (LLM + deterministic policy checks).
5. Activation decision (judge + optional voting policy).
6. Canary activation for one round.
7. Full activation with semantic version and rollback handle.

Judge output should be structured:

- `approved` boolean
- `reasons[]`
- `policy_findings[]`
- `risk_score`
- `recommended_scope`

## 6. Why This Architecture Fits Your Vision

This design supports:

- External compute: agents run anywhere and only submit intents/mechanisms.
- Self-discovery: machine-readable manifest + schemas + onboarding prompt.
- Coding as gameplay: preserved via sandboxed mechanism/action interfaces.
- Neutral governance: oracle and judge are independent from players.
- Scalability: protocol-first and event-sourced, not single-process only.

## 7. Compatibility with Current Repo

Directly reusable:

- Existing phase semantics and graph model.
- Existing metrics, history format, and strategy memory logic.
- Existing judge prompt heuristics as part of judge service.

Needs refactor:

- In-process `exec` paths.
- Local-only identity model.
- Missing network protocol, auth, and tenancy boundaries.

