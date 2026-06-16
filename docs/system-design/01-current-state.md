# Leviathan Current State Audit

Date: 2026-02-24
Scope: repository state in `/Users/lichenyu/leviathan` for design-stage planning.

## 1. System Snapshot

The repo currently contains two main simulation layers:

- `Leviathan/`: base island simulation (members, land, actions, reproduction, relationships).
- `MetaIsland/`: LLM-driven extension with agent-generated code, mechanism proposals, judge, contracts, physics, and graph execution.

Current execution orchestration is in-process and lockstep by round:

- Graph engine (`MetaIsland/graph_engine.py`) executes DAG layers.
- Default graph wiring is in `MetaIsland/metaIsland.py` (`new_round -> analyze -> propose_mechanisms -> judge -> execute_mechanisms -> agent_decisions -> execute_actions -> contracts -> produce -> consume -> environment -> log_status`).
- Nodes are in `MetaIsland/nodes/`.

## 2. What Is Already Working

### 2.1 Agent behavior loop and memory

- Agents run `analyze`, `agent_mechanism_proposal`, and `agent_code_decision`.
- Prompt generation is modularized with YAML templates (`MetaIsland/prompts/*.yaml`) and a loader (`MetaIsland/prompt_loader.py`).
- Rich memory and strategy metrics exist (signature extraction, diversity tracking, context tags, experiment outcomes) in:
  - `MetaIsland/meta_island_signature.py`
  - `MetaIsland/meta_island_population.py`
  - `MetaIsland/strategy_recommendations.py`

### 2.2 Mechanism/Judge/Contract scaffolding

- Mechanism proposals are judge-gated before execution.
- Contract system supports propose/sign/activate/execute lifecycle (`MetaIsland/contracts.py`).
- Physics constraint system exists (`MetaIsland/physics.py`).
- Judge abstraction exists (`MetaIsland/judge.py`) with model routing (`MetaIsland/model_router.py` + `config/models.yaml`).

### 2.3 Observability and evaluation

- Per-round execution history is persisted in JSON (`generated_code/execution_histories`).
- E2E smoke framework and metric aggregation exist:
  - `scripts/run_e2e_smoke.py`
  - `utils/eval_metrics.py`
  - `scripts/inspect_execution_history.py`
- Latest recorded summary (`execution_histories/e2e_smoke/latest_summary.json`) shows:
  - `mechanism_attempted_count=3`, `mechanism_approved_count=3`, `mechanism_executed_count=1`
  - high action-signature diversity (`population_signature_unique_ratio=1.0`)
  - but negative round-end survival delta (`round_end_population_avg_survival_delta=-13.57`)

## 3. Architectural Gaps vs Your Target Vision

Your target vision is a distributed, externally discoverable, multi-tenant multi-agent world where third-party agents can join quickly and safely.

### 3.1 No external protocol or discovery surface

- There is no public API server for third-party agents to join a world.
- No machine-readable discovery doc (`/.well-known/...`), no OpenAPI contract, and no agent onboarding handshake.
- Today, all agents are generated internally by the host runtime.

### 3.2 Code execution trust boundary is too weak for open participation

- Agent action code is executed with `exec(...)` in-process (`MetaIsland/metaIsland.py`).
- Mechanism code is executed with `exec(...)` in-process.
- Contract code is executed with Python builtins available (`MetaIsland/contracts.py`).
- This is acceptable for local experiments but unsafe for open internet participants.

### 3.3 Governance roles are not separated enough

- Oracle (world state transition), judge, and moderator concerns are mostly coupled in one runtime.
- Judge is currently one prompt-based approval path, not an independent policy service.
- There is no role/permission model for player, judge, observer, moderator.

### 3.4 Mechanism lifecycle is incomplete for production governance

- Physics constraints can be proposed, but runtime activation/apply behavior is not fully integrated in world settlement.
- Environment node currently reports constraint counts; it does not perform full deterministic enforcement pass.
- No formal mechanism versioning, staged rollout, or rollback pipeline.

### 3.5 Distributed operations are missing

- No per-agent authentication, quotas, tenancy boundaries, or admission control.
- No external action deadlines/SLAs, round claim/submit protocol, or commit-reveal flow.
- No event-sourced canonical ledger for independent replay/audit by external participants.

### 3.6 Reproducibility and runtime consistency issues remain

- Core metric tests pass locally (`test_eval_metrics.py`, `test_graph_system.py`).
- `test_mechanism_execution.py` currently fails at import time due local `numpy`/`matplotlib` binary mismatch, indicating environment drift risk for contributors.

## 4. Readiness Assessment

Current readiness by layer:

- Simulation kernel readiness: Medium
- Strategy/eval instrumentation readiness: High
- Open distributed protocol readiness: Low
- Security/sandbox readiness for untrusted agents: Low
- Governance and role isolation readiness: Low to Medium

## 5. Design Constraints To Preserve

The next architecture should preserve what is uniquely strong in this repo:

- Agent-level strategic diversity metrics and memory loop.
- Mechanism co-design as a first-class gameplay feature.
- Rich execution history and evaluability.
- Deterministic round semantics with explicit phase ordering.

## 6. Immediate Implications

For external onboarding and "one-shot connect" to be realistic:

- Do not expose raw in-process Python execution to external agents.
- Introduce a formal network protocol before scaling participant count.
- Separate world oracle, judge policy, and moderator controls.
- Keep coding as interaction, but route it through a constrained mechanism/action interface and sandbox.
