# Leviathan — Claude Code Project Guide

## What This Project Is

Leviathan is an agent-based social evolution simulation exploring how simple individual decisions converge into complex social systems (Hobbesian Leviathan). It is transitioning from a single-process research tool to a **distributed, externally discoverable, multi-agent world** where third-party agents can participate safely.

## Repository Layout

```
Leviathan/          # Base simulation kernel (Island, Member, Land, IslandExecution)
MetaIsland/         # LLM-driven extension layer
  metaIsland.py     # Main orchestrator (IslandExecution with all mixins)
  graph_engine.py   # DAG-based phase execution engine
  judge.py          # LLM-based mechanism approval
  contracts.py      # Contract propose/sign/activate/execute lifecycle
  physics.py        # Physics constraint system
  model_router.py   # LLM model routing (reads config/models.yaml)
  prompt_loader.py  # YAML prompt template system
  prompts/          # YAML prompt templates
  nodes/            # Graph execution nodes
  analyze.py        # Agent analysis phase
  agent_code_decision.py      # Agent action code generation
  agent_mechanism_proposal.py # Agent mechanism proposals
  meta_island_signature.py    # Signature extraction and diversity tracking
  meta_island_population.py   # Population metrics and memory
  meta_island_strategy.py     # Strategy metrics and recommendations
  meta_island_prompting.py    # Prompt generation for agents
  strategy_recommendations.py # Strategy recommendation engine
  llm_client.py     # LLM API client
  llm_utils.py      # LLM utility functions
  base_island.py    # MetaIsland base (extends Leviathan/Island)
  base_member.py    # MetaIsland member
  base_land.py      # MetaIsland land
config/
  models.yaml       # LLM model routing config (provider + model_id)
scripts/
  run_e2e_smoke.py  # End-to-end smoke test (requires live LLM)
  inspect_execution_history.py  # Execution history inspector
  llm_access_check.py          # LLM connectivity check
utils/
  eval_metrics.py   # Evaluation metrics aggregation
  error_tags.py     # Error classification
docs/
  system-design/    # Architecture design documents (6 docs, current -> distributed)
  design/           # Historical design documents (ARCHITECTURE.md, etc.)
assets/             # Images and visual assets
archive/            # Archived artifacts (codex logs, stale dirs, analysis dumps)
```

## Key Simulation Concepts

- **Round-based execution**: deterministic phases wired through a DAG graph engine
- **Phase order**: `new_round -> analyze -> propose_mechanisms -> judge -> execute_mechanisms -> agent_decisions -> execute_actions -> contracts -> produce -> consume -> environment -> log_status`
- **Agents can propose mechanisms and code** — not just act, but modify world rules
- **Judge-gated governance**: mechanism proposals must pass LLM-based judge approval before execution
- **Execution histories**: per-round JSON persistence in `execution_histories/`

## Architecture Vision (from docs/system-design/)

The project is evolving toward:
- **Role separation**: Oracle (deterministic state transitions), Judge (policy validator), Moderator (ops controls), Player Agent (submits actions), Observer (read-only)
- **Three-layer constitutional model**: immutable kernel (determinism, safety boundaries), amendable governance policy (judge rubrics, lifecycle params), fully open world rules (mechanisms, contracts, parameters)
- **External agent protocol**: discovery manifest, onboarding flow, round lifecycle with phase deadlines
- **Event-sourced ledger**: append-only event log, canonical receipts, deterministic replay

## Development Commands

```bash
# Run tests (offline, no LLM needed)
pytest -q test_graph_system.py
pytest -q test_eval_metrics.py

# Run all tests
pytest

# E2E smoke test (requires live LLM — see README for env setup)
python scripts/run_e2e_smoke.py

# Check LLM connectivity
python scripts/llm_access_check.py
```

## Environment Setup

- Python 3.7+
- Dependencies: `pip install -r requirements.txt`
- LLM access requires `.env` with API keys (OPENROUTER_API_KEY or OPENAI_API_KEY)
- Model routing configured in `config/models.yaml`

## Code Conventions

- **Bilingual comments**: the project uses both English and Chinese; preserve whichever language existing code uses
- **Async patterns**: MetaIsland agent operations are async (`async/await`)
- **Mixin-based architecture**: `IslandExecution` in MetaIsland composes multiple mixin classes for modularity
- **YAML prompts**: agent prompts live in `MetaIsland/prompts/*.yaml`, loaded via `prompt_loader.py`
- **Tests in `tests/`**: test files are `test_*.py` in the `tests/` directory (configured via `testpaths` in `pytest.ini`)

## Known Issues

- `test_mechanism_execution.py` may fail due to numpy/matplotlib binary mismatch — environment-dependent
- In-process code execution for agent/mechanism code is unsafe for external participation — being addressed in architecture redesign

## Important Constraints

- **Never run untrusted player code in the oracle process** (security boundary for distributed architecture)
- **Deterministic replay is non-negotiable** — same seed + same events must produce same state hash
- **Preserve existing phase semantics** when refactoring — the DAG phase order is a protocol specification
- **Event log integrity**: append-only, canonical hashing, no nondeterministic sources during settlement

## Current Branch

The `clawbot-self-improve` branch is the active development branch. `master` is the main branch.
