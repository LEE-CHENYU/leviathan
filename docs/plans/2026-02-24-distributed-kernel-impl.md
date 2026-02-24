# WorldKernel Extraction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract a pure WorldKernel facade from IslandExecution with typed schemas, deterministic receipt hashing, and golden tests proving semantic equivalence.

**Architecture:** Facade pattern — new `kernel/` package wraps existing `MetaIsland/metaIsland.py:IslandExecution` without modifying it. Schemas are Python dataclasses. Receipt hashing uses canonical JSON + SHA-256. An `ExecutionSandbox` protocol abstracts code execution for future swapability.

**Tech Stack:** Python 3.7+ dataclasses, hashlib, json, typing. No new dependencies.

---

### Task 1: Create kernel package and schemas

**Files:**
- Create: `kernel/__init__.py`
- Create: `kernel/schemas.py`
- Create: `test_world_kernel.py`

**Step 1: Write the failing test**

Create `test_world_kernel.py` at the repo root (matching existing test layout per `pytest.ini`). Include tests for all schema dataclasses: `WorldConfig`, `ActionIntent`, `ActionResult`, `MechanismProposal`, `MechanismResult`, `WorldSnapshot`, `RoundReceipt`. Test creation, default values, and `dataclasses.asdict()` round-trip.

**Step 2: Run test to verify it fails**

Run: `pytest test_world_kernel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kernel'`

**Step 3: Write minimal implementation**

Create `kernel/__init__.py` (empty docstring) and `kernel/schemas.py` with all 7 dataclasses as specified in the design doc (`docs/plans/2026-02-24-distributed-kernel-design.md`, Schemas section).

**Step 4: Run test to verify it passes**

Run: `pytest test_world_kernel.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add kernel/__init__.py kernel/schemas.py test_world_kernel.py
git commit -m "feat: add kernel schemas (WorldConfig, ActionIntent, RoundReceipt, etc.)"
```

---

### Task 2: Add receipt hashing module

**Files:**
- Create: `kernel/receipt.py`
- Modify: `test_world_kernel.py` (append tests)

**Step 1: Write the failing test**

Append tests for `canonical_json`, `compute_state_hash`, and `compute_receipt_hash`. Test: sorted-key determinism, no-whitespace guarantee, identical-bytes-for-identical-input, SHA-256 hex digest length (64 chars), and that different data produces different hashes.

**Step 2: Run test to verify it fails**

Run: `pytest test_world_kernel.py::test_canonical_json_sorted_keys -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `kernel/receipt.py` with:
- `canonical_json(obj) -> bytes` — `json.dumps` with `sort_keys=True, separators=(',',':'), ensure_ascii=False, default=str`, encoded to UTF-8
- `compute_state_hash(snapshot) -> str` — SHA-256 of canonical snapshot dict (excluding `state_hash` field itself)
- `compute_receipt_hash(receipt) -> str` — SHA-256 of canonical receipt dict

**Step 4: Run test to verify it passes**

Run: `pytest test_world_kernel.py -v`
Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add kernel/receipt.py test_world_kernel.py
git commit -m "feat: add deterministic receipt hashing (canonical JSON + SHA-256)"
```

---

### Task 3: Add ExecutionSandbox interface and InProcessSandbox

**Files:**
- Create: `kernel/execution_sandbox.py`
- Modify: `test_world_kernel.py` (append tests)

**Step 1: Write the failing test**

Append tests for `InProcessSandbox`:
- `test_sandbox_execute_simple_code` — code defines `agent_action`, sandbox returns `SandboxResult(success=True)`
- `test_sandbox_captures_error` — code raises ValueError, sandbox returns `success=False` with error message
- `test_sandbox_no_agent_action` — code doesn't define `agent_action`, sandbox returns error
- `test_sandbox_mechanism_code` — code defines `propose_modification`, sandbox returns success

**Step 2: Run test to verify it fails**

Run: `pytest test_world_kernel.py::test_sandbox_execute_simple_code -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `kernel/execution_sandbox.py` with:
- `SandboxContext` dataclass (execution_engine, member_index, extra_env)
- `SandboxResult` dataclass (success, error, traceback_str)
- `ExecutionSandbox` Protocol with `execute_agent_code` and `execute_mechanism_code`
- `InProcessSandbox` class implementing the protocol:
  - Builds restricted namespace with `np`, `math`, `execution_engine`
  - Compiles and runs code string, looks for expected function name
  - Calls function if engine is not None
  - Catches all exceptions and returns structured result

Note: `InProcessSandbox` uses Python's `compile()` + in-process code running, matching the existing `MetaIsland/metaIsland.py` execution model. The `ExecutionSandbox` protocol exists specifically to swap this out for subprocess/WASM sandboxing in later phases.

**Step 4: Run test to verify it passes**

Run: `pytest test_world_kernel.py -v`
Expected: All 17 tests PASS

**Step 5: Commit**

```bash
git add kernel/execution_sandbox.py test_world_kernel.py
git commit -m "feat: add ExecutionSandbox interface and InProcessSandbox"
```

---

### Task 4: Implement WorldKernel facade — constructor and get_snapshot

**Files:**
- Create: `kernel/world_kernel.py`
- Modify: `kernel/__init__.py` (add exports)
- Modify: `test_world_kernel.py` (append tests)

**Step 1: Write the failing test**

Append tests:
- `test_world_kernel_init` — creates WorldKernel with config + tmpdir, asserts `round_id == 0`
- `test_world_kernel_get_snapshot` — asserts WorldSnapshot has correct member count, round_id, 64-char hash
- `test_world_kernel_snapshot_deterministic` — two kernels with same seed produce identical `state_hash`

Uses `tempfile.TemporaryDirectory()` for save_path.

**Step 2: Run test to verify it fails**

Run: `pytest test_world_kernel.py::test_world_kernel_init -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `kernel/world_kernel.py` with:
- Lazy import of `MetaIsland.metaIsland.IslandExecution` (avoids circular imports)
- `__init__` creates `IslandExecution` internally with config params
- `round_id` property
- `get_snapshot()` collects member states, land, mechanisms, contracts, physics from the execution engine, builds `WorldSnapshot`, computes `state_hash` via `compute_state_hash`

Update `kernel/__init__.py` to export `WorldKernel` and all schema types.

**Step 4: Run test to verify it passes**

Run: `pytest test_world_kernel.py -v`
Expected: All 20 tests PASS

**Step 5: Run existing tests to verify no regression**

Run: `pytest test_graph_system.py test_eval_metrics.py -v`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add kernel/ test_world_kernel.py
git commit -m "feat: add WorldKernel facade with get_snapshot and deterministic hashing"
```

---

### Task 5: Implement begin_round, accept_actions, accept_mechanisms, settle_round

**Files:**
- Modify: `kernel/world_kernel.py` (add methods)
- Modify: `test_world_kernel.py` (append tests)

**Step 1: Write the failing tests**

Append tests:
- `test_accept_actions_basic` — submit one no-op action, assert success
- `test_accept_actions_error_handling` — submit bad code, assert `success=False` with error message
- `test_accept_actions_idempotency` — submit same idempotency_key twice, assert no duplicate execution
- `test_accept_mechanisms_basic` — submit one no-op mechanism, assert `executed=True`
- `test_settle_round_produces_receipt` — assert receipt has correct round_id, seed, valid hashes
- `test_full_round_lifecycle` — begin_round -> accept_actions -> settle_round end-to-end

**Step 2: Run tests to verify they fail**

Run: `pytest test_world_kernel.py::test_accept_actions_basic -v`
Expected: FAIL with `AttributeError: 'WorldKernel' object has no attribute 'begin_round'`

**Step 3: Write minimal implementation**

Add to `WorldKernel`:
- `begin_round()` — increments round_id, calls `self._execution.new_round()`, resets idempotency cache
- `accept_actions(actions)` — for each action: check idempotency, resolve agent index, capture old_stats, run via sandbox, capture new_stats, compute performance_change, return ActionResult
- `accept_mechanisms(mechanisms)` — for each mechanism: run via sandbox, return MechanismResult
- `settle_round(seed)` — capture snapshot_before, run `produce()` + `consume()`, capture snapshot_after, build RoundReceipt with deterministic timestamp (derived from hash, not wall clock)
- `get_round_receipt()` — return last receipt
- `_resolve_agent_index(agent_id)` — find member index by id

**Step 4: Run test to verify it passes**

Run: `pytest test_world_kernel.py -v`
Expected: All 26 tests PASS

**Step 5: Commit**

```bash
git add kernel/world_kernel.py test_world_kernel.py
git commit -m "feat: add begin_round, accept_actions, accept_mechanisms, settle_round"
```

---

### Task 6: Golden determinism tests

**Files:**
- Modify: `test_world_kernel.py` (append golden tests)

**Step 1: Write the golden tests**

Add helper `_run_deterministic_round(seed, member_count, land)` and three golden tests:
- `test_golden_determinism_same_seed` — identical seed + inputs produce identical `snapshot_hash_before`, `snapshot_hash_after`, and final `state_hash`
- `test_golden_determinism_different_seed` — different seeds produce different initial hashes
- `test_golden_determinism_multiple_rounds` — 3 rounds with identical inputs produce identical final state across two independent runs

**Step 2: Run golden tests**

Run: `pytest test_world_kernel.py -k golden -v`
Expected: All 3 golden tests PASS

**Step 3: Run ALL tests (new + existing)**

Run: `pytest test_world_kernel.py test_graph_system.py test_eval_metrics.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add test_world_kernel.py
git commit -m "feat: add golden determinism tests for WorldKernel"
```

---

### Task 7: Final verification and cleanup

**Files:** None new — verification only

**Step 1: Run all existing tests**

Run: `pytest test_graph_system.py test_eval_metrics.py test_code_cleaning.py test_llm_utils.py test_prompting.py -v`
Expected: All existing tests PASS (note: `test_mechanism_execution.py` may fail due to known pre-existing numpy/matplotlib issue — this is not a regression)

**Step 2: Run the full kernel test suite**

Run: `pytest test_world_kernel.py -v --tb=short`
Expected: All kernel tests PASS

**Step 3: Verify kernel package imports cleanly**

Run: `python -c "from kernel import WorldKernel, WorldConfig; print('OK')"`
Expected: prints `OK`

**Step 4: Final commit**

```bash
git add -A
git status
git commit -m "chore: finalize Phase 0 WorldKernel extraction"
```
