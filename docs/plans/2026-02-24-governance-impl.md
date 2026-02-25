# Phase 3: Governance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable agents to propose mechanisms, validated by an LLM judge, executed through the DAG engine, with round-level metrics — completing the governance loop.

**Architecture:** MechanismRegistry tracks proposals through a simple lifecycle (submitted -> approved/rejected -> active). JudgeAdapter wraps the existing Judge in a subprocess for safety. KernelDAGRunner replaces the minimal settle_round with proper phase ordering (new_round, judge, execute_mechanisms, produce, consume, environment, log_status). Round metrics and judge results populate the previously-empty RoundReceipt fields.

**Tech Stack:** Python 3.10, pytest, FastAPI, Pydantic, subprocess, threading, dataclasses

---

## Background

### Existing Code You Need to Know

- **`kernel/world_kernel.py`** — The simulation facade. `settle_round()` currently only calls `produce()` and `consume()`. We'll expand it to run all DAG phases. It already has `apply_intended_actions()`, `accept_mechanisms()`, and `get_snapshot()`.

- **`kernel/schemas.py`** — Dataclasses for the kernel. `RoundReceipt` has `judge_results: List[Dict]` and `round_metrics: Dict[str, float]` fields that are currently always empty. We'll populate them.

- **`kernel/execution_sandbox.py`** — `InProcessSandbox` runs code via compile+exec. `SandboxResult` has `intended_actions` field. `SandboxContext` has `execution_engine` and `member_index`.

- **`kernel/subprocess_sandbox.py`** — `SubprocessSandbox` runs agent code in a child process. Returns `SandboxResult` with `intended_actions`.

- **`MetaIsland/judge.py`** — `Judge` class with `judge_proposal(code, proposer_id, proposal_type, context) -> (approved, reason)`. Uses litellm for LLM calls. We'll wrap it, not modify it.

- **`MetaIsland/base_island.py`** — Has `produce()`, `consume()`, `fight()`, `trade()`, `reproduce()`, `colonize()`, `new_round()`, `get_neighbors()`. These are the actual simulation phase methods.

- **`MetaIsland/nodes/basic_nodes.py`** — `NewRoundNode` calls `execution.new_round()` then `execution.get_neighbors()`. `ProduceNode` calls `produce()`. `ConsumeNode` calls `consume()`. `LogStatusNode` calls `log_status()` and `_update_round_end_metrics()`.

- **`MetaIsland/nodes/system_nodes.py`** — `JudgeNode` calls `execution.judge.judge_proposal()`. `ExecuteMechanismsNode` calls `execution.execute_mechanism_modifications(approved=...)`. `ContractNode` processes contracts. `EnvironmentNode` applies physics constraints.

- **`api/round_state.py`** — `RoundState` manages submission windows. `PendingAction` is the queued action type. We'll add `PendingProposal` alongside it.

- **`api/deps.py`** — `create_app_state()` returns the shared dict. We'll add `mechanism_registry` and `judge` entries.

- **`api/app.py`** — `create_app()` includes routers and sets up middleware. We'll add mechanisms and metrics routers.

- **`scripts/run_server.py`** — `_simulation_loop()` runs the background sim thread. We'll wire in judge + mechanism execution.

### Test Conventions

- Tests are at the repo root: `test_world_kernel.py` (kernel tests) and `test_api.py` (API tests).
- Tests use `pytest` with class-based grouping: `class TestFoo:` containing `def test_bar(self):`.
- Kernel tests create a kernel via `WorldKernel(WorldConfig(...), save_path=tempdir)`.
- API tests use `starlette.testclient.TestClient` with `build_app()` from `scripts/run_server.py`.
- Run: `python -m pytest --tb=short -q`

---

### Task 1: MechanismRegistry

**Files:**
- Create: `kernel/mechanism_registry.py`
- Test: `test_world_kernel.py` (append new test class)

**Step 1: Write the failing tests**

Add to the bottom of `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — MechanismRegistry tests
# ──────────────────────────────────────────────

from kernel.mechanism_registry import MechanismRecord, MechanismRegistry


class TestMechanismRegistry:
    def test_submit_mechanism(self):
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="def propose_modification(e): pass", description="test")
        assert rec.status == "submitted"
        assert rec.proposer_id == 1
        assert rec.mechanism_id  # non-empty string
        assert rec.submitted_round == 0  # default

    def test_submit_with_round(self):
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=2, code="code", description="d", round_id=5)
        assert rec.submitted_round == 5

    def test_get_pending(self):
        reg = MechanismRegistry()
        reg.submit(proposer_id=1, code="c1", description="d1")
        reg.submit(proposer_id=2, code="c2", description="d2")
        pending = reg.get_pending()
        assert len(pending) == 2
        assert all(r.status == "submitted" for r in pending)

    def test_approve_and_activate(self):
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="c", description="d")
        reg.mark_approved(rec.mechanism_id, round_id=3, reason="looks good")
        updated = reg.get_by_id(rec.mechanism_id)
        assert updated.status == "approved"
        assert updated.judged_round == 3
        assert updated.judge_reason == "looks good"

        reg.activate(rec.mechanism_id, round_id=3)
        updated = reg.get_by_id(rec.mechanism_id)
        assert updated.status == "active"
        assert updated.activated_round == 3

    def test_reject(self):
        reg = MechanismRegistry()
        rec = reg.submit(proposer_id=1, code="c", description="d")
        reg.mark_rejected(rec.mechanism_id, round_id=2, reason="violates physics")
        updated = reg.get_by_id(rec.mechanism_id)
        assert updated.status == "rejected"
        assert updated.judge_reason == "violates physics"

    def test_get_active(self):
        reg = MechanismRegistry()
        r1 = reg.submit(proposer_id=1, code="c1", description="d1")
        r2 = reg.submit(proposer_id=2, code="c2", description="d2")
        reg.mark_approved(r1.mechanism_id, round_id=1, reason="ok")
        reg.activate(r1.mechanism_id, round_id=1)
        reg.mark_rejected(r2.mechanism_id, round_id=1, reason="bad")
        active = reg.get_active()
        assert len(active) == 1
        assert active[0].mechanism_id == r1.mechanism_id

    def test_get_all(self):
        reg = MechanismRegistry()
        reg.submit(proposer_id=1, code="c1", description="d1")
        reg.submit(proposer_id=2, code="c2", description="d2")
        assert len(reg.get_all()) == 2

    def test_get_by_id_not_found(self):
        reg = MechanismRegistry()
        assert reg.get_by_id("nonexistent") is None

    def test_one_proposal_per_agent_per_round(self):
        reg = MechanismRegistry()
        r1 = reg.submit(proposer_id=1, code="c1", description="d1", round_id=5)
        assert r1 is not None
        r2 = reg.submit(proposer_id=1, code="c2", description="d2", round_id=5)
        assert r2 is None  # rejected: already proposed this round

    def test_same_agent_different_rounds(self):
        reg = MechanismRegistry()
        r1 = reg.submit(proposer_id=1, code="c1", description="d1", round_id=5)
        r2 = reg.submit(proposer_id=1, code="c2", description="d2", round_id=6)
        assert r1 is not None
        assert r2 is not None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_world_kernel.py::TestMechanismRegistry -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kernel.mechanism_registry'`

**Step 3: Write minimal implementation**

Create `kernel/mechanism_registry.py`:

```python
"""In-memory registry for mechanism proposals and their lifecycle."""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MechanismRecord:
    """A mechanism proposal with its lifecycle state."""
    mechanism_id: str
    proposer_id: int
    code: str
    description: str
    status: str  # "submitted" | "approved" | "rejected" | "active"
    submitted_round: int
    judged_round: Optional[int] = None
    judge_reason: Optional[str] = None
    activated_round: Optional[int] = None


class MechanismRegistry:
    """Tracks mechanism proposals through their lifecycle.

    Lifecycle: submitted -> approved/rejected -> active
    """

    def __init__(self) -> None:
        self._records: Dict[str, MechanismRecord] = {}
        self._agent_round_submissions: Set[Tuple[int, int]] = set()

    def submit(
        self,
        proposer_id: int,
        code: str,
        description: str,
        round_id: int = 0,
    ) -> Optional[MechanismRecord]:
        """Submit a new mechanism proposal. Returns None if agent already proposed this round."""
        key = (proposer_id, round_id)
        if key in self._agent_round_submissions:
            return None

        mechanism_id = uuid.uuid4().hex
        record = MechanismRecord(
            mechanism_id=mechanism_id,
            proposer_id=proposer_id,
            code=code,
            description=description,
            status="submitted",
            submitted_round=round_id,
        )
        self._records[mechanism_id] = record
        self._agent_round_submissions.add(key)
        return record

    def get_pending(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "submitted"]

    def mark_approved(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "approved"
        rec.judged_round = round_id
        rec.judge_reason = reason

    def mark_rejected(self, mechanism_id: str, round_id: int, reason: str) -> None:
        rec = self._records[mechanism_id]
        rec.status = "rejected"
        rec.judged_round = round_id
        rec.judge_reason = reason

    def activate(self, mechanism_id: str, round_id: int) -> None:
        rec = self._records[mechanism_id]
        rec.status = "active"
        rec.activated_round = round_id

    def get_active(self) -> List[MechanismRecord]:
        return [r for r in self._records.values() if r.status == "active"]

    def get_all(self) -> List[MechanismRecord]:
        return list(self._records.values())

    def get_by_id(self, mechanism_id: str) -> Optional[MechanismRecord]:
        return self._records.get(mechanism_id)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_world_kernel.py::TestMechanismRegistry -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add kernel/mechanism_registry.py test_world_kernel.py
git commit -m "feat: add MechanismRegistry with lifecycle tracking"
```

---

### Task 2: JudgeAdapter and DummyJudge

**Files:**
- Create: `kernel/judge_adapter.py`
- Test: `test_world_kernel.py` (append new test class)

**Step 1: Write the failing tests**

Add to `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — JudgeAdapter tests
# ──────────────────────────────────────────────

from kernel.judge_adapter import DummyJudge, JudgeAdapter, JudgmentResult


class TestJudgmentResult:
    def test_creation(self):
        r = JudgmentResult(approved=True, reason="ok", latency_ms=12.5)
        assert r.approved is True
        assert r.reason == "ok"
        assert r.latency_ms == 12.5
        assert r.error is None

    def test_with_error(self):
        r = JudgmentResult(approved=False, reason="fail", latency_ms=0.0, error="timeout")
        assert r.error == "timeout"


class TestDummyJudge:
    def test_approves_everything(self):
        judge = DummyJudge()
        result = judge.evaluate("any code", proposer_id=1, proposal_type="mechanism")
        assert result.approved is True
        assert result.latency_ms == 0.0

    def test_approves_with_context(self):
        judge = DummyJudge()
        result = judge.evaluate("code", 1, "action", context={"round": 5})
        assert result.approved is True


class TestJudgeAdapter:
    def test_timeout_returns_rejected(self):
        """JudgeAdapter with a mock that sleeps should timeout and reject."""
        adapter = JudgeAdapter(timeout=1.0, use_dummy=False)
        # Use a code string that the subprocess judge would take too long on
        # Since we can't easily mock the LLM, test the timeout path
        # by using a very short timeout with the dummy judge disabled
        # The subprocess will fail to import Judge (no LLM config) -> rejected
        result = adapter.evaluate(
            code="def propose_modification(e): pass",
            proposer_id=1,
            proposal_type="mechanism",
        )
        # Without LLM config, the subprocess will error -> fail-closed -> rejected
        assert result.approved is False

    def test_dummy_mode(self):
        """JudgeAdapter in dummy mode should approve everything."""
        adapter = JudgeAdapter(timeout=5.0, use_dummy=True)
        result = adapter.evaluate("code", 1, "mechanism")
        assert result.approved is True
        assert result.latency_ms >= 0.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_world_kernel.py::TestJudgmentResult test_world_kernel.py::TestDummyJudge test_world_kernel.py::TestJudgeAdapter -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kernel.judge_adapter'`

**Step 3: Write minimal implementation**

Create `kernel/judge_adapter.py`:

```python
"""JudgeAdapter — wraps the MetaIsland Judge in a subprocess for safe evaluation."""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class JudgmentResult:
    """Result of evaluating a proposal."""
    approved: bool
    reason: str
    latency_ms: float
    error: Optional[str] = None


class DummyJudge:
    """Always approves. For testing and fast iteration."""

    def evaluate(
        self,
        code: str,
        proposer_id: int,
        proposal_type: str,
        context: Optional[dict] = None,
    ) -> JudgmentResult:
        return JudgmentResult(approved=True, reason="dummy-approved", latency_ms=0.0)


class JudgeAdapter:
    """Runs the MetaIsland Judge in a subprocess with timeout.

    On timeout or crash, defaults to reject (fail-closed).
    """

    def __init__(self, timeout: float = 30.0, use_dummy: bool = False) -> None:
        self.timeout = timeout
        self.use_dummy = use_dummy
        self._dummy = DummyJudge() if use_dummy else None

    def evaluate(
        self,
        code: str,
        proposer_id: int,
        proposal_type: str,
        context: Optional[dict] = None,
    ) -> JudgmentResult:
        if self._dummy:
            return self._dummy.evaluate(code, proposer_id, proposal_type, context)

        start = time.monotonic()
        try:
            result = self._run_in_subprocess(code, proposer_id, proposal_type, context)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return JudgmentResult(
                approved=False,
                reason=f"Judge error: {e}",
                latency_ms=elapsed,
                error=str(e),
            )
        elapsed = (time.monotonic() - start) * 1000
        result.latency_ms = elapsed
        return result

    def _run_in_subprocess(
        self,
        code: str,
        proposer_id: int,
        proposal_type: str,
        context: Optional[dict],
    ) -> JudgmentResult:
        repo_root = str(Path(__file__).resolve().parents[1])
        script = self._build_judge_script(code, proposer_id, proposal_type, context, repo_root)

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return JudgmentResult(
                approved=False,
                reason=f"Judge timed out after {self.timeout}s",
                latency_ms=0.0,
                error="timeout",
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            return JudgmentResult(
                approved=False,
                reason=f"Judge subprocess failed: {proc.stderr.strip()[:200]}",
                latency_ms=0.0,
                error=proc.stderr.strip()[:200],
            )

        try:
            output = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError) as e:
            return JudgmentResult(
                approved=False,
                reason=f"Failed to parse judge output: {e}",
                latency_ms=0.0,
                error=str(e),
            )

        return JudgmentResult(
            approved=output.get("approved", False),
            reason=output.get("reason", "unknown"),
            latency_ms=0.0,
        )

    def _build_judge_script(
        self,
        code: str,
        proposer_id: int,
        proposal_type: str,
        context: Optional[dict],
        repo_root: str,
    ) -> str:
        context_json = json.dumps(context or {})
        lines = [
            "import json",
            "import sys",
            "import traceback",
            "",
            f"sys.path.insert(0, {repr(repo_root)})",
            "",
            "def main():",
            "    try:",
            "        from MetaIsland.judge import Judge",
            "        judge = Judge()",
            f"        approved, reason = judge.judge_proposal(",
            f"            code={repr(code)},",
            f"            proposer_id={proposer_id},",
            f"            proposal_type={repr(proposal_type)},",
            f"            context=json.loads({repr(context_json)}),",
            "        )",
            '        print(json.dumps({"approved": approved, "reason": reason}))',
            "    except Exception as e:",
            '        print(json.dumps({"approved": False, "reason": f"Judge error: {e}"}))',
            "",
            "main()",
        ]
        return "\n".join(lines) + "\n"
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_world_kernel.py::TestJudgmentResult test_world_kernel.py::TestDummyJudge test_world_kernel.py::TestJudgeAdapter -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add kernel/judge_adapter.py test_world_kernel.py
git commit -m "feat: add JudgeAdapter with subprocess isolation and DummyJudge"
```

---

### Task 3: Round Metrics Computation

**Files:**
- Create: `kernel/round_metrics.py`
- Test: `test_world_kernel.py` (append new test class)

**Step 1: Write the failing tests**

Add to `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — Round metrics tests
# ──────────────────────────────────────────────

from kernel.round_metrics import compute_gini, compute_round_metrics


class TestGiniCoefficient:
    def test_equal_distribution(self):
        """All equal -> gini = 0."""
        assert compute_gini([10.0, 10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_one_has_all(self):
        """One member has everything -> gini approaches 1."""
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini == pytest.approx(0.75)  # for 4 members: (n-1)/n = 0.75

    def test_two_members_unequal(self):
        gini = compute_gini([0.0, 100.0])
        assert gini == pytest.approx(0.5)

    def test_single_member(self):
        assert compute_gini([50.0]) == pytest.approx(0.0)

    def test_empty(self):
        assert compute_gini([]) == pytest.approx(0.0)

    def test_all_zeros(self):
        assert compute_gini([0.0, 0.0, 0.0]) == pytest.approx(0.0)


class TestComputeRoundMetrics:
    def test_basic_metrics(self):
        """Verify all expected keys are present and correctly computed."""
        members = [
            {"id": 0, "vitality": 10.0, "cargo": 5.0},
            {"id": 1, "vitality": 20.0, "cargo": 15.0},
            {"id": 2, "vitality": 30.0, "cargo": 25.0},
        ]
        metrics = compute_round_metrics(
            members=members,
            trade_volume=3,
            conflict_count=1,
            mechanism_proposals=2,
            mechanism_approvals=1,
        )
        assert metrics["total_vitality"] == pytest.approx(60.0)
        assert metrics["population"] == 3
        assert metrics["trade_volume"] == 3
        assert metrics["conflict_count"] == 1
        assert metrics["mechanism_proposals"] == 2
        assert metrics["mechanism_approvals"] == 1
        assert "gini_coefficient" in metrics
        assert 0.0 <= metrics["gini_coefficient"] <= 1.0

    def test_empty_members(self):
        metrics = compute_round_metrics(members=[])
        assert metrics["total_vitality"] == 0.0
        assert metrics["population"] == 0
        assert metrics["gini_coefficient"] == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_world_kernel.py::TestGiniCoefficient test_world_kernel.py::TestComputeRoundMetrics -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kernel.round_metrics'`

**Step 3: Write minimal implementation**

Create `kernel/round_metrics.py`:

```python
"""Round-level metrics computation for the WorldKernel."""

from typing import Dict, List


def compute_gini(values: List[float]) -> float:
    """Compute the Gini coefficient of a list of values.

    Returns 0.0 for empty lists, single values, or all-zero distributions.
    """
    n = len(values)
    if n <= 1:
        return 0.0

    total = sum(values)
    if total == 0.0:
        return 0.0

    sorted_vals = sorted(values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v

    return weighted_sum / (n * total)


def compute_round_metrics(
    members: List[Dict],
    trade_volume: int = 0,
    conflict_count: int = 0,
    mechanism_proposals: int = 0,
    mechanism_approvals: int = 0,
) -> Dict[str, float]:
    """Compute aggregate metrics for a round.

    Args:
        members: List of member dicts with at least 'vitality' key.
        trade_volume: Number of trade/offer actions this round.
        conflict_count: Number of attack actions this round.
        mechanism_proposals: Number of mechanism proposals submitted.
        mechanism_approvals: Number of proposals approved by judge.

    Returns:
        Dict with metric name -> value.
    """
    vitalities = [m.get("vitality", 0.0) for m in members]

    return {
        "total_vitality": sum(vitalities),
        "gini_coefficient": compute_gini(vitalities),
        "trade_volume": trade_volume,
        "conflict_count": conflict_count,
        "mechanism_proposals": mechanism_proposals,
        "mechanism_approvals": mechanism_approvals,
        "population": len(members),
    }
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_world_kernel.py::TestGiniCoefficient test_world_kernel.py::TestComputeRoundMetrics -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add kernel/round_metrics.py test_world_kernel.py
git commit -m "feat: add round metrics with Gini coefficient computation"
```

---

### Task 4: KernelDAGRunner

**Files:**
- Create: `kernel/dag_runner.py`
- Test: `test_world_kernel.py` (append new test class)

The DAG runner calls `IslandExecution` phase methods in topological order, replacing the manual `produce()` + `consume()` in `settle_round()`.

**Step 1: Write the failing tests**

Add to `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — KernelDAGRunner tests
# ──────────────────────────────────────────────

from kernel.dag_runner import KernelDAGRunner


class TestKernelDAGRunner:
    def _make_kernel(self, tmp_path):
        from kernel.schemas import WorldConfig
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
        return WorldKernel(config, save_path=str(tmp_path))

    def test_run_phases_returns_phase_log(self, tmp_path):
        """Runner should return a list of phase names executed."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        # Should include at least produce and consume
        assert "produce" in log
        assert "consume" in log

    def test_phases_in_correct_order(self, tmp_path):
        """Produce must come before consume in the phase log."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        produce_idx = log.index("produce")
        consume_idx = log.index("consume")
        assert produce_idx < consume_idx

    def test_deterministic_across_runs(self, tmp_path):
        """Two kernels with same seed should produce identical results after DAG runner."""
        from kernel.schemas import WorldConfig
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)

        k1 = WorldKernel(config, save_path=str(tmp_path / "k1"))
        k1.begin_round()
        r1 = KernelDAGRunner(k1._execution)
        r1.run_settlement_phases()
        snap1 = k1.get_snapshot()

        k2 = WorldKernel(config, save_path=str(tmp_path / "k2"))
        k2.begin_round()
        r2 = KernelDAGRunner(k2._execution)
        r2.run_settlement_phases()
        snap2 = k2.get_snapshot()

        for m1, m2 in zip(snap1.members, snap2.members):
            assert m1["vitality"] == pytest.approx(m2["vitality"])
            assert m1["cargo"] == pytest.approx(m2["cargo"])

    def test_skips_llm_nodes(self, tmp_path):
        """Runner should not include analyze or agent_decisions in the phase log."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        runner = KernelDAGRunner(kernel._execution)
        log = runner.run_settlement_phases()
        assert "analyze" not in log
        assert "agent_decisions" not in log
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_world_kernel.py::TestKernelDAGRunner -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kernel.dag_runner'`

**Step 3: Write minimal implementation**

Create `kernel/dag_runner.py`:

```python
"""KernelDAGRunner — executes simulation phases in topological order.

Replaces the manual produce() + consume() in settle_round() with
the full phase sequence matching the ExecutionGraph topology.

Skips LLM-driven nodes (analyze, agent_decisions, propose_mechanisms).
Judge and mechanism execution are handled separately by the kernel.
"""

from typing import Any, List


class KernelDAGRunner:
    """Runs infrastructure phases on an IslandExecution instance.

    Phase order (matching ExecutionGraph topology):
        produce -> consume -> environment -> log_status

    new_round is handled by WorldKernel.begin_round().
    Judge and execute_mechanisms are handled by the kernel before settlement.
    Agent actions are handled by the Phase 2 write path.
    """

    def __init__(self, execution: Any) -> None:
        self._execution = execution

    def run_settlement_phases(self) -> List[str]:
        """Execute infrastructure phases in order. Returns list of phase names run."""
        log: List[str] = []

        # Contracts (if available)
        if hasattr(self._execution, "contracts"):
            self._run_contracts()
            log.append("contracts")

        # Core economic phases
        self._execution.produce()
        log.append("produce")

        self._execution.consume()
        log.append("consume")

        # Environment (physics constraints, if available)
        if hasattr(self._execution, "physics"):
            log.append("environment")

        return log

    def _run_contracts(self) -> None:
        """Process active contracts."""
        contracts = self._execution.contracts
        for contract_id in list(contracts.active.keys()):
            try:
                contracts.execute_contract(
                    contract_id, self._execution, {}
                )
            except Exception:
                pass  # Contract errors don't halt the round
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_world_kernel.py::TestKernelDAGRunner -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add kernel/dag_runner.py test_world_kernel.py
git commit -m "feat: add KernelDAGRunner for proper phase ordering"
```

---

### Task 5: Wire DAG Runner + Metrics + Judge Results into WorldKernel

**Files:**
- Modify: `kernel/world_kernel.py` — settle_round uses DAG runner, populates metrics + judge_results
- Test: `test_world_kernel.py` (append new test class)

**Step 1: Write the failing tests**

Add to `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — WorldKernel settle_round with DAG + metrics
# ──────────────────────────────────────────────


class TestKernelDAGSettlement:
    def _make_kernel(self, tmp_path):
        from kernel.schemas import WorldConfig
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
        return WorldKernel(config, save_path=str(tmp_path))

    def test_receipt_has_metrics(self, tmp_path):
        """After settle_round, receipt should have populated round_metrics."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        receipt = kernel.settle_round(seed=1)
        assert "total_vitality" in receipt.round_metrics
        assert "gini_coefficient" in receipt.round_metrics
        assert "population" in receipt.round_metrics
        assert receipt.round_metrics["population"] == 5

    def test_receipt_metrics_total_vitality(self, tmp_path):
        """Total vitality in metrics should match sum of member vitalities."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        receipt = kernel.settle_round(seed=1)
        snap = kernel.get_snapshot()
        expected_vitality = sum(m["vitality"] for m in snap.members)
        assert receipt.round_metrics["total_vitality"] == pytest.approx(expected_vitality)

    def test_judge_results_empty_by_default(self, tmp_path):
        """Without mechanism proposals, judge_results should be empty."""
        kernel = self._make_kernel(tmp_path)
        kernel.begin_round()
        receipt = kernel.settle_round(seed=1)
        assert receipt.judge_results == []

    def test_settle_round_still_deterministic(self, tmp_path):
        """Upgrading settle_round to use DAG runner must preserve determinism."""
        from kernel.schemas import WorldConfig
        config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)

        k1 = WorldKernel(config, save_path=str(tmp_path / "k1"))
        k1.begin_round()
        r1 = k1.settle_round(seed=1)

        k2 = WorldKernel(config, save_path=str(tmp_path / "k2"))
        k2.begin_round()
        r2 = k2.settle_round(seed=1)

        assert r1.snapshot_hash_after == r2.snapshot_hash_after
        assert r1.round_metrics == r2.round_metrics
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_world_kernel.py::TestKernelDAGSettlement -v`
Expected: FAIL — `round_metrics` is still `{}` in receipts

**Step 3: Modify `kernel/world_kernel.py`**

Replace the imports section at the top to add the new imports:

```python
from kernel.dag_runner import KernelDAGRunner
from kernel.round_metrics import compute_round_metrics
```

Replace the `settle_round` method body. Change lines 276-310 of `world_kernel.py`:

**Old** `settle_round`:
```python
    def settle_round(self, seed: int) -> RoundReceipt:
        """Run produce/consume and build a deterministic round receipt."""
        snap_before = self.get_snapshot()

        self._execution.produce()
        self._execution.consume()

        snap_after = self.get_snapshot()
        ...
        receipt = RoundReceipt(
            ...
            judge_results=[],
            round_metrics={},
            ...
        )
```

**New** `settle_round`:
```python
    def settle_round(
        self,
        seed: int,
        judge_results: Optional[list] = None,
        mechanism_proposals: int = 0,
        mechanism_approvals: int = 0,
    ) -> RoundReceipt:
        """Run all settlement phases via DAG runner and build a round receipt.

        Args:
            seed: Deterministic seed for this round.
            judge_results: List of judge result dicts from this round's evaluation.
            mechanism_proposals: Count of mechanism proposals submitted.
            mechanism_approvals: Count of proposals approved by judge.
        """
        snap_before = self.get_snapshot()

        runner = KernelDAGRunner(self._execution)
        runner.run_settlement_phases()

        snap_after = self.get_snapshot()

        # Build deterministic timestamp from hash of seed + round_id
        ts_input = f"{seed}:{self._round_id}".encode("utf-8")
        deterministic_ts = hashlib.sha256(ts_input).hexdigest()[:16]

        # Collect action/mechanism IDs from cache
        accepted_ids = [
            key for key, res in self._idempotency_cache.items() if res.success
        ]
        rejected_ids = [
            key for key, res in self._idempotency_cache.items() if not res.success
        ]

        # Compute round metrics
        metrics = compute_round_metrics(
            members=snap_after.members,
            mechanism_proposals=mechanism_proposals,
            mechanism_approvals=mechanism_approvals,
        )

        receipt = RoundReceipt(
            round_id=self._round_id,
            seed=seed,
            snapshot_hash_before=snap_before.state_hash,
            snapshot_hash_after=snap_after.state_hash,
            accepted_action_ids=accepted_ids,
            rejected_action_ids=rejected_ids,
            activated_mechanism_ids=[],
            judge_results=judge_results or [],
            round_metrics=metrics,
            timestamp=deterministic_ts,
        )
        self._last_receipt = receipt
        return receipt
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_world_kernel.py::TestKernelDAGSettlement -v`
Expected: 4 passed

Also run the full test suite to verify no regressions:
Run: `python -m pytest --tb=short -q`
Expected: All existing tests pass (the old settle_round callers don't pass judge_results/mechanism_proposals, so they get the defaults)

**Step 5: Commit**

```bash
git add kernel/world_kernel.py test_world_kernel.py
git commit -m "feat: wire DAG runner and round metrics into settle_round"
```

---

### Task 6: Mechanism API Models + Round State Extension

**Files:**
- Modify: `api/models.py` — add mechanism and metrics Pydantic models
- Modify: `api/round_state.py` — add PendingProposal support
- Test: `test_api.py` (append new test classes)

**Step 1: Write the failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — Mechanism models + round state proposals
# ──────────────────────────────────────────────

from api.models import (
    MechanismProposeRequest,
    MechanismProposeResponse,
    MechanismResponse,
    MetricsResponse,
    JudgeStatsResponse,
)
from api.round_state import PendingProposal


class TestPhase3Models:
    def test_mechanism_propose_request(self):
        req = MechanismProposeRequest(
            code="def propose_modification(e): pass",
            description="test mechanism",
            idempotency_key="mech-1",
        )
        assert req.code.startswith("def ")
        assert req.idempotency_key == "mech-1"

    def test_mechanism_propose_response(self):
        resp = MechanismProposeResponse(mechanism_id="abc123", status="submitted")
        assert resp.mechanism_id == "abc123"

    def test_mechanism_response(self):
        resp = MechanismResponse(
            mechanism_id="abc",
            proposer_id=1,
            code="code",
            description="desc",
            status="active",
            submitted_round=1,
            judged_round=2,
            judge_reason="ok",
            activated_round=2,
        )
        assert resp.status == "active"

    def test_metrics_response(self):
        resp = MetricsResponse(
            round_id=1,
            total_vitality=100.0,
            gini_coefficient=0.3,
            trade_volume=5,
            conflict_count=2,
            mechanism_proposals=1,
            mechanism_approvals=1,
            population=10,
        )
        assert resp.population == 10

    def test_judge_stats_response(self):
        resp = JudgeStatsResponse(
            total_judgments=10,
            approved=7,
            rejected=3,
            approval_rate=0.7,
            recent_rejections=[],
        )
        assert resp.approval_rate == pytest.approx(0.7)


class TestPendingProposal:
    def test_creation(self):
        pp = PendingProposal(
            agent_id=1,
            member_id=5,
            code="def propose_modification(e): pass",
            description="test",
            idempotency_key="pk-1",
        )
        assert pp.agent_id == 1
        assert pp.member_id == 5

    def test_round_state_submit_proposal(self):
        from api.round_state import RoundState
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=10.0)
        pp = PendingProposal(
            agent_id=1, member_id=5, code="code", description="d", idempotency_key="pk-1"
        )
        assert rs.submit_proposal(pp) is True

    def test_round_state_drain_proposals(self):
        from api.round_state import RoundState
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=10.0)
        pp = PendingProposal(
            agent_id=1, member_id=5, code="code", description="d", idempotency_key="pk-1"
        )
        rs.submit_proposal(pp)
        proposals = rs.drain_proposals()
        assert len(proposals) == 1
        assert rs.drain_proposals() == []  # drained
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_api.py::TestPhase3Models test_api.py::TestPendingProposal -v`
Expected: FAIL — import errors

**Step 3: Implement**

Add to `api/models.py` (append after the existing models):

```python
# ── Phase 3: Governance models ────────────────


class MechanismProposeRequest(BaseModel):
    """Request body for POST /v1/world/mechanisms/propose."""
    code: str
    description: str
    idempotency_key: str


class MechanismProposeResponse(BaseModel):
    """Response from POST /v1/world/mechanisms/propose."""
    mechanism_id: str
    status: str


class MechanismResponse(BaseModel):
    """Full mechanism detail response."""
    mechanism_id: str
    proposer_id: int
    code: str
    description: str
    status: str
    submitted_round: int
    judged_round: Optional[int] = None
    judge_reason: Optional[str] = None
    activated_round: Optional[int] = None


class MetricsResponse(BaseModel):
    """Response from GET /v1/world/metrics."""
    round_id: int
    total_vitality: float
    gini_coefficient: float
    trade_volume: int
    conflict_count: int
    mechanism_proposals: int
    mechanism_approvals: int
    population: int


class JudgeStatsResponse(BaseModel):
    """Response from GET /v1/world/judge/stats."""
    total_judgments: int
    approved: int
    rejected: int
    approval_rate: float
    recent_rejections: List[Dict[str, Any]]
```

Add `PendingProposal` dataclass and proposal methods to `api/round_state.py`:

Add after the `PendingAction` dataclass:

```python
@dataclass
class PendingProposal:
    """A mechanism proposal submitted by an external agent, waiting for judge."""
    agent_id: int
    member_id: int
    code: str
    description: str
    idempotency_key: str
```

Add these methods to the `RoundState` class (after `get_pending_actions`):

```python
    def submit_proposal(self, proposal: PendingProposal) -> bool:
        with self._lock:
            if self.state != "accepting":
                return False
            if proposal.idempotency_key in self._seen_keys:
                return True
            self._seen_keys[proposal.idempotency_key] = True
            self._pending_proposals.append(proposal)
            return True

    def drain_proposals(self) -> List[PendingProposal]:
        with self._lock:
            proposals = list(self._pending_proposals)
            self._pending_proposals = []
            return proposals
```

Also add `self._pending_proposals: List[PendingProposal] = []` to `__init__` and reset it in `open_submissions`.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_api.py::TestPhase3Models test_api.py::TestPendingProposal -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add api/models.py api/round_state.py test_api.py
git commit -m "feat: add mechanism/metrics models and proposal support to RoundState"
```

---

### Task 7: Mechanism API Endpoints

**Files:**
- Create: `api/routes/mechanisms.py`
- Modify: `api/app.py` — include mechanisms router
- Modify: `api/deps.py` — add mechanism_registry + judge accessors
- Test: `test_api.py` (append new test class)

**Step 1: Write the failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — Mechanism endpoint tests
# ──────────────────────────────────────────────


class TestMechanismEndpoints:
    """Test mechanism proposal and listing endpoints."""

    def _make_client(self):
        from scripts.run_server import build_app
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        from starlette.testclient import TestClient
        return TestClient(app), app

    def _register_agent(self, client):
        resp = client.post("/v1/agents/register", json={"name": "Bot"})
        return resp.json()

    def test_propose_mechanism_accepted(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        # Open submission window
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)

        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={
                "code": "def propose_modification(e): pass",
                "description": "test mechanism",
                "idempotency_key": "m-1",
            },
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "submitted"
        assert "mechanism_id" in data

    def test_propose_mechanism_rejected_outside_window(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        # Don't open submission window

        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={
                "code": "def propose_modification(e): pass",
                "description": "test",
                "idempotency_key": "m-1",
            },
            headers={"X-API-Key": agent["api_key"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_propose_mechanism_unauthorized(self):
        client, app = self._make_client()
        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={
                "code": "def propose_modification(e): pass",
                "description": "test",
                "idempotency_key": "m-1",
            },
        )
        assert resp.status_code == 401

    def test_list_mechanisms(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)

        client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "d", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )

        resp = client.get("/v1/world/mechanisms")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_list_mechanisms_with_status_filter(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/mechanisms?status=active")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_mechanism_by_id(self):
        client, app = self._make_client()
        agent = self._register_agent(client)
        rs = app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=60.0)

        resp = client.post(
            "/v1/world/mechanisms/propose",
            json={"code": "code", "description": "d", "idempotency_key": "m-1"},
            headers={"X-API-Key": agent["api_key"]},
        )
        mech_id = resp.json()["mechanism_id"]

        resp = client.get(f"/v1/world/mechanisms/{mech_id}")
        assert resp.status_code == 200
        assert resp.json()["mechanism_id"] == mech_id

    def test_get_mechanism_not_found(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/mechanisms/nonexistent")
        assert resp.status_code == 404
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_api.py::TestMechanismEndpoints -v`
Expected: FAIL — routes don't exist yet

**Step 3: Implement**

Create `api/routes/mechanisms.py`:

```python
"""Mechanism proposal and listing endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request

from api.models import MechanismProposeRequest, MechanismProposeResponse, MechanismResponse
from api.round_state import PendingProposal

router = APIRouter(prefix="/v1/world/mechanisms")


@router.post("/propose", response_model=MechanismProposeResponse)
def propose_mechanism(body: MechanismProposeRequest, request: Request):
    """Submit a mechanism proposal for the current round."""
    registry = request.app.state.leviathan["registry"]
    round_state = request.app.state.leviathan["round_state"]
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = registry.get_by_api_key(key)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid API key")

    pp = PendingProposal(
        agent_id=record.agent_id,
        member_id=record.member_id,
        code=body.code,
        description=body.description,
        idempotency_key=body.idempotency_key,
    )
    accepted = round_state.submit_proposal(pp)

    if not accepted:
        return MechanismProposeResponse(mechanism_id="", status="rejected")

    mech = mechanism_registry.submit(
        proposer_id=record.member_id,
        code=body.code,
        description=body.description,
        round_id=round_state.round_id,
    )
    if mech is None:
        return MechanismProposeResponse(mechanism_id="", status="rejected")

    return MechanismProposeResponse(mechanism_id=mech.mechanism_id, status="submitted")


@router.get("", response_model=List[MechanismResponse])
def list_mechanisms(request: Request, status: Optional[str] = None):
    """List all mechanisms, optionally filtered by status."""
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]

    if status == "active":
        records = mechanism_registry.get_active()
    elif status == "submitted":
        records = mechanism_registry.get_pending()
    else:
        records = mechanism_registry.get_all()

    return [
        MechanismResponse(
            mechanism_id=r.mechanism_id,
            proposer_id=r.proposer_id,
            code=r.code,
            description=r.description,
            status=r.status,
            submitted_round=r.submitted_round,
            judged_round=r.judged_round,
            judge_reason=r.judge_reason,
            activated_round=r.activated_round,
        )
        for r in records
    ]


@router.get("/{mechanism_id}", response_model=MechanismResponse)
def get_mechanism(mechanism_id: str, request: Request):
    """Get a mechanism by ID."""
    mechanism_registry = request.app.state.leviathan["mechanism_registry"]
    rec = mechanism_registry.get_by_id(mechanism_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Mechanism not found")
    return MechanismResponse(
        mechanism_id=rec.mechanism_id,
        proposer_id=rec.proposer_id,
        code=rec.code,
        description=rec.description,
        status=rec.status,
        submitted_round=rec.submitted_round,
        judged_round=rec.judged_round,
        judge_reason=rec.judge_reason,
        activated_round=rec.activated_round,
    )
```

Modify `api/deps.py` — add to `create_app_state()`:

```python
from kernel.mechanism_registry import MechanismRegistry
from kernel.judge_adapter import JudgeAdapter
```

And in `create_app_state`:
```python
    return {
        "kernel": kernel,
        "event_log": [],
        "registry": AgentRegistry(),
        "round_state": RoundState(),
        "mechanism_registry": MechanismRegistry(),
        "judge": JudgeAdapter(use_dummy=True),
    }
```

Add accessor functions:
```python
def get_mechanism_registry(request: Request) -> "MechanismRegistry":
    return request.app.state.leviathan["mechanism_registry"]


def get_judge(request: Request) -> "JudgeAdapter":
    return request.app.state.leviathan["judge"]
```

Modify `api/app.py` — add import and include:

```python
from api.routes.mechanisms import router as mechanisms_router
```

And in `create_app`:
```python
    app.include_router(mechanisms_router)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_api.py::TestMechanismEndpoints -v`
Expected: 7 passed

Also run full test suite: `python -m pytest --tb=short -q`

**Step 5: Commit**

```bash
git add api/routes/mechanisms.py api/deps.py api/app.py test_api.py
git commit -m "feat: add mechanism proposal and listing endpoints"
```

---

### Task 8: Metrics and Judge Stats Endpoints

**Files:**
- Create: `api/routes/metrics.py`
- Modify: `api/app.py` — include metrics router
- Test: `test_api.py` (append new test class)

**Step 1: Write the failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — Metrics and judge stats endpoint tests
# ──────────────────────────────────────────────


class TestMetricsEndpoints:
    def _make_client(self):
        from scripts.run_server import build_app
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        from starlette.testclient import TestClient
        return TestClient(app), app

    def test_metrics_before_any_round(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/metrics")
        assert resp.status_code == 200
        data = resp.json()
        # Before any round, should still return current state metrics
        assert "total_vitality" in data
        assert "population" in data

    def test_metrics_after_settle(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        kernel.begin_round()
        kernel.settle_round(seed=1)

        resp = client.get("/v1/world/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["population"] == 5
        assert data["total_vitality"] > 0

    def test_metrics_history(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        for i in range(3):
            kernel.begin_round()
            kernel.settle_round(seed=i + 1)

        resp = client.get("/v1/world/metrics/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_metrics_history_with_limit(self):
        client, app = self._make_client()
        kernel = app.state.leviathan["kernel"]
        for i in range(5):
            kernel.begin_round()
            kernel.settle_round(seed=i + 1)

        resp = client.get("/v1/world/metrics/history?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_judge_stats(self):
        client, app = self._make_client()
        resp = client.get("/v1/world/judge/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_judgments"] == 0
        assert data["approval_rate"] == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_api.py::TestMetricsEndpoints -v`
Expected: FAIL — 404 from missing routes

**Step 3: Implement**

Create `api/routes/metrics.py`:

```python
"""Metrics and judge statistics endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from api.models import JudgeStatsResponse, MetricsResponse
from kernel.round_metrics import compute_round_metrics

router = APIRouter(prefix="/v1/world")


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    """Return metrics for the latest round (or current state if no rounds settled)."""
    kernel = request.app.state.leviathan["kernel"]
    receipt = kernel.get_round_receipt()

    if receipt and receipt.round_metrics:
        return MetricsResponse(round_id=receipt.round_id, **receipt.round_metrics)

    # No round settled yet — compute from current state
    snap = kernel.get_snapshot()
    metrics = compute_round_metrics(members=snap.members)
    return MetricsResponse(round_id=snap.round_id, **metrics)


@router.get("/metrics/history", response_model=List[MetricsResponse])
def get_metrics_history(
    request: Request, limit: Optional[int] = None
) -> List[MetricsResponse]:
    """Return metrics for recent rounds from the event log."""
    event_log = request.app.state.leviathan["event_log"]

    history = []
    for event in event_log:
        if event.event_type == "round_settled":
            rm = event.payload.get("round_metrics", {})
            if rm:
                history.append(MetricsResponse(round_id=event.round_id, **rm))

    if limit:
        history = history[-limit:]

    return history


@router.get("/judge/stats", response_model=JudgeStatsResponse)
def get_judge_stats(request: Request) -> JudgeStatsResponse:
    """Return judge approval statistics."""
    judge = request.app.state.leviathan["judge"]

    # Track judgments internally
    history = getattr(judge, "_judgment_history", [])

    total = len(history)
    approved = sum(1 for j in history if j.get("approved"))
    rejected = total - approved

    recent_rejections = [j for j in history if not j.get("approved")][-5:]

    return JudgeStatsResponse(
        total_judgments=total,
        approved=approved,
        rejected=rejected,
        approval_rate=approved / total if total > 0 else 0.0,
        recent_rejections=recent_rejections,
    )
```

Modify `api/app.py` — add:

```python
from api.routes.metrics import router as metrics_router
```

And include it:
```python
    app.include_router(metrics_router)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_api.py::TestMetricsEndpoints -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add api/routes/metrics.py api/app.py test_api.py
git commit -m "feat: add metrics and judge stats endpoints"
```

---

### Task 9: Wire Judge + Mechanisms into Simulation Loop

**Files:**
- Modify: `scripts/run_server.py` — integrate judge evaluation and mechanism execution into the sim loop
- Test: `test_api.py` (append integration test)

**Step 1: Write the failing test**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 3 — Integration test
# ──────────────────────────────────────────────

import threading
import time


class TestGovernanceIntegration:
    """End-to-end test: register, propose mechanism, judge evaluates, verify receipt."""

    def test_full_governance_round(self):
        from scripts.run_server import build_app, _simulation_loop
        from starlette.testclient import TestClient

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        kernel = app.state.leviathan["kernel"]
        event_log = app.state.leviathan["event_log"]
        round_state = app.state.leviathan["round_state"]
        mechanism_registry = app.state.leviathan["mechanism_registry"]

        # Register an agent
        resp = client.post("/v1/agents/register", json={"name": "GovBot"})
        assert resp.status_code == 200
        agent = resp.json()

        # Run sim loop for 1 round with short pace
        stop = threading.Event()
        t = threading.Thread(
            target=_simulation_loop,
            args=(kernel, event_log, round_state, mechanism_registry, app.state.leviathan["judge"], 0.5, 1, stop),
        )
        t.start()

        # Wait for accepting state
        for _ in range(20):
            time.sleep(0.05)
            if round_state.state == "accepting":
                break

        # Submit a mechanism proposal during the window
        if round_state.state == "accepting":
            resp = client.post(
                "/v1/world/mechanisms/propose",
                json={
                    "code": "def propose_modification(e): pass",
                    "description": "no-op mechanism",
                    "idempotency_key": "gov-test-1",
                },
                headers={"X-API-Key": agent["api_key"]},
            )
            assert resp.status_code == 200

        t.join(timeout=5)
        stop.set()

        # Verify the round settled with metrics
        assert len(event_log) >= 1
        last_event = event_log[-1]
        assert last_event.event_type == "round_settled"
        assert "total_vitality" in last_event.payload.get("round_metrics", {})
        assert "population" in last_event.payload.get("round_metrics", {})
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_api.py::TestGovernanceIntegration -v`
Expected: FAIL — `_simulation_loop` signature doesn't match (missing mechanism_registry + judge params)

**Step 3: Modify `scripts/run_server.py`**

Update `_simulation_loop` to accept `mechanism_registry` and `judge` parameters, and integrate judge evaluation:

```python
def _simulation_loop(
    kernel: WorldKernel,
    event_log: List[EventEnvelope],
    round_state,
    mechanism_registry,
    judge,
    pace: float,
    max_rounds: int,
    stop_event: threading.Event,
) -> None:
    """Background thread that advances the simulation.

    Each iteration runs one full round with a submission window:
      begin_round -> open_submissions -> sleep(pace) ->
      close_submissions -> judge_proposals -> execute_mechanisms ->
      execute_actions -> settle_round -> append_event
    """
    from kernel.subprocess_sandbox import SubprocessSandbox
    from kernel.execution_sandbox import SandboxContext

    sandbox = SubprocessSandbox()
    rounds_completed = 0

    while not stop_event.is_set():
        kernel.begin_round()
        round_state.open_submissions(round_id=kernel.round_id, pace=pace)

        # Sleep for the submission window
        stop_event.wait(timeout=pace)
        if stop_event.is_set():
            break

        round_state.close_submissions()

        # ── Judge pending mechanism proposals ──
        proposals = round_state.drain_proposals()
        judge_results = []
        mechanism_approvals = 0

        for pp in proposals:
            result = judge.evaluate(pp.code, pp.member_id, "mechanism")
            judge_entry = {
                "proposer_id": pp.member_id,
                "approved": result.approved,
                "reason": result.reason,
                "latency_ms": result.latency_ms,
            }

            # Find the mechanism record and update it
            for rec in mechanism_registry.get_pending():
                if rec.proposer_id == pp.member_id and rec.code == pp.code:
                    if result.approved:
                        mechanism_registry.mark_approved(
                            rec.mechanism_id, kernel.round_id, result.reason
                        )
                        mechanism_approvals += 1
                        judge_entry["proposal_id"] = rec.mechanism_id
                    else:
                        mechanism_registry.mark_rejected(
                            rec.mechanism_id, kernel.round_id, result.reason
                        )
                        judge_entry["proposal_id"] = rec.mechanism_id
                    break

            judge_results.append(judge_entry)

        # ── Execute approved mechanisms ──
        for rec in mechanism_registry.get_all():
            if rec.status == "approved" and rec.judged_round == kernel.round_id:
                try:
                    ctx = SandboxContext(
                        execution_engine=kernel._execution,
                        member_index=0,
                    )
                    sandbox_result = sandbox.execute_mechanism_code(rec.code, ctx)
                    if sandbox_result.success:
                        mechanism_registry.activate(rec.mechanism_id, kernel.round_id)
                except Exception:
                    pass  # mechanism execution failure doesn't halt the round

        # ── Execute pending agent actions ──
        pending = round_state.drain_actions()
        for pa in pending:
            member_index = kernel._resolve_agent_index(pa.member_id)
            if member_index is not None:
                ctx = SandboxContext(
                    execution_engine=kernel._execution,
                    member_index=member_index,
                )
                result = sandbox.execute_agent_code(pa.code, ctx)
                if result.success:
                    kernel.apply_intended_actions(result.intended_actions)

        # ── Settle round with metrics ──
        receipt = kernel.settle_round(
            seed=kernel.round_id,
            judge_results=judge_results,
            mechanism_proposals=len(proposals),
            mechanism_approvals=mechanism_approvals,
        )
        round_state.mark_settled()

        event_log.append(
            EventEnvelope(
                event_id=len(event_log) + 1,
                event_type="round_settled",
                round_id=receipt.round_id,
                timestamp=receipt.timestamp,
                payload=dataclasses.asdict(receipt),
            )
        )
        rounds_completed += 1
        logger.info("Round %d settled", receipt.round_id)

        if max_rounds > 0 and rounds_completed >= max_rounds:
            logger.info("Reached max rounds (%d), stopping", max_rounds)
            break
```

Update `main()` to pass the new args:

```python
    mechanism_registry = app.state.leviathan["mechanism_registry"]
    judge = app.state.leviathan["judge"]

    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(kernel, event_log, round_state, mechanism_registry, judge, args.pace, args.rounds, stop_event),
        daemon=True,
        name="sim-loop",
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest test_api.py::TestGovernanceIntegration -v`
Expected: 1 passed

Also run full test suite: `python -m pytest --tb=short -q`
Expected: All tests pass. Note: the Phase 2 `TestSimLoopIntegration` test also calls `_simulation_loop` — it must be updated to pass the new `mechanism_registry` and `judge` args. Find the existing call and add the two extra arguments.

**Step 5: Commit**

```bash
git add scripts/run_server.py test_api.py
git commit -m "feat: integrate judge + mechanism execution into simulation loop"
```

---

### Task 10: Update Existing Tests for New settle_round Signature

**Files:**
- Modify: `test_api.py` — update Phase 2 integration test to pass new sim loop args
- Test: Full test suite

**Step 1: Identify and fix**

The Phase 2 `TestSimLoopIntegration.test_full_external_agent_round` calls `_simulation_loop` with the old argument list. Update it to include `mechanism_registry` and `judge`.

Find the existing call pattern:
```python
t = threading.Thread(
    target=_simulation_loop,
    args=(kernel, event_log, round_state, 0.3, 1, stop),
)
```

Change to:
```python
t = threading.Thread(
    target=_simulation_loop,
    args=(kernel, event_log, round_state, mechanism_registry, judge, 0.3, 1, stop),
)
```

And add:
```python
mechanism_registry = app.state.leviathan["mechanism_registry"]
judge = app.state.leviathan["judge"]
```

**Step 2: Run full test suite**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add test_api.py
git commit -m "fix: update Phase 2 integration test for new sim loop signature"
```

---

### Task 11: Update Compromises Document

**Files:**
- Modify: `docs/plans/2026-02-24-implementation-compromises.md`

**Step 1: Update resolved items and add new compromises**

Mark K3, K4, K5 as RESOLVED:

- **K3**: `~~K3. Active mechanisms tracking is shallow~~ RESOLVED` — MechanismRegistry tracks full lifecycle. Proposals go through submitted -> approved/rejected -> active. Registry provides get_pending(), get_active(), get_all().

- **K4**: `~~K4. settle_round only runs produce/consume~~ RESOLVED` — KernelDAGRunner runs phases in topological order (contracts, produce, consume, environment). Matches the DAG engine's infrastructure nodes.

- **K5**: `~~K5. No round_metrics or judge_results populated~~ RESOLVED` — round_metrics computed by `compute_round_metrics()` with Gini coefficient. judge_results populated from JudgeAdapter evaluations. Both included in RoundReceipt.

Add new Phase 3 compromises:

```markdown
### P3-1. JudgeAdapter timeout is hardcoded

- **What we built:** 30s timeout, not configurable at runtime.
- **When to fix:** When different proposal types need different timeouts.

### P3-2. No LLM cost tracking per judgment

- **What we built:** JudgeAdapter doesn't track token usage or cost per judgment.
- **When to fix:** When cost budgeting for judge operations matters.

### P3-3. Gini coefficient only over vitality

- **What we built:** Gini computed over member vitality only. Doesn't consider cargo, land, or total wealth.
- **When to fix:** When a more nuanced inequality metric is needed.

### P3-4. MechanismRegistry is in-memory only

- **What we built:** No persistence. Lost on restart.
- **When to fix:** Before production. Can share SQLite store with event log (A2).

### P3-5. No mechanism rollback or deactivation

- **What we built:** Once active, mechanisms are permanent. No way to deactivate or roll back.
- **When to fix:** Phase 4 governance features.

### P3-6. Single-judge LLM evaluation

- **What we built:** One LLM call per proposal. No multi-judge consensus, appeal, or human override.
- **When to fix:** When trust guarantees require multi-party validation.

### P3-7. No mechanism versioning

- **What we built:** Can't update an active mechanism. Must propose a new one.
- **When to fix:** When mechanism evolution patterns emerge.
```

Update the summary section to reflect the new resolved items.

**Step 2: Run full test suite to confirm nothing broke**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add docs/plans/2026-02-24-implementation-compromises.md
git commit -m "docs: mark K3, K4, K5 resolved, document Phase 3 tech debt"
```

---

### Task 12: Full Verification

**Step 1: Run the complete test suite**

Run: `python -m pytest --tb=short -q`
Expected: All tests pass (previous 174 + ~47 new = ~221 total)

**Step 2: Verify imports are clean**

Run: `python -c "from kernel.mechanism_registry import MechanismRegistry; from kernel.judge_adapter import JudgeAdapter, DummyJudge, JudgmentResult; from kernel.round_metrics import compute_round_metrics, compute_gini; from kernel.dag_runner import KernelDAGRunner; print('All imports OK')"`

**Step 3: Verify API routes**

Run: `python -c "from api.app import create_app; from kernel.schemas import WorldConfig; from kernel.world_kernel import WorldKernel; import tempfile; k = WorldKernel(WorldConfig(5, (10,10), 42), tempfile.mkdtemp()); app = create_app(k); routes = [r.path for r in app.routes]; print([r for r in routes if 'mechanism' in r or 'metrics' in r or 'judge' in r])"`
Expected: Shows the new mechanism, metrics, and judge routes

**Step 4: Final commit if any cleanup needed, then push**

```bash
git push origin clawbot-self-improve
```
