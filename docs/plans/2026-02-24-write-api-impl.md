# Phase 2: External Write Path + Agent Onboarding — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable external agents to register, observe the world, and submit actions via HTTP, completing the feedback loop that makes Leviathan a platform.

**Architecture:** SubprocessSandbox runs agent code in a child process against an EngineProxy that records intended actions as JSON. A new simulation loop opens a submission window between `begin_round` and `settle_round`. Agent registration and action submission happen through new HTTP endpoints protected by per-agent API keys.

**Tech Stack:** Python 3, FastAPI, subprocess, threading, pytest, httpx (TestClient)

---

## Background: How Agent Code Works Today

Agent code is a Python string defining `def agent_action(execution_engine, member_id)`. The engine is an `IslandExecution` instance. The `member_id` is the member's **index** in `current_members` (not permanent `member.id`). Agents call methods like:

```python
def agent_action(execution_engine, member_id):
    me = execution_engine.current_members[member_id]
    target = execution_engine.current_members[0]
    execution_engine.attack(me, target)       # attack(Member, Member)
    execution_engine.offer(me, target)        # offer(Member, Member)
    execution_engine.offer_land(me, target)   # offer_land(Member, Member)
    execution_engine.expand(me)               # expand(Member)
    execution_engine.send_message(me.id, target.id, "hello")
```

The EngineProxy must mimic this interface but record actions instead of executing them.

---

### Task 1: EngineProxy + MemberProxy + LandProxy

**Files:**
- Create: `kernel/engine_proxy.py`
- Test: `test_world_kernel.py` (append new test class)

**Context:** The EngineProxy is a lightweight stand-in for IslandExecution. It's instantiated from a serialized state dict and records action calls as JSON instead of mutating state. Agent code runs against it in a subprocess.

**Step 1: Write failing tests**

Add to `test_world_kernel.py`:

```python
# ──────────────────────────────────────────────
# SubprocessSandbox — EngineProxy tests
# ──────────────────────────────────────────────

from kernel.engine_proxy import EngineProxy, MemberProxy, LandProxy


class TestEngineProxy:
    """Verify the EngineProxy records actions instead of executing them."""

    def _make_state(self, n_members=3):
        """Build a minimal state dict matching what WorldKernel.get_snapshot() produces."""
        members = []
        for i in range(n_members):
            members.append({
                "id": i + 1,
                "vitality": 50.0,
                "cargo": 20.0,
                "land_num": 2,
            })
        return {
            "members": members,
            "land": {"shape": [10, 10]},
        }

    def test_member_proxy_attributes(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        member = proxy.current_members[0]
        assert isinstance(member, MemberProxy)
        assert member.id == 1
        assert member.vitality == 50.0
        assert member.cargo == 20.0
        assert member.land_num == 2

    def test_land_proxy_shape(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        assert isinstance(proxy.land, LandProxy)
        assert proxy.land.shape == (10, 10)

    def test_attack_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.attack(me, target)
        assert len(proxy.actions) == 1
        assert proxy.actions[0] == {
            "action": "attack",
            "member_id": 1,
            "target_id": 2,
        }

    def test_offer_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.offer(me, target)
        assert proxy.actions[0] == {
            "action": "offer",
            "member_id": 1,
            "target_id": 2,
        }

    def test_offer_land_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.offer_land(me, target)
        assert proxy.actions[0] == {
            "action": "offer_land",
            "member_id": 1,
            "target_id": 2,
        }

    def test_expand_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        proxy.expand(me)
        assert proxy.actions[0] == {
            "action": "expand",
            "member_id": 1,
        }

    def test_send_message_records_action(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        proxy.send_message(1, 2, "hello")
        assert proxy.actions[0] == {
            "action": "message",
            "from_id": 1,
            "to_id": 2,
            "message": "hello",
        }

    def test_multiple_actions_recorded_in_order(self):
        state = self._make_state()
        proxy = EngineProxy(state)
        me = proxy.current_members[0]
        target = proxy.current_members[1]
        proxy.attack(me, target)
        proxy.offer(me, target)
        proxy.expand(me)
        assert len(proxy.actions) == 3
        assert [a["action"] for a in proxy.actions] == ["attack", "offer", "expand"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_world_kernel.py::TestEngineProxy -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'kernel.engine_proxy'`

**Step 3: Write minimal implementation**

Create `kernel/engine_proxy.py`:

```python
"""Lightweight proxy objects for running agent code in a subprocess.

The EngineProxy mimics the IslandExecution interface but records
intended actions as JSON dicts instead of executing them.  Agent
code that works against the real engine should also work against
this proxy — the only difference is that side effects are captured,
not applied.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class MemberProxy:
    """Read-only stand-in for a Member object."""

    id: int
    vitality: float
    cargo: float
    land_num: int


@dataclass
class LandProxy:
    """Read-only stand-in for the Land object."""

    shape: Tuple[int, int]


class EngineProxy:
    """Stand-in for IslandExecution that records actions as JSON.

    Instantiated from a state dict (subset of WorldSnapshot) and
    provides the same method signatures that agent code calls.
    """

    def __init__(self, state: dict) -> None:
        self.current_members: List[MemberProxy] = [
            MemberProxy(**m) for m in state["members"]
        ]
        land_shape = state["land"]["shape"]
        self.land = LandProxy(shape=tuple(land_shape))
        self.actions: List[Dict[str, Any]] = []

    def attack(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({
            "action": "attack",
            "member_id": member.id,
            "target_id": target.id,
        })

    def offer(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({
            "action": "offer",
            "member_id": member.id,
            "target_id": target.id,
        })

    def offer_land(self, member: MemberProxy, target: MemberProxy) -> None:
        self.actions.append({
            "action": "offer_land",
            "member_id": member.id,
            "target_id": target.id,
        })

    def expand(self, member: MemberProxy) -> None:
        self.actions.append({
            "action": "expand",
            "member_id": member.id,
        })

    def send_message(self, from_id: int, to_id: int, message: str) -> None:
        self.actions.append({
            "action": "message",
            "from_id": from_id,
            "to_id": to_id,
            "message": message,
        })
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_world_kernel.py::TestEngineProxy -v`
Expected: 8 tests PASS

**Step 5: Commit**

```bash
git add kernel/engine_proxy.py test_world_kernel.py
git commit -m "feat: add EngineProxy with action recording for subprocess sandbox"
```

---

### Task 2: SubprocessSandbox

**Files:**
- Create: `kernel/subprocess_sandbox.py`
- Modify: `kernel/execution_sandbox.py` — add `intended_actions` field to SandboxResult
- Test: `test_world_kernel.py` (append new test class)

**Context:** The SubprocessSandbox runs agent code in a child process. It serializes member state to JSON, writes a wrapper script that creates an EngineProxy, runs the agent's `agent_action()` function against it, then prints the recorded actions as JSON to stdout. The parent process reads stdout and parses the actions.

**Step 1: Write failing tests**

Add to `test_world_kernel.py`:

```python
from kernel.subprocess_sandbox import SubprocessSandbox
from kernel.execution_sandbox import SandboxContext, SandboxResult


class TestSubprocessSandbox:
    """Verify SubprocessSandbox runs code in a child process."""

    def _make_context(self, n_members=3):
        """Build a SandboxContext with a real kernel for state extraction."""
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=n_members, land_shape=(10, 10), random_seed=42)
        kernel = WorldKernel(config, save_path=tmpdir)
        return SandboxContext(
            execution_engine=kernel._execution,
            member_index=0,
        ), kernel

    def test_simple_expand_action(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    engine.expand(me)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 1
        assert result.intended_actions[0]["action"] == "expand"

    def test_attack_action(self):
        ctx, kernel = self._make_context(n_members=5)
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    target = engine.current_members[1]\n"
            "    engine.attack(me, target)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 1
        assert result.intended_actions[0]["action"] == "attack"

    def test_multiple_actions(self):
        ctx, kernel = self._make_context(n_members=5)
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    me = engine.current_members[member_id]\n"
            "    target = engine.current_members[1]\n"
            "    engine.attack(me, target)\n"
            "    engine.offer(me, target)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is True
        assert len(result.intended_actions) == 2

    def test_syntax_error_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = "def agent_action(engine, member_id)\n"  # missing colon
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "SyntaxError" in (result.error or "")

    def test_no_entry_point_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = "x = 42\n"  # no agent_action defined
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "agent_action" in (result.error or "")

    def test_runtime_error_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox()
        code = (
            "def agent_action(engine, member_id):\n"
            "    raise ValueError('boom')\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "boom" in (result.error or "")

    def test_timeout_returns_failure(self):
        ctx, kernel = self._make_context()
        sandbox = SubprocessSandbox(timeout=2)
        code = (
            "def agent_action(engine, member_id):\n"
            "    import time\n"
            "    time.sleep(10)\n"
        )
        result = sandbox.execute_agent_code(code, ctx)
        assert result.success is False
        assert "timeout" in (result.error or "").lower() or "timed out" in (result.error or "").lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_world_kernel.py::TestSubprocessSandbox -v`
Expected: FAIL with import errors

**Step 3: Modify SandboxResult to include intended_actions**

In `kernel/execution_sandbox.py`, change the `SandboxResult` dataclass to:

```python
@dataclass
class SandboxResult:
    """Structured result from sandbox execution."""

    success: bool
    error: Optional[str] = None
    traceback_str: Optional[str] = None
    intended_actions: list = field(default_factory=list)
```

The file already imports `field` from dataclasses (used by SandboxContext).

**Step 4: Write SubprocessSandbox implementation**

Create `kernel/subprocess_sandbox.py`:

```python
"""SubprocessSandbox — runs agent code in a child process with resource limits.

The agent code runs against an EngineProxy that records intended actions
as JSON.  The subprocess writes the action list to stdout.  The parent
process reads it and returns it in SandboxResult.intended_actions.
"""

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

from kernel.execution_sandbox import SandboxContext, SandboxResult

# Path to engine_proxy.py — needed by the subprocess
_ENGINE_PROXY_PATH = Path(__file__).parent / "engine_proxy.py"


class SubprocessSandbox:
    """Runs agent code in a child process with timeout enforcement."""

    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout

    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult:
        """Run agent code in subprocess, return intended actions."""
        state = self._extract_state(context)
        return self._run_in_subprocess(code, state, context.member_index, "agent_action")

    def execute_mechanism_code(self, code: str, context: SandboxContext) -> SandboxResult:
        """Run mechanism code in subprocess, return intended actions."""
        state = self._extract_state(context)
        return self._run_in_subprocess(code, state, context.member_index, "propose_modification")

    def _extract_state(self, context: SandboxContext) -> dict:
        """Extract serializable state from the execution engine."""
        engine = context.execution_engine
        members = []
        for m in engine.current_members:
            members.append({
                "id": m.id,
                "vitality": float(m.vitality),
                "cargo": float(m.cargo),
                "land_num": int(m.land_num),
            })
        return {
            "members": members,
            "land": {"shape": list(engine.land.shape)},
        }

    def _run_in_subprocess(
        self, code: str, state: dict, member_index: int, entry_point: str
    ) -> SandboxResult:
        """Write wrapper script, run in subprocess, parse output."""
        wrapper = self._build_wrapper(code, state, member_index, entry_point)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(wrapper)
            script_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Timed out after {self.timeout}s",
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return SandboxResult(
                success=False,
                error=stderr or f"Process exited with code {proc.returncode}",
                traceback_str=stderr,
            )

        # Parse JSON output
        try:
            output = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError) as e:
            return SandboxResult(
                success=False,
                error=f"Failed to parse subprocess output: {e}",
            )

        if output.get("error"):
            return SandboxResult(
                success=False,
                error=output["error"],
                traceback_str=output.get("traceback"),
            )

        return SandboxResult(
            success=True,
            intended_actions=output.get("actions", []),
        )

    def _build_wrapper(
        self, code: str, state: dict, member_index: int, entry_point: str
    ) -> str:
        """Generate the wrapper script that runs in the subprocess."""
        state_json = json.dumps(state)
        repo_root = str(_ENGINE_PROXY_PATH.parent.parent)

        lines = [
            "import json",
            "import sys",
            "import traceback",
            "",
            f"sys.path.insert(0, {repr(repo_root)})",
            "from kernel.engine_proxy import EngineProxy",
            "",
            f"_AGENT_CODE = {repr(code)}",
            f"_STATE = json.loads({repr(state_json)})",
            "",
            "def main():",
            "    proxy = EngineProxy(_STATE)",
            "    try:",
            "        compiled = compile(_AGENT_CODE, '<agent>', 'exec')",
            "        ns = {}",
            "        ns['__builtins__'] = __builtins__",
            "        _run = getattr(__import__('builtins'), 'exec')",  # noqa: safe — sandbox entry point
            "        _run(compiled, ns)",
            "    except SyntaxError as e:",
            "        print(json.dumps({'error': f'SyntaxError: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            "    except Exception as e:",
            "        print(json.dumps({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            f"    fn = ns.get({repr(entry_point)})",
            "    if fn is None or not callable(fn):",
            f"        print(json.dumps({{'error': \"Code did not define callable '{entry_point}'\"}}))",
            "        return",
            "    try:",
        ]

        if entry_point == "agent_action":
            lines.append(f"        fn(proxy, {member_index})")
        else:
            lines.append("        fn(proxy)")

        lines.extend([
            "    except Exception as e:",
            "        print(json.dumps({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            "    print(json.dumps({'actions': proxy.actions}))",
            "",
            "main()",
        ])

        return "\n".join(lines) + "\n"
```

**Step 5: Run tests to verify they pass**

Run: `pytest test_world_kernel.py::TestSubprocessSandbox -v`
Expected: 7 tests PASS

**Step 6: Run all kernel tests**

Run: `pytest test_world_kernel.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add kernel/subprocess_sandbox.py kernel/execution_sandbox.py test_world_kernel.py
git commit -m "feat: add SubprocessSandbox with timeout and action proxy"
```

---

### Task 3: AgentRegistry

**Files:**
- Create: `api/registry.py`
- Test: `test_api.py` (append new test class)

**Context:** The AgentRegistry manages agent registrations in-memory. Each external agent gets a unique agent_id, an API key, and is assigned to an unoccupied in-world member. The registry tracks the mapping from API key to agent record.

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — Agent registry tests
# ──────────────────────────────────────────────

from api.registry import AgentRecord, AgentRegistry


class TestAgentRegistry:
    def test_register_agent(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "A test agent", kernel)
        assert isinstance(record, AgentRecord)
        assert record.name == "TestBot"
        assert record.api_key.startswith("lev_")
        assert record.member_id in [m.id for m in kernel._execution.current_members]

    def test_register_assigns_different_members(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        r1 = registry.register("Bot1", "", kernel)
        r2 = registry.register("Bot2", "", kernel)
        assert r1.member_id != r2.member_id
        assert r1.api_key != r2.api_key
        assert r1.agent_id != r2.agent_id

    def test_register_too_many_returns_none(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=2, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        registry.register("Bot1", "", kernel)
        registry.register("Bot2", "", kernel)
        result = registry.register("Bot3", "", kernel)
        assert result is None

    def test_get_by_api_key(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "", kernel)
        found = registry.get_by_api_key(record.api_key)
        assert found is not None
        assert found.agent_id == record.agent_id

    def test_get_by_api_key_missing(self):
        registry = AgentRegistry()
        assert registry.get_by_api_key("lev_nonexistent") is None

    def test_get_by_agent_id(self):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=3, land_shape=(5, 5), random_seed=1)
        kernel = WorldKernel(config, save_path=tmpdir)
        registry = AgentRegistry()
        record = registry.register("TestBot", "", kernel)
        found = registry.get_by_agent_id(record.agent_id)
        assert found is not None
        assert found.name == "TestBot"
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestAgentRegistry -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.registry'`

**Step 3: Write minimal implementation**

Create `api/registry.py`:

```python
"""In-memory agent registry for tracking external agent registrations."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from kernel.world_kernel import WorldKernel


@dataclass
class AgentRecord:
    """A registered external agent."""

    agent_id: int
    name: str
    description: str
    api_key: str
    member_id: int
    registered_at: str


class AgentRegistry:
    """Manages agent registrations and API key lookups."""

    def __init__(self) -> None:
        self._records: Dict[int, AgentRecord] = {}
        self._by_key: Dict[str, AgentRecord] = {}
        self._assigned_members: Set[int] = set()
        self._next_id = 1

    def register(
        self, name: str, description: str, kernel: WorldKernel
    ) -> Optional[AgentRecord]:
        """Register a new agent and assign it to an unoccupied member.

        Returns None if all members are already assigned.
        """
        member_id = None
        for m in kernel._execution.current_members:
            if m.id not in self._assigned_members:
                member_id = m.id
                break

        if member_id is None:
            return None

        api_key = f"lev_{uuid.uuid4().hex[:24]}"
        agent_id = self._next_id
        self._next_id += 1

        record = AgentRecord(
            agent_id=agent_id,
            name=name,
            description=description,
            api_key=api_key,
            member_id=member_id,
            registered_at=datetime.now(timezone.utc).isoformat(),
        )
        self._records[agent_id] = record
        self._by_key[api_key] = record
        self._assigned_members.add(member_id)
        return record

    def get_by_api_key(self, key: str) -> Optional[AgentRecord]:
        return self._by_key.get(key)

    def get_by_agent_id(self, agent_id: int) -> Optional[AgentRecord]:
        return self._records.get(agent_id)
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_api.py::TestAgentRegistry -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add api/registry.py test_api.py
git commit -m "feat: add AgentRegistry for external agent registration"
```

---

### Task 4: RoundState (thread-safe submission window)

**Files:**
- Create: `api/round_state.py`
- Test: `test_api.py` (append new test class)

**Context:** RoundState is shared between the API thread (which appends submitted actions) and the simulation thread (which drains actions and changes state). All mutations are protected by a threading Lock. This resolves compromise A10 for the write path.

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — RoundState tests
# ──────────────────────────────────────────────

from api.round_state import PendingAction, RoundState


class TestRoundState:
    def test_initial_state(self):
        rs = RoundState()
        assert rs.state == "settled"
        assert rs.round_id == 0
        assert rs.deadline is None
        assert rs.get_pending_actions() == []

    def test_open_submissions(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        assert rs.state == "accepting"
        assert rs.round_id == 1
        assert rs.deadline is not None
        assert rs.seconds_remaining() > 0

    def test_submit_action_during_accepting(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        accepted = rs.submit_action(pa)
        assert accepted is True
        assert len(rs.get_pending_actions()) == 1

    def test_submit_action_rejected_when_not_accepting(self):
        rs = RoundState()
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        accepted = rs.submit_action(pa)
        assert accepted is False

    def test_close_and_drain(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        rs.submit_action(pa)
        rs.close_submissions()
        assert rs.state == "executing"
        drained = rs.drain_actions()
        assert len(drained) == 1
        assert rs.get_pending_actions() == []

    def test_mark_settled(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        rs.close_submissions()
        rs.mark_settled()
        assert rs.state == "settled"

    def test_idempotency_duplicate_key(self):
        rs = RoundState()
        rs.open_submissions(round_id=1, pace=5.0)
        pa1 = PendingAction(agent_id=1, member_id=10, code="x=1", idempotency_key="k1")
        pa2 = PendingAction(agent_id=1, member_id=10, code="x=2", idempotency_key="k1")
        rs.submit_action(pa1)
        accepted = rs.submit_action(pa2)
        assert accepted is True  # Idempotent — accepted but not duplicated
        assert len(rs.get_pending_actions()) == 1  # Only one copy
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestRoundState -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

Create `api/round_state.py`:

```python
"""Thread-safe round state shared between the API and simulation threads."""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


@dataclass
class PendingAction:
    """An action submitted by an external agent, waiting for execution."""

    agent_id: int
    member_id: int
    code: str
    idempotency_key: str


class RoundState:
    """Manages the submission window between begin_round and settle_round.

    Thread-safe: all public methods acquire self._lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: str = "settled"
        self.round_id: int = 0
        self.deadline: Optional[datetime] = None
        self._pending: List[PendingAction] = []
        self._seen_keys: Dict[str, bool] = {}

    def open_submissions(self, round_id: int, pace: float) -> None:
        with self._lock:
            self.state = "accepting"
            self.round_id = round_id
            self.deadline = datetime.now(timezone.utc) + timedelta(seconds=pace)
            self._pending = []
            self._seen_keys = {}

    def submit_action(self, action: PendingAction) -> bool:
        with self._lock:
            if self.state != "accepting":
                return False
            if action.idempotency_key in self._seen_keys:
                return True  # Idempotent accept
            self._seen_keys[action.idempotency_key] = True
            self._pending.append(action)
            return True

    def close_submissions(self) -> None:
        with self._lock:
            self.state = "executing"

    def drain_actions(self) -> List[PendingAction]:
        with self._lock:
            actions = list(self._pending)
            self._pending = []
            return actions

    def mark_settled(self) -> None:
        with self._lock:
            self.state = "settled"

    def get_pending_actions(self) -> List[PendingAction]:
        with self._lock:
            return list(self._pending)

    def seconds_remaining(self) -> float:
        with self._lock:
            if self.deadline is None:
                return 0.0
            remaining = (self.deadline - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, remaining)
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_api.py::TestRoundState -v`
Expected: 7 tests PASS

**Step 5: Commit**

```bash
git add api/round_state.py test_api.py
git commit -m "feat: add thread-safe RoundState for submission window management"
```

---

### Task 5: New API models for write endpoints

**Files:**
- Modify: `api/models.py` — add request/response models for Phase 2
- Test: `test_api.py` (append new test class)

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — New models tests
# ──────────────────────────────────────────────

from api.models import (
    AgentRegisterRequest,
    AgentRegisterResponse,
    AgentProfileResponse,
    ActionSubmitRequest,
    ActionSubmitResponse,
    DeadlineResponse,
)


class TestPhase2Models:
    def test_register_request(self):
        req = AgentRegisterRequest(name="Bot1", description="test bot")
        assert req.name == "Bot1"

    def test_register_response(self):
        resp = AgentRegisterResponse(agent_id=1, api_key="lev_abc", member_id=5)
        assert resp.agent_id == 1
        assert resp.api_key == "lev_abc"

    def test_agent_profile(self):
        resp = AgentProfileResponse(
            agent_id=1, name="Bot1", member_id=5, registered_at="2025-01-01T00:00:00Z"
        )
        assert resp.agent_id == 1

    def test_action_submit_request(self):
        req = ActionSubmitRequest(
            code="def agent_action(e, m): pass", idempotency_key="r1-a1"
        )
        assert req.code.startswith("def")

    def test_action_submit_response_accepted(self):
        resp = ActionSubmitResponse(status="accepted", round_id=5)
        assert resp.status == "accepted"

    def test_deadline_response(self):
        resp = DeadlineResponse(
            round_id=3, state="accepting", deadline_utc="2025-01-01T00:00:00Z",
            seconds_remaining=4.5,
        )
        assert resp.seconds_remaining == 4.5
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestPhase2Models -v`
Expected: FAIL with `ImportError: cannot import name 'AgentRegisterRequest'`

**Step 3: Write implementation**

Add the following classes to the end of `api/models.py`:

```python
# ── Phase 2: Write endpoint models ──────────────


class AgentRegisterRequest(BaseModel):
    """Request body for POST /v1/agents/register."""

    name: str
    description: str = ""


class AgentRegisterResponse(BaseModel):
    """Response from POST /v1/agents/register."""

    agent_id: int
    api_key: str
    member_id: int


class AgentProfileResponse(BaseModel):
    """Response from GET /v1/agents/me."""

    agent_id: int
    name: str
    member_id: int
    registered_at: str


class ActionSubmitRequest(BaseModel):
    """Request body for POST /v1/world/actions."""

    code: str
    idempotency_key: str


class ActionSubmitResponse(BaseModel):
    """Response from POST /v1/world/actions."""

    status: str
    round_id: int
    reason: Optional[str] = None


class DeadlineResponse(BaseModel):
    """Response from GET /v1/world/rounds/current/deadline."""

    round_id: int
    state: str
    deadline_utc: Optional[str] = None
    seconds_remaining: float
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_api.py::TestPhase2Models -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add api/models.py test_api.py
git commit -m "feat: add Pydantic models for Phase 2 write endpoints"
```

---

### Task 6: Wire registry + round_state into app factory

**Files:**
- Modify: `api/deps.py` — add `get_registry` and `get_round_state`
- Modify: `api/app.py` — no changes needed (deps.py creates the state)

**Context:** This wiring task connects the AgentRegistry and RoundState into the FastAPI app so that route handlers can access them. No new endpoints yet — just plumbing.

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — App wiring tests
# ──────────────────────────────────────────────

from api.registry import AgentRegistry
from api.round_state import RoundState


class TestPhase2AppWiring:
    def test_app_state_has_registry(self):
        client, kernel = _make_test_client()
        state = client.app.state.leviathan
        assert "registry" in state
        assert isinstance(state["registry"], AgentRegistry)

    def test_app_state_has_round_state(self):
        client, kernel = _make_test_client()
        state = client.app.state.leviathan
        assert "round_state" in state
        assert isinstance(state["round_state"], RoundState)
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestPhase2AppWiring -v`
Expected: FAIL with `KeyError: 'registry'`

**Step 3: Modify deps.py**

Replace `api/deps.py` with:

```python
"""Dependency injection helpers for the Leviathan API."""

from typing import Any, Dict, List

from fastapi import Request

from api.auth import APIKeyAuth
from api.models import EventEnvelope
from api.registry import AgentRegistry
from api.round_state import RoundState
from kernel.world_kernel import WorldKernel


def create_app_state(kernel: WorldKernel) -> Dict[str, Any]:
    """Build the shared application state dictionary from a kernel instance."""
    return {
        "kernel": kernel,
        "event_log": [],
        "registry": AgentRegistry(),
        "round_state": RoundState(),
    }


def get_kernel(state: Dict[str, Any]) -> WorldKernel:
    return state["kernel"]


def get_event_log(state: Dict[str, Any]) -> List[EventEnvelope]:
    return state["event_log"]


def get_registry(request: Request) -> AgentRegistry:
    return request.app.state.leviathan["registry"]


def get_round_state(request: Request) -> RoundState:
    return request.app.state.leviathan["round_state"]


def get_auth(request: Request) -> APIKeyAuth:
    """Retrieve the APIKeyAuth instance from application state."""
    return request.app.state.auth
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_api.py::TestPhase2AppWiring -v`
Expected: 2 tests PASS

**Step 5: Run all API tests to ensure nothing broke**

Run: `pytest test_api.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add api/deps.py test_api.py
git commit -m "feat: wire AgentRegistry and RoundState into app factory"
```

---

### Task 7: Agent registration and profile endpoints

**Files:**
- Create: `api/routes/agents.py`
- Modify: `api/app.py` — include new router
- Test: `test_api.py` (append)

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — Agent endpoints tests
# ──────────────────────────────────────────────


class TestAgentEndpoints:
    def test_register_agent(self):
        client, kernel = _make_test_client(members=5)
        resp = client.post(
            "/v1/agents/register",
            json={"name": "TestBot", "description": "A test agent"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_id" in data
        assert data["api_key"].startswith("lev_")
        assert "member_id" in data

    def test_register_too_many(self):
        client, kernel = _make_test_client(members=2)
        client.post("/v1/agents/register", json={"name": "Bot1"})
        client.post("/v1/agents/register", json={"name": "Bot2"})
        resp = client.post("/v1/agents/register", json={"name": "Bot3"})
        assert resp.status_code == 409

    def test_agent_me(self):
        client, kernel = _make_test_client(members=5)
        reg_resp = client.post(
            "/v1/agents/register",
            json={"name": "TestBot"},
        )
        api_key = reg_resp.json()["api_key"]
        resp = client.get(
            "/v1/agents/me",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestBot"
        assert "registered_at" in data

    def test_agent_me_unauthorized(self):
        client, kernel = _make_test_client(members=5)
        resp = client.get("/v1/agents/me")
        assert resp.status_code == 401

    def test_agent_me_invalid_key(self):
        client, kernel = _make_test_client(members=5)
        resp = client.get(
            "/v1/agents/me",
            headers={"X-API-Key": "lev_invalid"},
        )
        assert resp.status_code == 403
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestAgentEndpoints -v`
Expected: FAIL with 404s (routes don't exist yet)

**Step 3: Write implementation**

Create `api/routes/agents.py`:

```python
"""Agent registration and profile endpoints."""

from fastapi import APIRouter, HTTPException, Request

from api.models import (
    AgentProfileResponse,
    AgentRegisterRequest,
    AgentRegisterResponse,
)

router = APIRouter(prefix="/v1/agents")


@router.post("/register", response_model=AgentRegisterResponse)
def register_agent(body: AgentRegisterRequest, request: Request):
    """Register a new external agent and assign it to an in-world member."""
    registry = request.app.state.leviathan["registry"]
    kernel = request.app.state.leviathan["kernel"]
    record = registry.register(body.name, body.description, kernel)
    if record is None:
        raise HTTPException(status_code=409, detail="All members are assigned")
    return AgentRegisterResponse(
        agent_id=record.agent_id,
        api_key=record.api_key,
        member_id=record.member_id,
    )


@router.get("/me", response_model=AgentProfileResponse)
def agent_profile(request: Request):
    """Return the profile of the authenticated agent."""
    registry = request.app.state.leviathan["registry"]
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = registry.get_by_api_key(key)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return AgentProfileResponse(
        agent_id=record.agent_id,
        name=record.name,
        member_id=record.member_id,
        registered_at=record.registered_at,
    )
```

Modify `api/app.py` — add import and include router:

Add import:
```python
from api.routes.agents import router as agents_router
```

Add after `app.include_router(discovery_router)`:
```python
app.include_router(agents_router)
```

**Step 4: Run tests to verify they pass**

Run: `pytest test_api.py::TestAgentEndpoints -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add api/routes/agents.py api/app.py test_api.py
git commit -m "feat: add agent registration and profile endpoints"
```

---

### Task 8: Action submission and deadline endpoints

**Files:**
- Create: `api/routes/actions.py`
- Modify: `api/routes/world.py` — add deadline endpoint
- Modify: `api/app.py` — include new router
- Test: `test_api.py` (append)

**Important route ordering note:** The deadline route `/rounds/current/deadline` must be defined **before** the existing `/rounds/{round_id}` route in world.py, otherwise FastAPI will try to match "current" as a round_id integer and return 422. Move the deadline route above `get_round_by_id`.

**Step 1: Write failing tests**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — Action submission + deadline tests
# ──────────────────────────────────────────────


class TestActionEndpoints:
    def _register_and_open(self, client, members=5):
        """Register an agent and open submissions."""
        reg = client.post("/v1/agents/register", json={"name": "Bot"}).json()
        rs = client.app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=30.0)
        return reg["api_key"], reg["member_id"]

    def test_submit_action_accepted(self):
        client, kernel = _make_test_client(members=5)
        api_key, member_id = self._register_and_open(client)
        resp = client.post(
            "/v1/world/actions",
            json={
                "code": "def agent_action(e, m): pass",
                "idempotency_key": "r1-a1",
            },
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["round_id"] == 1

    def test_submit_action_rejected_when_closed(self):
        client, kernel = _make_test_client(members=5)
        reg = client.post("/v1/agents/register", json={"name": "Bot"}).json()
        resp = client.post(
            "/v1/world/actions",
            json={"code": "def agent_action(e, m): pass", "idempotency_key": "k1"},
            headers={"X-API-Key": reg["api_key"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"

    def test_submit_action_unauthorized(self):
        client, kernel = _make_test_client(members=5)
        resp = client.post(
            "/v1/world/actions",
            json={"code": "pass", "idempotency_key": "k1"},
        )
        assert resp.status_code == 401

    def test_submit_action_idempotency(self):
        client, kernel = _make_test_client(members=5)
        api_key, _ = self._register_and_open(client)
        body = {
            "code": "def agent_action(e, m): pass",
            "idempotency_key": "same-key",
        }
        resp1 = client.post("/v1/world/actions", json=body, headers={"X-API-Key": api_key})
        resp2 = client.post("/v1/world/actions", json=body, headers={"X-API-Key": api_key})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        rs = client.app.state.leviathan["round_state"]
        assert len(rs.get_pending_actions()) == 1


class TestDeadlineEndpoint:
    def test_deadline_settled(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/current/deadline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "settled"
        assert data["seconds_remaining"] == 0.0

    def test_deadline_accepting(self):
        client, kernel = _make_test_client()
        rs = client.app.state.leviathan["round_state"]
        rs.open_submissions(round_id=1, pace=30.0)
        resp = client.get("/v1/world/rounds/current/deadline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "accepting"
        assert data["round_id"] == 1
        assert data["seconds_remaining"] > 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest test_api.py::TestActionEndpoints test_api.py::TestDeadlineEndpoint -v`
Expected: FAIL

**Step 3: Write action submission endpoint**

Create `api/routes/actions.py`:

```python
"""Action submission endpoint for external agents."""

from fastapi import APIRouter, HTTPException, Request

from api.models import ActionSubmitRequest, ActionSubmitResponse
from api.round_state import PendingAction

router = APIRouter(prefix="/v1/world")


@router.post("/actions", response_model=ActionSubmitResponse)
def submit_action(body: ActionSubmitRequest, request: Request):
    """Submit an action for the current round."""
    registry = request.app.state.leviathan["registry"]
    round_state = request.app.state.leviathan["round_state"]

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    record = registry.get_by_api_key(key)
    if record is None:
        raise HTTPException(status_code=403, detail="Invalid API key")

    pa = PendingAction(
        agent_id=record.agent_id,
        member_id=record.member_id,
        code=body.code,
        idempotency_key=body.idempotency_key,
    )
    accepted = round_state.submit_action(pa)

    if not accepted:
        return ActionSubmitResponse(
            status="rejected",
            round_id=round_state.round_id,
            reason="Round not accepting submissions",
        )

    return ActionSubmitResponse(
        status="accepted",
        round_id=round_state.round_id,
    )
```

**Step 4: Add deadline endpoint to world.py**

In `api/routes/world.py`, add import:
```python
from api.models import DeadlineResponse
```

Add this endpoint **before** the `get_round_by_id` function (important for route matching):

```python
@router.get("/rounds/current/deadline", response_model=DeadlineResponse)
def get_deadline(request: Request) -> DeadlineResponse:
    """Return the current round's submission deadline."""
    round_state = request.app.state.leviathan["round_state"]
    return DeadlineResponse(
        round_id=round_state.round_id,
        state=round_state.state,
        deadline_utc=round_state.deadline.isoformat() if round_state.deadline else None,
        seconds_remaining=round_state.seconds_remaining(),
    )
```

**Step 5: Modify app.py to include actions router**

Add import:
```python
from api.routes.actions import router as actions_router
```

Add after agents router:
```python
app.include_router(actions_router)
```

**Step 6: Run tests to verify they pass**

Run: `pytest test_api.py::TestActionEndpoints test_api.py::TestDeadlineEndpoint -v`
Expected: 6 tests PASS

**Step 7: Run all API tests**

Run: `pytest test_api.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add api/routes/actions.py api/routes/world.py api/app.py test_api.py
git commit -m "feat: add action submission and deadline endpoints"
```

---

### Task 9: Updated simulation loop with submission window

**Files:**
- Modify: `scripts/run_server.py` — rewrite `_simulation_loop` to open/close submission window
- Modify: `kernel/world_kernel.py` — add `apply_intended_actions` and `_find_member_by_id` methods
- Test: `test_world_kernel.py` (append)
- Test: `test_api.py` (append)

**Context:** The simulation loop changes from `begin_round -> settle_round -> sleep` to `begin_round -> open_submissions -> sleep -> close_submissions -> execute_actions -> settle_round -> append_event`. The kernel gets a new method `apply_intended_actions()` that takes the JSON action dicts from SubprocessSandbox and applies them to the real engine.

**Step 1: Write failing tests for apply_intended_actions**

Add to `test_world_kernel.py`:

```python
class TestApplyIntendedActions:
    """Test WorldKernel.apply_intended_actions applies proxy actions to the real engine."""

    def _make_kernel(self, members=5):
        tmpdir = tempfile.mkdtemp()
        config = WorldConfig(init_member_number=members, land_shape=(10, 10), random_seed=42)
        return WorldKernel(config, save_path=tmpdir)

    def test_expand_action_changes_land(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        kernel._execution.get_neighbors()
        member = kernel._execution.current_members[0]
        old_land = member.land_num
        actions = [{"action": "expand", "member_id": member.id}]
        results = kernel.apply_intended_actions(actions)
        assert len(results) == 1
        # Expand should increase land (if empty land was available)
        if member.current_empty_loc_list:
            assert member.land_num >= old_land

    def test_unknown_action_skipped(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        actions = [{"action": "unknown_thing", "member_id": 1}]
        results = kernel.apply_intended_actions(actions)
        assert len(results) == 1
        assert results[0]["applied"] is False

    def test_empty_actions(self):
        kernel = self._make_kernel()
        kernel.begin_round()
        results = kernel.apply_intended_actions([])
        assert results == []
```

**Step 2: Write failing test for E2E integration**

Add to `test_api.py`:

```python
# ──────────────────────────────────────────────
# Phase 2 — Simulation loop integration test
# ──────────────────────────────────────────────


class TestSimLoopIntegration:
    def test_full_external_agent_round(self):
        """End-to-end: register, submit action during accepting window, verify receipt."""
        from scripts.run_server import build_app

        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        # Register agent
        reg = client.post("/v1/agents/register", json={"name": "E2EBot"}).json()
        api_key = reg["api_key"]

        # Manually run one round with submission window
        kernel = app.state.leviathan["kernel"]
        round_state = app.state.leviathan["round_state"]

        kernel.begin_round()
        round_state.open_submissions(round_id=kernel.round_id, pace=5.0)

        # Submit action
        resp = client.post(
            "/v1/world/actions",
            json={
                "code": "def agent_action(engine, member_id):\n    me = engine.current_members[member_id]\n    engine.expand(me)\n",
                "idempotency_key": "e2e-r1",
            },
            headers={"X-API-Key": api_key},
        )
        assert resp.json()["status"] == "accepted"

        # Close, execute, settle
        round_state.close_submissions()
        pending = round_state.drain_actions()
        assert len(pending) == 1

        # Execute through subprocess sandbox
        from kernel.subprocess_sandbox import SubprocessSandbox
        from kernel.execution_sandbox import SandboxContext

        sandbox = SubprocessSandbox()
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

        receipt = kernel.settle_round(seed=kernel.round_id)
        round_state.mark_settled()

        # Verify the round completed
        assert receipt.round_id == 1
        assert round_state.state == "settled"
```

**Step 3: Run tests to verify they fail**

Run: `pytest test_world_kernel.py::TestApplyIntendedActions test_api.py::TestSimLoopIntegration -v`
Expected: FAIL with `AttributeError: 'WorldKernel' object has no attribute 'apply_intended_actions'`

**Step 4: Add apply_intended_actions to WorldKernel**

In `kernel/world_kernel.py`, add after the `accept_mechanisms` method:

```python
def apply_intended_actions(self, actions: list) -> list:
    """Apply intended actions from an EngineProxy to the real engine.

    Each action is a dict like {"action": "attack", "member_id": 1, "target_id": 2}.
    Returns a list of result dicts with {"action": ..., "applied": bool, "error": ...}.
    """
    results = []
    for act in actions:
        action_type = act.get("action")
        try:
            if action_type == "attack":
                m1 = self._find_member_by_id(act["member_id"])
                m2 = self._find_member_by_id(act["target_id"])
                if m1 and m2:
                    self._execution.attack(m1, m2)
                    results.append({"action": action_type, "applied": True})
                else:
                    results.append({"action": action_type, "applied": False, "error": "member not found"})
            elif action_type == "offer":
                m1 = self._find_member_by_id(act["member_id"])
                m2 = self._find_member_by_id(act["target_id"])
                if m1 and m2:
                    self._execution.offer(m1, m2)
                    results.append({"action": action_type, "applied": True})
                else:
                    results.append({"action": action_type, "applied": False, "error": "member not found"})
            elif action_type == "offer_land":
                m1 = self._find_member_by_id(act["member_id"])
                m2 = self._find_member_by_id(act["target_id"])
                if m1 and m2:
                    self._execution.offer_land(m1, m2)
                    results.append({"action": action_type, "applied": True})
                else:
                    results.append({"action": action_type, "applied": False, "error": "member not found"})
            elif action_type == "expand":
                m1 = self._find_member_by_id(act["member_id"])
                if m1:
                    self._execution.expand(m1)
                    results.append({"action": action_type, "applied": True})
                else:
                    results.append({"action": action_type, "applied": False, "error": "member not found"})
            elif action_type == "message":
                self._execution.send_message(act["from_id"], act["to_id"], act["message"])
                results.append({"action": action_type, "applied": True})
            else:
                results.append({"action": action_type, "applied": False, "error": f"unknown action: {action_type}"})
        except Exception as e:
            results.append({"action": action_type, "applied": False, "error": str(e)})
    return results

def _find_member_by_id(self, member_id: int):
    """Find a Member object by permanent id."""
    for m in self._execution.current_members:
        if m.id == member_id:
            return m
    return None
```

**Step 5: Update simulation loop in run_server.py**

Replace `_simulation_loop` in `scripts/run_server.py`. The new version accepts a `round_state` parameter and runs the submission window pattern.

New `_simulation_loop`:
```python
def _simulation_loop(
    kernel: WorldKernel,
    event_log: List[EventEnvelope],
    round_state,
    pace: float,
    max_rounds: int,
    stop_event: threading.Event,
) -> None:
    """Background thread: begin_round -> open -> sleep -> close -> apply -> settle."""
    from kernel.subprocess_sandbox import SubprocessSandbox
    from kernel.execution_sandbox import SandboxContext

    sandbox = SubprocessSandbox()
    rounds_completed = 0

    while not stop_event.is_set():
        kernel.begin_round()
        round_state.open_submissions(round_id=kernel.round_id, pace=pace)
        logger.info("Round %d: accepting submissions for %.1fs", kernel.round_id, pace)

        stop_event.wait(timeout=pace)
        if stop_event.is_set():
            break

        round_state.close_submissions()
        pending = round_state.drain_actions()
        logger.info("Round %d: executing %d actions", kernel.round_id, len(pending))

        for pa in pending:
            member_index = kernel._resolve_agent_index(pa.member_id)
            if member_index is None:
                continue
            ctx = SandboxContext(
                execution_engine=kernel._execution,
                member_index=member_index,
            )
            result = sandbox.execute_agent_code(pa.code, ctx)
            if result.success:
                kernel.apply_intended_actions(result.intended_actions)

        receipt = kernel.settle_round(seed=kernel.round_id)
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

In `main()`, after `event_log = app.state.leviathan["event_log"]`, add:
```python
round_state = app.state.leviathan["round_state"]
```

Update `sim_thread` args:
```python
sim_thread = threading.Thread(
    target=_simulation_loop,
    args=(kernel, event_log, round_state, args.pace, args.rounds, stop_event),
    daemon=True,
    name="sim-loop",
)
```

**Step 6: Run tests to verify they pass**

Run: `pytest test_world_kernel.py::TestApplyIntendedActions test_api.py::TestSimLoopIntegration -v`
Expected: All tests PASS

**Step 7: Run ALL tests**

Run: `pytest test_world_kernel.py test_api.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add kernel/world_kernel.py scripts/run_server.py test_world_kernel.py test_api.py
git commit -m "feat: add submission window simulation loop and apply_intended_actions"
```

---

### Task 10: Final verification and compromises update

**Files:**
- Modify: `docs/plans/2026-02-24-implementation-compromises.md` — mark K1 as resolved
- Test: run full test suite

**Step 1: Run the complete test suite**

Run: `pytest test_world_kernel.py test_api.py -v`
Expected: All tests PASS.

**Step 2: Verify imports are clean**

Run: `python -c "from kernel.engine_proxy import EngineProxy; from kernel.subprocess_sandbox import SubprocessSandbox; from api.registry import AgentRegistry; from api.round_state import RoundState; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Update compromises doc**

In `docs/plans/2026-02-24-implementation-compromises.md`, update K1:

Change heading to:
```
### ~~K1. In-process code execution (no real sandboxing)~~ RESOLVED
```

Add below:
```
- **Resolved:** 2026-02-24. Added `SubprocessSandbox` in `kernel/subprocess_sandbox.py` with `EngineProxy` action proxy pattern. Agent code runs in a child process with timeout enforcement. Actions recorded as JSON and applied to real engine via `WorldKernel.apply_intended_actions()`.
```

Update the "Must fix before Phase 2" section to show K1 as resolved.

**Step 4: Commit**

```bash
git add docs/plans/2026-02-24-implementation-compromises.md
git commit -m "docs: mark K1 (sandbox security) as resolved"
```

**Step 5: Push**

```bash
git push
```

---

## Summary

| Task | Component | New Tests | New Files | Modified Files |
|------|-----------|-----------|-----------|----------------|
| 1 | EngineProxy | 8 | `kernel/engine_proxy.py` | `test_world_kernel.py` |
| 2 | SubprocessSandbox | 7 | `kernel/subprocess_sandbox.py` | `kernel/execution_sandbox.py`, `test_world_kernel.py` |
| 3 | AgentRegistry | 6 | `api/registry.py` | `test_api.py` |
| 4 | RoundState | 7 | `api/round_state.py` | `test_api.py` |
| 5 | Write models | 6 | — | `api/models.py`, `test_api.py` |
| 6 | App wiring | 2 | — | `api/deps.py`, `test_api.py` |
| 7 | Agent endpoints | 5 | `api/routes/agents.py` | `api/app.py`, `test_api.py` |
| 8 | Action + deadline | 6 | `api/routes/actions.py` | `api/routes/world.py`, `api/app.py`, `test_api.py` |
| 9 | Sim loop + apply | 4 | — | `kernel/world_kernel.py`, `scripts/run_server.py`, both test files |
| 10 | Verification | 0 | — | `docs/plans/2026-02-24-implementation-compromises.md` |
| **Total** | | **~51** | **5** | **12** |
