# Phase 1 Read API — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose WorldKernel state via read-only HTTP endpoints so external tools can observe the simulation.

**Architecture:** FastAPI app in `api/` package wraps a single `WorldKernel` instance via dependency injection. Pydantic models serialize kernel dataclasses. A `scripts/run_server.py` creates the kernel, runs a sim loop in a background thread, and serves the API. All tests use FastAPI's TestClient with no real server.

**Tech Stack:** FastAPI, Pydantic v2, uvicorn, httpx (test client). All already installed.

---

### Task 1: Create api package with Pydantic response models

**Files:**
- Create: `api/__init__.py`
- Create: `api/models.py`
- Create: `test_api.py`

**Step 1: Write the failing test**

Create `test_api.py` at the repo root (matching `pytest.ini` pattern). Test the Pydantic models can be constructed and serialized:

```python
"""Tests for the Phase 1 Read API."""

import pytest
from api.models import WorldInfo, RoundInfo, RoundReceiptResponse, EventEnvelope, AgentDiscovery


class TestModels:
    def test_world_info_creation(self):
        info = WorldInfo(world_id="abc", round_id=1, member_count=5, state_hash="deadbeef")
        d = info.model_dump()
        assert d["world_id"] == "abc"
        assert d["round_id"] == 1
        assert d["member_count"] == 5

    def test_round_receipt_response(self):
        receipt = RoundReceiptResponse(
            round_id=1,
            seed=42,
            snapshot_hash_before="aaa",
            snapshot_hash_after="bbb",
            accepted_action_ids=["k1"],
            rejected_action_ids=[],
            activated_mechanism_ids=[],
            judge_results=[],
            round_metrics={},
            timestamp="abc123",
        )
        d = receipt.model_dump()
        assert d["round_id"] == 1
        assert d["seed"] == 42

    def test_round_info_no_receipt(self):
        info = RoundInfo(round_id=0, last_receipt=None)
        d = info.model_dump()
        assert d["last_receipt"] is None

    def test_event_envelope(self):
        env = EventEnvelope(
            event_id=1,
            event_type="round_settled",
            round_id=1,
            timestamp="abc",
            payload={"key": "val"},
        )
        d = env.model_dump()
        assert d["event_type"] == "round_settled"

    def test_agent_discovery(self):
        disc = AgentDiscovery(
            name="Leviathan",
            version="0.1.0",
            api_version="v1",
            capabilities=["read_snapshot"],
            endpoints={"base": "/v1/world"},
        )
        d = disc.model_dump()
        assert "read_snapshot" in d["capabilities"]
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestModels::test_world_info_creation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

**Step 3: Write minimal implementation**

Create `api/__init__.py`:
```python
"""Phase 1 Read API package."""
```

Create `api/models.py`:
```python
"""Pydantic response models for the Read API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class WorldInfo(BaseModel):
    """Summary metadata about the world."""
    world_id: str
    round_id: int
    member_count: int
    state_hash: str


class RoundReceiptResponse(BaseModel):
    """Serialized round receipt."""
    round_id: int
    seed: int
    snapshot_hash_before: str
    snapshot_hash_after: str
    accepted_action_ids: List[str]
    rejected_action_ids: List[str]
    activated_mechanism_ids: List[str]
    judge_results: List[Dict[str, Any]]
    round_metrics: Dict[str, float]
    timestamp: str


class RoundInfo(BaseModel):
    """Current round status."""
    round_id: int
    last_receipt: Optional[RoundReceiptResponse] = None


class EventEnvelope(BaseModel):
    """Wrapper for events in the polling feed."""
    event_id: int
    event_type: str
    round_id: int
    timestamp: str
    payload: Dict[str, Any]


class AgentDiscovery(BaseModel):
    """Agent discovery manifest."""
    name: str
    version: str
    api_version: str
    capabilities: List[str]
    endpoints: Dict[str, str]
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add api/__init__.py api/models.py test_api.py
git commit -m "feat(api): add Pydantic response models for Read API"
```

---

### Task 2: Add dependency injection module

**Files:**
- Create: `api/deps.py`
- Modify: `test_api.py` (append tests)

**Step 1: Write the failing test**

Append to `test_api.py`:

```python
import tempfile
from kernel import WorldKernel, WorldConfig
from api.deps import create_app_state, get_kernel, get_event_log
from api.models import EventEnvelope


class TestDeps:
    def test_create_app_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
            kernel = WorldKernel(config, save_path=tmpdir)
            state = create_app_state(kernel)
            assert state["kernel"] is kernel
            assert isinstance(state["event_log"], list)
            assert len(state["event_log"]) == 0

    def test_get_kernel_returns_kernel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
            kernel = WorldKernel(config, save_path=tmpdir)
            state = create_app_state(kernel)
            retrieved = get_kernel(state)
            assert retrieved is kernel

    def test_get_event_log_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WorldConfig(init_member_number=5, land_shape=(10, 10), random_seed=42)
            kernel = WorldKernel(config, save_path=tmpdir)
            state = create_app_state(kernel)
            log = get_event_log(state)
            assert isinstance(log, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestDeps::test_create_app_state -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `api/deps.py`:
```python
"""Dependency injection for the Read API.

The app state is a plain dict holding the kernel and event log.
FastAPI routes access these via Depends() using the accessor functions.
"""

from typing import Any, Dict, List

from kernel import WorldKernel
from api.models import EventEnvelope


def create_app_state(kernel: WorldKernel) -> Dict[str, Any]:
    """Build the shared application state dict."""
    return {
        "kernel": kernel,
        "event_log": [],
    }


def get_kernel(state: Dict[str, Any]) -> WorldKernel:
    """Extract the WorldKernel from app state."""
    return state["kernel"]


def get_event_log(state: Dict[str, Any]) -> List[EventEnvelope]:
    """Extract the event log from app state."""
    return state["event_log"]
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add api/deps.py test_api.py
git commit -m "feat(api): add dependency injection module"
```

---

### Task 3: Create FastAPI app factory with health endpoint

**Files:**
- Create: `api/app.py`
- Modify: `test_api.py` (append tests)

**Step 1: Write the failing test**

Append to `test_api.py`:

```python
import tempfile
from fastapi.testclient import TestClient
from kernel import WorldKernel, WorldConfig
from api.app import create_app


def _make_test_client(members=5, seed=42):
    """Helper: create a TestClient with a fresh WorldKernel."""
    tmpdir = tempfile.mkdtemp()
    config = WorldConfig(init_member_number=members, land_shape=(10, 10), random_seed=seed)
    kernel = WorldKernel(config, save_path=tmpdir)
    app = create_app(kernel)
    return TestClient(app), kernel


class TestHealthEndpoint:
    def test_health(self):
        client, _ = _make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestHealthEndpoint::test_health -v`
Expected: FAIL with `ImportError: cannot import name 'create_app'`

**Step 3: Write minimal implementation**

Create `api/app.py`:
```python
"""FastAPI application factory for the Read API."""

from fastapi import FastAPI

from kernel import WorldKernel
from api.deps import create_app_state


def create_app(kernel: WorldKernel) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        kernel: The WorldKernel instance to serve.

    Returns:
        Configured FastAPI app ready for uvicorn.
    """
    app = FastAPI(
        title="Leviathan API",
        version="0.1.0",
        description="Read-only API for observing Leviathan world simulation state.",
    )

    # Store shared state on the app instance
    app.state.leviathan = create_app_state(kernel)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add api/app.py test_api.py
git commit -m "feat(api): add FastAPI app factory with /health endpoint"
```

---

### Task 4: Add world info and snapshot endpoints

**Files:**
- Create: `api/routes/__init__.py`
- Create: `api/routes/world.py`
- Modify: `api/app.py` (register router)
- Modify: `test_api.py` (append tests)

**Step 1: Write the failing tests**

Append to `test_api.py`:

```python
class TestWorldEndpoints:
    def test_world_info(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 0
        assert data["member_count"] == 5
        assert len(data["state_hash"]) == 64
        assert "world_id" in data

    def test_snapshot(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["members"]) == 5
        assert "land" in data
        assert len(data["state_hash"]) == 64
        assert data["round_id"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestWorldEndpoints::test_world_info -v`
Expected: FAIL with 404 (route not registered)

**Step 3: Write minimal implementation**

Create `api/routes/__init__.py`:
```python
"""Route modules for the Read API."""
```

Create `api/routes/world.py`:
```python
"""World state endpoints: /v1/world/*"""

import dataclasses

from fastapi import APIRouter, Request

from api.models import WorldInfo, RoundReceiptResponse, RoundInfo, EventEnvelope
from kernel import WorldKernel
from typing import List

router = APIRouter(prefix="/v1/world", tags=["world"])


def _get_kernel(request: Request) -> WorldKernel:
    return request.app.state.leviathan["kernel"]


def _get_event_log(request: Request) -> List[EventEnvelope]:
    return request.app.state.leviathan["event_log"]


@router.get("", response_model=WorldInfo)
def world_info(request: Request):
    """Return summary metadata about the world."""
    kernel = _get_kernel(request)
    snapshot = kernel.get_snapshot()
    return WorldInfo(
        world_id=snapshot.world_id,
        round_id=snapshot.round_id,
        member_count=len(snapshot.members),
        state_hash=snapshot.state_hash,
    )


@router.get("/snapshot")
def world_snapshot(request: Request):
    """Return the full current world snapshot."""
    kernel = _get_kernel(request)
    snapshot = kernel.get_snapshot()
    return dataclasses.asdict(snapshot)
```

Update `api/app.py` — add router import and include after app creation:

Add to `create_app` function, after the health endpoint:
```python
    from api.routes.world import router as world_router
    app.include_router(world_router)
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add api/routes/__init__.py api/routes/world.py api/app.py test_api.py
git commit -m "feat(api): add /v1/world and /v1/world/snapshot endpoints"
```

---

### Task 5: Add round info and round-by-id endpoints

**Files:**
- Modify: `api/routes/world.py` (add endpoints)
- Modify: `test_api.py` (append tests)

**Step 1: Write the failing tests**

Append to `test_api.py`. These tests need a helper that settles rounds:

```python
def _settle_rounds(kernel, app, n=1):
    """Run n rounds on the kernel and append events to the app's event log."""
    event_log = app.state.leviathan["event_log"]
    for i in range(n):
        kernel.begin_round()
        receipt = kernel.settle_round(seed=kernel.round_id)
        event_log.append(EventEnvelope(
            event_id=len(event_log) + 1,
            event_type="round_settled",
            round_id=receipt.round_id,
            timestamp=receipt.timestamp,
            payload=dataclasses.asdict(receipt),
        ))


class TestRoundEndpoints:
    def test_rounds_current_before_any_round(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 0
        assert data["last_receipt"] is None

    def test_rounds_current_after_settle(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=1)
        resp = client.get("/v1/world/rounds/current")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 1
        assert data["last_receipt"] is not None
        assert data["last_receipt"]["round_id"] == 1

    def test_round_by_id(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=2)
        resp = client.get("/v1/world/rounds/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round_id"] == 1

    def test_round_not_found(self):
        client, kernel = _make_test_client()
        resp = client.get("/v1/world/rounds/999")
        assert resp.status_code == 404
```

Note: add `import dataclasses` and `from api.models import EventEnvelope` at the top of `test_api.py` if not already present.

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestRoundEndpoints::test_rounds_current_before_any_round -v`
Expected: FAIL with 404 or `Method Not Allowed`

**Step 3: Write minimal implementation**

Add to `api/routes/world.py`:

```python
from fastapi import HTTPException

@router.get("/rounds/current", response_model=RoundInfo)
def rounds_current(request: Request):
    """Return the current round status and last receipt."""
    kernel = _get_kernel(request)
    receipt = kernel.get_round_receipt()
    receipt_resp = None
    if receipt is not None:
        receipt_resp = RoundReceiptResponse(**dataclasses.asdict(receipt))
    return RoundInfo(round_id=kernel.round_id, last_receipt=receipt_resp)


@router.get("/rounds/{round_id}", response_model=RoundReceiptResponse)
def round_by_id(request: Request, round_id: int):
    """Return the receipt for a specific round."""
    event_log = _get_event_log(request)
    for event in event_log:
        if event.event_type == "round_settled" and event.round_id == round_id:
            return RoundReceiptResponse(**event.payload)
    raise HTTPException(status_code=404, detail=f"Round {round_id} not found")
```

**Important ordering note:** The `/rounds/current` route MUST be defined BEFORE `/rounds/{round_id}` in the code, otherwise FastAPI will try to parse "current" as an integer and return a 422 error.

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 15 tests PASS

**Step 5: Commit**

```bash
git add api/routes/world.py test_api.py
git commit -m "feat(api): add /v1/world/rounds/current and /v1/world/rounds/{round_id}"
```

---

### Task 6: Add events polling endpoint

**Files:**
- Modify: `api/routes/world.py` (add endpoint)
- Modify: `test_api.py` (append tests)

**Step 1: Write the failing tests**

Append to `test_api.py`:

```python
class TestEventsEndpoint:
    def test_events_polling(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=3)
        resp = client.get("/v1/world/events", params={"since_round": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2  # rounds 2 and 3
        assert data[0]["round_id"] == 2
        assert data[1]["round_id"] == 3

    def test_events_all(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=2)
        resp = client.get("/v1/world/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_events_empty(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=2)
        resp = client.get("/v1/world/events", params={"since_round": 999})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 0

    def test_events_monotonic_ids(self):
        client, kernel = _make_test_client()
        app = client.app
        _settle_rounds(kernel, app, n=3)
        resp = client.get("/v1/world/events")
        data = resp.json()
        ids = [e["event_id"] for e in data]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # all unique
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestEventsEndpoint::test_events_polling -v`
Expected: FAIL with 404 or `Method Not Allowed`

**Step 3: Write minimal implementation**

Add to `api/routes/world.py`:

```python
from typing import Optional

@router.get("/events", response_model=List[EventEnvelope])
def events(request: Request, since_round: Optional[int] = None):
    """Return events, optionally filtered to rounds after since_round."""
    event_log = _get_event_log(request)
    if since_round is not None:
        return [e for e in event_log if e.round_id > since_round]
    return event_log
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 19 tests PASS

**Step 5: Commit**

```bash
git add api/routes/world.py test_api.py
git commit -m "feat(api): add /v1/world/events polling endpoint"
```

---

### Task 7: Add agent discovery endpoint

**Files:**
- Create: `api/routes/discovery.py`
- Modify: `api/app.py` (register router)
- Modify: `test_api.py` (append test)

**Step 1: Write the failing test**

Append to `test_api.py`:

```python
class TestDiscovery:
    def test_discovery_manifest(self):
        client, _ = _make_test_client()
        resp = client.get("/.well-known/leviathan-agent.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Leviathan"
        assert data["api_version"] == "v1"
        assert "read_snapshot" in data["capabilities"]
        assert "read_events" in data["capabilities"]
        assert "read_receipts" in data["capabilities"]
        assert data["endpoints"]["base"] == "/v1/world"
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestDiscovery::test_discovery_manifest -v`
Expected: FAIL with 404

**Step 3: Write minimal implementation**

Create `api/routes/discovery.py`:
```python
"""Agent discovery endpoint: /.well-known/leviathan-agent.json"""

from fastapi import APIRouter

from api.models import AgentDiscovery

router = APIRouter(tags=["discovery"])


@router.get("/.well-known/leviathan-agent.json", response_model=AgentDiscovery)
def agent_discovery():
    """Return the agent discovery manifest."""
    return AgentDiscovery(
        name="Leviathan",
        version="0.1.0",
        api_version="v1",
        capabilities=["read_snapshot", "read_events", "read_receipts"],
        endpoints={"base": "/v1/world"},
    )
```

Update `api/app.py` — add inside `create_app`, after the world router include:
```python
    from api.routes.discovery import router as discovery_router
    app.include_router(discovery_router)
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 20 tests PASS

**Step 5: Commit**

```bash
git add api/routes/discovery.py api/app.py test_api.py
git commit -m "feat(api): add /.well-known/leviathan-agent.json discovery endpoint"
```

---

### Task 8: Add server runner script

**Files:**
- Create: `scripts/run_server.py`
- Modify: `test_api.py` (append integration test)

**Step 1: Write the failing test**

Append to `test_api.py` — test that the runner module's `build_app` helper works end-to-end:

```python
class TestServerIntegration:
    def test_build_app_creates_working_api(self):
        """Integration: build_app produces an app that serves all endpoints."""
        from scripts.run_server import build_app
        app = build_app(members=5, land_w=10, land_h=10, seed=42)
        client = TestClient(app)

        # Health
        assert client.get("/health").status_code == 200

        # World info
        resp = client.get("/v1/world")
        assert resp.status_code == 200
        assert resp.json()["member_count"] == 5

        # Discovery
        resp = client.get("/.well-known/leviathan-agent.json")
        assert resp.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `pytest test_api.py::TestServerIntegration::test_build_app_creates_working_api -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Create `scripts/run_server.py`:
```python
#!/usr/bin/env python3
"""Run the Leviathan Read API server with a live simulation loop.

Usage:
    python scripts/run_server.py --members 10 --land 20x20 --seed 42 --port 8000
"""

import argparse
import dataclasses
import tempfile
import threading
import time
import signal
import sys

from kernel import WorldKernel, WorldConfig
from api.app import create_app
from api.models import EventEnvelope


def build_app(members: int = 10, land_w: int = 20, land_h: int = 20,
              seed: int = 42) -> "FastAPI":
    """Create a fully configured FastAPI app with a WorldKernel.

    This is the testable entry point — no server, no threads.
    """
    tmpdir = tempfile.mkdtemp()
    config = WorldConfig(
        init_member_number=members,
        land_shape=(land_w, land_h),
        random_seed=seed,
    )
    kernel = WorldKernel(config, save_path=tmpdir)
    app = create_app(kernel)
    return app


def _simulation_loop(kernel, event_log, pace, max_rounds, stop_event):
    """Run simulation rounds in a background thread."""
    round_count = 0
    while not stop_event.is_set():
        if max_rounds > 0 and round_count >= max_rounds:
            break
        kernel.begin_round()
        receipt = kernel.settle_round(seed=kernel.round_id)
        event_log.append(EventEnvelope(
            event_id=len(event_log) + 1,
            event_type="round_settled",
            round_id=receipt.round_id,
            timestamp=receipt.timestamp,
            payload=dataclasses.asdict(receipt),
        ))
        round_count += 1
        stop_event.wait(timeout=pace)


def main():
    parser = argparse.ArgumentParser(description="Leviathan Read API Server")
    parser.add_argument("--members", type=int, default=10, help="Number of initial members")
    parser.add_argument("--land", type=str, default="20x20", help="Land shape WxH")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--rounds", type=int, default=0, help="Max rounds (0=infinite)")
    parser.add_argument("--pace", type=float, default=2.0, help="Seconds between rounds")
    args = parser.parse_args()

    land_w, land_h = (int(x) for x in args.land.split("x"))
    app = build_app(members=args.members, land_w=land_w, land_h=land_h, seed=args.seed)

    # Get references for the sim loop
    kernel = app.state.leviathan["kernel"]
    event_log = app.state.leviathan["event_log"]

    # Start simulation in background
    stop_event = threading.Event()
    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(kernel, event_log, args.pace, args.rounds, stop_event),
        daemon=True,
    )
    sim_thread.start()

    # Graceful shutdown
    def shutdown_handler(signum, frame):
        print("\nShutting down simulation...")
        stop_event.set()
        sim_thread.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest test_api.py -v`
Expected: All 21 tests PASS

**Step 5: Commit**

```bash
git add scripts/run_server.py test_api.py
git commit -m "feat(api): add run_server.py with background simulation loop"
```

---

### Task 9: Update requirements.txt and run full regression

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add the API dependencies to `requirements.txt`:
```
# API (Phase 1)
fastapi>=0.100.0
uvicorn>=0.20.0
httpx>=0.24.0
```

**Step 2: Run all API tests**

Run: `pytest test_api.py -v --tb=short`
Expected: All 21 tests PASS

**Step 3: Run all kernel tests**

Run: `pytest test_world_kernel.py -v --tb=short`
Expected: All 51 tests PASS

**Step 4: Run all existing tests**

Run: `pytest test_graph_system.py test_eval_metrics.py test_code_cleaning.py test_llm_utils.py test_prompting.py -v --tb=short`
Expected: All 34 existing tests PASS

**Step 5: Verify server starts (manual smoke test)**

Run: `timeout 5 python scripts/run_server.py --members 5 --land 10x10 --seed 42 --port 8111 --rounds 2 --pace 0.5 2>&1 || true`
Expected: Server starts, runs 2 rounds, output shows uvicorn startup

**Step 6: Verify API docs render**

Run: `python -c "from api.app import create_app; from kernel import WorldKernel, WorldConfig; import tempfile; k = WorldKernel(WorldConfig(5,(10,10),42), tempfile.mkdtemp()); app = create_app(k); print('OpenAPI routes:', [r.path for r in app.routes])"`
Expected: Prints route list including /v1/world, /health, etc.

**Step 7: Commit**

```bash
git add requirements.txt
git commit -m "chore: add API dependencies to requirements.txt"
```

---

### Task 10: Final verification and cleanup

**Files:** None new — verification only

**Step 1: Run the complete test suite**

Run: `pytest test_api.py test_world_kernel.py test_graph_system.py test_eval_metrics.py test_code_cleaning.py test_llm_utils.py test_prompting.py -v --tb=short`
Expected: All 106 tests PASS (21 API + 51 kernel + 34 existing)

**Step 2: Verify clean imports**

Run: `python -c "from api.app import create_app; from api.models import WorldInfo, EventEnvelope; from api.routes.world import router; print('OK')"`
Expected: Prints `OK`

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If any uncommitted changes:
git add -A
git commit -m "chore: finalize Phase 1 Read API"
```
