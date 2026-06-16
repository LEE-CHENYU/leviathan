#!/usr/bin/env python3
"""Production smoke test: start server, register agents, submit actions, verify receipts.

Usage:
    python scripts/smoke_test.py

This script:
  1. Starts the Leviathan API server as a subprocess
  2. Registers multiple agents
  3. Waits for a submission window
  4. Submits agent actions (deterministic, no LLM needed)
  5. Submits a mechanism proposal
  6. Waits for the round to settle
  7. Verifies receipt integrity (constitution_hash, oracle_signature)
  8. Tests moderator features (pause, ban, unban, status)
  9. Checks metrics and event log
  10. Cleans up
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "scripts" / "run_server.py"

# ── Configuration ────────────────────────────────────────────
BASE_URL = "http://localhost:18765"  # non-standard port to avoid conflicts
API_KEY = "smoke-test-key-001"
MODERATOR_KEY = "smoke-moderator-key-001"
SERVER_ARGS = [
    sys.executable, str(SERVER_SCRIPT),
    "--members", "5",
    "--land", "10x10",
    "--seed", "42",
    "--port", "18765",
    "--rounds", "10",       # stop after 10 rounds
    "--pace", "3.0",        # 3s window — enough time for the test
    "--api-keys", "",       # open access (agents register and get keys)
    "--moderator-keys", MODERATOR_KEY,
    "--rate-limit", "300",
]

# ── Deterministic agent code (no LLM needed) ────────────────
EXPAND_CODE = """\
def agent_action(execution_engine, member_id):
    me = execution_engine.current_members[member_id]
    execution_engine.expand(me)
"""

OFFER_CODE = """\
def agent_action(execution_engine, member_id):
    members = execution_engine.current_members
    me = members[member_id]
    partner_idx = (member_id + 1) % len(members)
    partner = members[partner_idx]
    if hasattr(execution_engine, 'offer') and getattr(me, 'cargo', 0) > 0:
        execution_engine.offer(me, partner)
    else:
        execution_engine.expand(me)
"""

MECHANISM_CODE = """\
def mechanism_rule(execution_engine):
    # Simple no-op mechanism for testing
    pass
"""


class SmokeTestFailure(Exception):
    pass


def log(msg: str, level: str = "INFO"):
    icon = {"INFO": " ", "OK": "+", "FAIL": "!", "WARN": "~", "STEP": ">"}
    print(f"  [{icon.get(level, ' ')}] {msg}")


def assert_eq(actual, expected, label: str):
    if actual != expected:
        raise SmokeTestFailure(f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(value, collection, label: str):
    if value not in collection:
        raise SmokeTestFailure(f"{label}: {value!r} not in {collection!r}")


def wait_for_server(timeout: float = 30.0):
    """Poll /health until the server responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(0.3)
    raise SmokeTestFailure(f"Server did not start within {timeout}s")


def wait_for_accepting(timeout: float = 15.0) -> dict:
    """Poll deadline endpoint until state == 'accepting'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/v1/world/rounds/current/deadline", timeout=2)
            data = r.json()
            if data.get("state") == "accepting" and data.get("seconds_remaining", 0) > 0.5:
                return data
        except Exception:
            pass
        time.sleep(0.2)
    raise SmokeTestFailure(f"No accepting window within {timeout}s")


def wait_for_round(round_id: int, timeout: float = 20.0) -> dict:
    """Wait until a specific round has settled and return its receipt."""
    deadline_t = time.time() + timeout
    while time.time() < deadline_t:
        try:
            r = requests.get(f"{BASE_URL}/v1/world/rounds/{round_id}", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.3)
    raise SmokeTestFailure(f"Round {round_id} did not settle within {timeout}s")


# ═══════════════════════════════════════════════════════════════
#  Test Steps
# ═══════════════════════════════════════════════════════════════

def test_discovery(results: dict):
    """Step 1: Verify agent discovery endpoint."""
    log("Testing agent discovery endpoint...", "STEP")
    r = requests.get(f"{BASE_URL}/.well-known/leviathan-agent.json")
    assert_eq(r.status_code, 200, "discovery status")
    data = r.json()
    assert_in("endpoints", data, "discovery keys")
    log("Agent discovery OK", "OK")
    results["discovery"] = "PASS"


def test_world_info(results: dict):
    """Step 2: Verify world info."""
    log("Testing world info...", "STEP")
    r = requests.get(f"{BASE_URL}/v1/world")
    assert_eq(r.status_code, 200, "world status")
    data = r.json()
    assert_eq(data["member_count"], 5, "member_count")
    assert_in("world_public_key", data, "world info keys")
    assert data["world_public_key"] is not None, "world_public_key should not be None"
    log(f"World: {data['world_id']}, members={data['member_count']}, pubkey={data['world_public_key'][:16]}...", "OK")
    results["world_info"] = "PASS"
    return data


def test_register_agents(results: dict) -> list:
    """Step 3: Register multiple agents."""
    log("Registering agents...", "STEP")
    agents = []
    for i in range(3):
        r = requests.post(f"{BASE_URL}/v1/agents/register", json={
            "name": f"SmokeBot-{i}",
            "description": f"Smoke test agent #{i}",
        })
        assert_eq(r.status_code, 200, f"register agent {i}")
        data = r.json()
        assert_in("api_key", data, f"agent {i} response")
        assert_in("member_id", data, f"agent {i} response")
        agents.append(data)
        log(f"  Agent {i}: id={data['agent_id']}, member={data['member_id']}, key={data['api_key'][:12]}...", "OK")

    # Verify /agents/me
    r = requests.get(f"{BASE_URL}/v1/agents/me", headers={"X-API-Key": agents[0]["api_key"]})
    assert_eq(r.status_code, 200, "agents/me status")
    profile = r.json()
    assert_eq(profile["name"], "SmokeBot-0", "agent name")
    log(f"Agent profile verified: {profile['name']}", "OK")
    results["register_agents"] = "PASS"
    return agents


def test_submit_actions(agents: list, results: dict) -> int:
    """Step 4: Wait for accepting window and submit actions."""
    log("Waiting for submission window...", "STEP")
    deadline = wait_for_accepting()
    round_id = deadline["round_id"]
    log(f"Round {round_id} accepting, {deadline['seconds_remaining']:.1f}s remaining", "OK")

    # Agent 0: expand
    r = requests.post(f"{BASE_URL}/v1/world/actions", headers={
        "X-API-Key": agents[0]["api_key"],
    }, json={
        "code": EXPAND_CODE,
        "idempotency_key": f"smoke-r{round_id}-agent0",
    })
    assert_eq(r.status_code, 200, "submit action 0")
    data = r.json()
    assert_eq(data["status"], "accepted", "action 0 status")
    log(f"  Agent 0 action accepted (expand)", "OK")

    # Agent 1: offer/expand
    r = requests.post(f"{BASE_URL}/v1/world/actions", headers={
        "X-API-Key": agents[1]["api_key"],
    }, json={
        "code": OFFER_CODE,
        "idempotency_key": f"smoke-r{round_id}-agent1",
    })
    assert_eq(r.status_code, 200, "submit action 1")
    data = r.json()
    assert_eq(data["status"], "accepted", "action 1 status")
    log(f"  Agent 1 action accepted (offer/expand)", "OK")

    # Test idempotency: re-submit agent 0 — should succeed but not duplicate
    r = requests.post(f"{BASE_URL}/v1/world/actions", headers={
        "X-API-Key": agents[0]["api_key"],
    }, json={
        "code": EXPAND_CODE,
        "idempotency_key": f"smoke-r{round_id}-agent0",  # same key
    })
    assert_eq(r.status_code, 200, "idempotent re-submit")
    log(f"  Idempotent re-submit OK", "OK")

    results["submit_actions"] = "PASS"
    return round_id


def test_submit_mechanism(agents: list, round_id: int, results: dict):
    """Step 5: Submit a mechanism proposal."""
    log("Submitting mechanism proposal...", "STEP")
    r = requests.post(f"{BASE_URL}/v1/world/mechanisms/propose", headers={
        "X-API-Key": agents[2]["api_key"],
    }, json={
        "code": MECHANISM_CODE,
        "description": "Smoke test no-op mechanism",
        "idempotency_key": f"smoke-mech-r{round_id}-agent2",
    })
    assert_eq(r.status_code, 200, "propose mechanism")
    data = r.json()
    assert_in(data["status"], ["submitted", "rejected"], "mechanism status")
    log(f"  Mechanism proposal: status={data['status']}, id={data.get('mechanism_id', 'n/a')[:12]}", "OK")
    results["submit_mechanism"] = "PASS"


def test_round_settlement(round_id: int, results: dict) -> dict:
    """Step 6: Wait for round to settle and verify receipt."""
    log(f"Waiting for round {round_id} to settle...", "STEP")
    receipt = wait_for_round(round_id)
    log(f"Round {round_id} settled!", "OK")

    # Verify receipt fields
    assert_in("constitution_hash", receipt, "receipt keys")
    assert_in("oracle_signature", receipt, "receipt keys")
    assert_in("world_public_key", receipt, "receipt keys")
    assert receipt["constitution_hash"] is not None, "constitution_hash should exist"
    assert receipt["oracle_signature"] is not None, "oracle_signature should exist"
    assert receipt["world_public_key"] is not None, "world_public_key should exist"

    log(f"  constitution_hash: {receipt['constitution_hash'][:16]}...", "OK")
    log(f"  oracle_signature:  {receipt['oracle_signature'][:16]}...", "OK")
    log(f"  world_public_key:  {receipt['world_public_key'][:16]}...", "OK")

    # Verify round_metrics
    metrics = receipt.get("round_metrics")
    if metrics:
        log(f"  metrics: pop={metrics.get('population')}, gini={metrics.get('gini_coefficient', 'n/a'):.3f}, vitality={metrics.get('total_vitality', 'n/a')}", "OK")

    # Federation fields should be None
    assert receipt.get("origin_world_id") is None, "federation fields should be None"
    assert receipt.get("bridge_channel_id") is None, "federation fields should be None"
    log(f"  Federation fields correctly None", "OK")

    results["round_settlement"] = "PASS"
    return receipt


def test_metrics(results: dict):
    """Step 7: Verify metrics endpoints."""
    log("Testing metrics endpoints...", "STEP")

    r = requests.get(f"{BASE_URL}/v1/world/metrics")
    if r.status_code == 200:
        data = r.json()
        log(f"  Current metrics: population={data.get('population')}, gini={data.get('gini_coefficient', 'n/a')}", "OK")
    else:
        log(f"  Metrics not available yet (status={r.status_code})", "WARN")

    r = requests.get(f"{BASE_URL}/v1/world/metrics/history?limit=5")
    assert_eq(r.status_code, 200, "metrics history status")
    history = r.json()
    log(f"  Metrics history: {len(history)} entries", "OK")

    r = requests.get(f"{BASE_URL}/v1/world/judge/stats")
    assert_eq(r.status_code, 200, "judge stats status")
    stats = r.json()
    log(f"  Judge stats: total={stats.get('total_judgments', 0)}, approval_rate={stats.get('approval_rate', 0):.1%}", "OK")

    results["metrics"] = "PASS"


def test_events(round_id: int, results: dict):
    """Step 8: Verify event log."""
    log("Testing event log...", "STEP")
    r = requests.get(f"{BASE_URL}/v1/world/events")
    assert_eq(r.status_code, 200, "events status")
    events = r.json()
    assert len(events) > 0, "should have at least one event"
    log(f"  Total events: {len(events)}", "OK")

    # Check for round_settled event
    settled_events = [e for e in events if e["event_type"] == "round_settled"]
    assert len(settled_events) > 0, "should have round_settled events"
    log(f"  Round-settled events: {len(settled_events)}", "OK")

    # Test since_round filter
    r = requests.get(f"{BASE_URL}/v1/world/events?since_round={round_id}")
    assert_eq(r.status_code, 200, "events since_round status")
    filtered = r.json()
    log(f"  Events since round {round_id}: {len(filtered)}", "OK")

    results["events"] = "PASS"


def test_moderator(agents: list, results: dict):
    """Step 9: Test moderator features."""
    log("Testing moderator features...", "STEP")
    headers = {"X-API-Key": MODERATOR_KEY}

    # Status
    r = requests.get(f"{BASE_URL}/v1/admin/status", headers=headers)
    assert_eq(r.status_code, 200, "admin status")
    status = r.json()
    log(f"  Status: paused={status['paused']}, banned={status['banned_agents']}", "OK")

    # Pause
    r = requests.post(f"{BASE_URL}/v1/admin/pause", headers=headers)
    assert_eq(r.status_code, 200, "admin pause")
    log(f"  Paused simulation", "OK")

    # Verify paused
    r = requests.get(f"{BASE_URL}/v1/admin/status", headers=headers)
    assert r.json()["paused"] is True, "should be paused"

    # Resume
    r = requests.post(f"{BASE_URL}/v1/admin/resume", headers=headers)
    assert_eq(r.status_code, 200, "admin resume")
    log(f"  Resumed simulation", "OK")

    # Ban agent 2
    agent2_member = agents[2]["member_id"]
    r = requests.post(f"{BASE_URL}/v1/admin/ban/{agent2_member}", headers=headers)
    assert_eq(r.status_code, 200, "admin ban")
    log(f"  Banned agent member_id={agent2_member}", "OK")

    # Verify banned agent can't submit actions
    deadline = None
    try:
        deadline = wait_for_accepting(timeout=10)
    except SmokeTestFailure:
        log(f"  No accepting window (sim may be between rounds)", "WARN")

    if deadline:
        r = requests.post(f"{BASE_URL}/v1/world/actions", headers={
            "X-API-Key": agents[2]["api_key"],
        }, json={
            "code": EXPAND_CODE,
            "idempotency_key": f"smoke-banned-test",
        })
        assert_eq(r.status_code, 403, "banned agent should get 403")
        log(f"  Banned agent correctly rejected (403)", "OK")

    # Unban
    r = requests.post(f"{BASE_URL}/v1/admin/unban/{agent2_member}", headers=headers)
    assert_eq(r.status_code, 200, "admin unban")
    log(f"  Unbanned agent member_id={agent2_member}", "OK")

    # Non-moderator should be rejected
    r = requests.get(f"{BASE_URL}/v1/admin/status", headers={"X-API-Key": agents[0]["api_key"]})
    assert_eq(r.status_code, 403, "non-moderator rejected")
    log(f"  Non-moderator correctly rejected (403)", "OK")

    # Check admin events were emitted
    r = requests.get(f"{BASE_URL}/v1/world/events")
    events = r.json()
    admin_events = [e for e in events if e["event_type"].startswith("admin_")]
    log(f"  Admin events emitted: {len(admin_events)} ({', '.join(e['event_type'] for e in admin_events)})", "OK")

    results["moderator"] = "PASS"


def test_snapshot(results: dict):
    """Step 10: Verify full snapshot."""
    log("Testing full snapshot...", "STEP")
    r = requests.get(f"{BASE_URL}/v1/world/snapshot")
    assert_eq(r.status_code, 200, "snapshot status")
    snap = r.json()
    assert_in("members", snap, "snapshot keys")
    members = snap["members"]
    log(f"  Snapshot: {len(members)} members", "OK")
    for m in members[:3]:
        log(f"    member {m['id']}: vitality={m.get('vitality', '?')}, cargo={m.get('cargo', '?')}", "OK")
    results["snapshot"] = "PASS"


def test_mechanisms_list(results: dict):
    """Step 11: List mechanisms."""
    log("Testing mechanism listing...", "STEP")
    r = requests.get(f"{BASE_URL}/v1/world/mechanisms")
    assert_eq(r.status_code, 200, "mechanisms list status")
    mechs = r.json()
    log(f"  Total mechanisms: {len(mechs)}", "OK")
    for m in mechs[:3]:
        log(f"    {m['mechanism_id'][:12]}... status={m['status']}, proposer={m['proposer_id']}", "OK")
    results["mechanisms_list"] = "PASS"


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  LEVIATHAN PRODUCTION SMOKE TEST")
    print("=" * 60 + "\n")

    results = {}
    server_proc = None

    try:
        # ── Start server ─────────────────────────────────────
        log("Starting Leviathan server...", "STEP")
        log(f"  Command: {' '.join(SERVER_ARGS)}")
        server_proc = subprocess.Popen(
            SERVER_ARGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log(f"  Server PID: {server_proc.pid}")

        wait_for_server(timeout=30)
        log("Server is up!", "OK")

        # ── Run tests ────────────────────────────────────────
        test_discovery(results)
        world = test_world_info(results)
        agents = test_register_agents(results)
        round_id = test_submit_actions(agents, results)
        test_submit_mechanism(agents, round_id, results)
        receipt = test_round_settlement(round_id, results)
        test_metrics(results)
        test_events(round_id, results)
        test_snapshot(results)
        test_mechanisms_list(results)
        test_moderator(agents, results)

    except SmokeTestFailure as e:
        log(f"SMOKE TEST FAILED: {e}", "FAIL")
        results["_failure"] = str(e)
    except Exception as e:
        log(f"UNEXPECTED ERROR: {type(e).__name__}: {e}", "FAIL")
        results["_failure"] = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
    finally:
        # ── Shutdown server ──────────────────────────────────
        if server_proc and server_proc.poll() is None:
            log("Shutting down server...", "STEP")
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
                log("Server stopped cleanly", "OK")
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
                log("Server killed (timeout)", "WARN")

        # Print server output on failure
        if "_failure" in results and server_proc:
            try:
                stdout = server_proc.stdout.read().decode("utf-8", errors="replace")
                if stdout.strip():
                    print("\n--- Server output (last 50 lines) ---")
                    for line in stdout.strip().split("\n")[-50:]:
                        print(f"    {line}")
                    print("--- End server output ---\n")
            except Exception:
                pass

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for k, v in results.items():
        if k.startswith("_"):
            continue
        status = "PASS" if v == "PASS" else "FAIL"
        icon = "+" if status == "PASS" else "!"
        print(f"  [{icon}] {k}: {status}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed")

    if "_failure" in results:
        print(f"\n  FAILURE: {results['_failure']}")
        print("\n" + "=" * 60)
        return 1

    print("\n  ALL SMOKE TESTS PASSED!")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
