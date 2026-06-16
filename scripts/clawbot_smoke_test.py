#!/usr/bin/env python3
"""Production smoke test with Clawbot LLM-driven agents.

Usage:
    python scripts/clawbot_smoke_test.py [--offline] [--bots N] [--rounds N]

This script:
  1. Starts the Leviathan API server
  2. Spawns N Clawbot agents that register via the API
  3. Each Clawbot observes the world via the API, generates agent_action code via LLM,
     and submits it during the submission window
  4. Runs for several rounds and reports results
  5. Tests moderator features (pause/resume/ban) mid-game

Clawbots use the same LLM infrastructure as MetaIsland (model_router + aisuite).
Pass --offline to use the OfflineClient (no API keys needed) for CI/testing.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env if present
_env_path = ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# When using OpenRouter as the default provider, ensure OPENAI_API_KEY points to
# the OpenRouter key. model_router normalizes "openrouter" → "openai" provider,
# but aisuite sends OPENAI_API_KEY as the Bearer token. If both keys exist in .env,
# the wrong key gets used for OpenRouter endpoints.
_or_key = os.environ.get("OPENROUTER_API_KEY")
if _or_key:
    # Peek at models.yaml to check if default provider is openrouter
    import yaml as _yaml
    _models_yaml = ROOT / "config" / "models.yaml"
    if _models_yaml.exists():
        _cfg = _yaml.safe_load(_models_yaml.read_text()) or {}
        if _cfg.get("default", {}).get("provider") == "openrouter":
            os.environ["OPENAI_API_KEY"] = _or_key
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from MetaIsland.model_router import model_router
from MetaIsland.llm_client import get_llm_client, OfflineClient
from MetaIsland.llm_utils import build_chat_kwargs

# ── Configuration ────────────────────────────────────────────
BASE_URL = "http://localhost:18766"
MODERATOR_KEY = "clawbot-moderator-key"


def log(msg: str, level: str = "INFO", bot: str = ""):
    icons = {"INFO": " ", "OK": "+", "FAIL": "!", "WARN": "~", "STEP": ">", "LLM": "*"}
    prefix = f"[{bot}]" if bot else ""
    print(f"  [{icons.get(level, ' ')}]{prefix} {msg}")


# ═══════════════════════════════════════════════════════════════
#  Clawbot Agent
# ═══════════════════════════════════════════════════════════════

class ClawbotAgent:
    """An LLM-driven agent that interacts with the Leviathan API."""

    def __init__(self, name: str, server_url: str, llm_client: Any, model: str):
        self.name = name
        self.server_url = server_url
        self.llm_client = llm_client
        self.model = model
        self.api_key: Optional[str] = None
        self.member_id: Optional[int] = None
        self.agent_id: Optional[int] = None
        self.rounds_acted: int = 0
        self.actions_accepted: int = 0
        self.actions_rejected: int = 0
        self.llm_calls: int = 0
        self.llm_failures: int = 0
        self.last_code: str = ""

    def register(self) -> bool:
        """Register with the Leviathan server."""
        try:
            r = requests.post(f"{self.server_url}/v1/agents/register", json={
                "name": self.name,
                "description": f"Clawbot agent: {self.name}",
            }, timeout=5)
            if r.status_code == 200:
                data = r.json()
                self.api_key = data["api_key"]
                self.member_id = data["member_id"]
                self.agent_id = data["agent_id"]
                return True
            elif r.status_code == 409:
                log(f"Registration full (409)", "WARN", self.name)
                return False
            else:
                log(f"Registration failed: {r.status_code}", "FAIL", self.name)
                return False
        except Exception as e:
            log(f"Registration error: {e}", "FAIL", self.name)
            return False

    def observe_world(self) -> Optional[Dict]:
        """Observe world state via API."""
        try:
            r = requests.get(f"{self.server_url}/v1/world/snapshot", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def observe_metrics(self) -> Optional[Dict]:
        """Get latest metrics via API."""
        try:
            r = requests.get(f"{self.server_url}/v1/world/metrics", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def build_prompt(self, snapshot: Dict, metrics: Optional[Dict]) -> str:
        """Build an LLM prompt from API-observed state.

        This is what a real Clawbot would do — it only has access to public API data.
        """
        members = snapshot.get("members", [])
        my_member = None
        others = []
        for m in members:
            if m.get("id") == self.member_id:
                my_member = m
            else:
                others.append(m)

        my_state = "unknown"
        if my_member:
            my_state = (
                f"vitality={my_member.get('vitality', '?')}, "
                f"cargo={my_member.get('cargo', '?')}, "
                f"land_num={my_member.get('land_num', '?')}"
            )

        others_summary = []
        for o in others[:4]:
            others_summary.append(
                f"  - member {o.get('id', '?')}: vitality={o.get('vitality', '?')}, "
                f"cargo={o.get('cargo', '?')}"
            )

        metrics_text = "No metrics available yet."
        if metrics:
            metrics_text = (
                f"Population: {metrics.get('population', '?')}, "
                f"Total vitality: {metrics.get('total_vitality', '?')}, "
                f"Gini coefficient: {metrics.get('gini_coefficient', '?'):.3f}, "
                f"Trade volume: {metrics.get('trade_volume', '?')}, "
                f"Conflicts: {metrics.get('conflict_count', '?')}"
            )

        prompt = textwrap.dedent(f"""\
            You are member {self.member_id} in a survival society simulation called Leviathan.
            Your name is {self.name}.

            ## Your Current State
            {my_state}

            ## Other Members
            {chr(10).join(others_summary) if others_summary else "No other members visible."}

            ## World Metrics
            {metrics_text}

            ## Available Actions
            The execution_engine provides these methods:
            - execution_engine.expand(me) — expand territory, costs energy but gains land
            - execution_engine.offer(me, target) — trade cargo with another member
            - execution_engine.attack(me, target) — attack another member (risky)
            - execution_engine.current_members — list of all members (index by member_id)

            ## Your Task
            Write a Python function called agent_action that takes (execution_engine, member_id)
            and performs one strategic action. Think about your survival, resource accumulation,
            and relationships with other members.

            Respond with ONLY the Python function, no explanation:

            ```python
            def agent_action(execution_engine, member_id):
                members = execution_engine.current_members
                me = members[member_id]
                # your strategy here
            ```
        """)
        return prompt

    def generate_action_code(self, snapshot: Dict, metrics: Optional[Dict]) -> Optional[str]:
        """Generate agent_action code using LLM."""
        prompt = self.build_prompt(snapshot, metrics)
        self.llm_calls += 1

        try:
            chat_kwargs = build_chat_kwargs()
            completion = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **chat_kwargs,
            )
            raw = completion.choices[0].message.content
            code = self._clean_code(raw)
            if code and "def agent_action" in code:
                self.last_code = code
                return code
            else:
                self.llm_failures += 1
                log(f"LLM returned invalid code (no agent_action)", "WARN", self.name)
                return self._fallback_code()
        except Exception as e:
            self.llm_failures += 1
            log(f"LLM error: {type(e).__name__}: {e}", "WARN", self.name)
            return self._fallback_code()

    def submit_action(self, code: str, round_id: int) -> Optional[str]:
        """Submit action code to the server."""
        try:
            r = requests.post(f"{self.server_url}/v1/world/actions", headers={
                "X-API-Key": self.api_key,
            }, json={
                "code": code,
                "idempotency_key": f"{self.name}-round-{round_id}",
            }, timeout=5)
            if r.status_code == 200:
                data = r.json()
                status = data.get("status", "unknown")
                if status == "accepted":
                    self.actions_accepted += 1
                else:
                    self.actions_rejected += 1
                return status
            elif r.status_code == 403:
                log(f"Banned! Cannot submit action", "WARN", self.name)
                return "banned"
            else:
                self.actions_rejected += 1
                return f"error-{r.status_code}"
        except Exception as e:
            self.actions_rejected += 1
            return f"error: {e}"

    def play_round(self, round_id: int) -> str:
        """Observe, think, act — one complete round."""
        snapshot = self.observe_world()
        if not snapshot:
            return "no-snapshot"

        metrics = self.observe_metrics()
        code = self.generate_action_code(snapshot, metrics)
        if not code:
            return "no-code"

        status = self.submit_action(code, round_id)
        self.rounds_acted += 1
        return status or "unknown"

    def _clean_code(self, raw: str) -> str:
        """Extract Python code from LLM response."""
        if not raw:
            return ""
        if "```python" in raw:
            raw = raw.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        return raw.strip()

    def _fallback_code(self) -> str:
        """Deterministic fallback when LLM fails."""
        strategy = (self.member_id or 0) % 3
        if strategy == 0:
            return textwrap.dedent("""\
                def agent_action(execution_engine, member_id):
                    me = execution_engine.current_members[member_id]
                    execution_engine.expand(me)
            """)
        elif strategy == 1:
            return textwrap.dedent("""\
                def agent_action(execution_engine, member_id):
                    members = execution_engine.current_members
                    me = members[member_id]
                    target = members[(member_id + 1) % len(members)]
                    if hasattr(execution_engine, 'offer') and getattr(me, 'cargo', 0) > 0:
                        execution_engine.offer(me, target)
                    else:
                        execution_engine.expand(me)
            """)
        else:
            return textwrap.dedent("""\
                def agent_action(execution_engine, member_id):
                    members = execution_engine.current_members
                    me = members[member_id]
                    target = members[(member_id + 1) % len(members)]
                    if hasattr(execution_engine, 'attack'):
                        execution_engine.attack(me, target)
                    else:
                        execution_engine.expand(me)
            """)

    def stats_dict(self) -> Dict:
        return {
            "name": self.name,
            "member_id": self.member_id,
            "rounds_acted": self.rounds_acted,
            "accepted": self.actions_accepted,
            "rejected": self.actions_rejected,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
        }


# ═══════════════════════════════════════════════════════════════
#  Server Management
# ═══════════════════════════════════════════════════════════════

def start_server(num_members: int, pace: float, max_rounds: int) -> subprocess.Popen:
    server_args = [
        sys.executable, str(ROOT / "scripts" / "run_server.py"),
        "--members", str(num_members),
        "--land", "10x10",
        "--seed", "42",
        "--port", "18766",
        "--rounds", str(max_rounds),
        "--pace", str(pace),
        "--api-keys", "",
        "--moderator-keys", MODERATOR_KEY,
        "--rate-limit", "300",
    ]
    log(f"Starting server: {' '.join(server_args[-14:])}", "STEP")
    proc = subprocess.Popen(
        server_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log(f"Server PID: {proc.pid}")
    return proc


def wait_for_server(timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.3)
    return False


def wait_for_accepting(timeout: float = 15.0) -> Optional[Dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/v1/world/rounds/current/deadline", timeout=2)
            data = r.json()
            if data.get("state") == "accepting" and data.get("seconds_remaining", 0) > 0.8:
                return data
        except Exception:
            pass
        time.sleep(0.2)
    return None


def wait_for_settled(round_id: int, timeout: float = 15.0) -> Optional[Dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/v1/world/rounds/{round_id}", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.3)
    return None


# ═══════════════════════════════════════════════════════════════
#  Main Test
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clawbot production smoke test")
    parser.add_argument("--offline", action="store_true", help="Use OfflineClient (no API keys)")
    parser.add_argument("--bots", type=int, default=3, help="Number of Clawbot agents")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds to play")
    parser.add_argument("--pace", type=float, default=4.0, help="Submission window (seconds)")
    args = parser.parse_args()

    num_members = max(args.bots + 2, 5)  # extra unregistered members

    print("\n" + "=" * 60)
    print("  CLAWBOT PRODUCTION SMOKE TEST")
    print("=" * 60)

    # ── Determine LLM mode ───────────────────────────────────
    if args.offline:
        os.environ["LLM_OFFLINE"] = "1"

    llm_client = get_llm_client()
    is_offline = isinstance(llm_client, OfflineClient)
    provider, model_id = model_router("default")
    model_str = f"{provider}:{model_id}"

    if is_offline:
        log("LLM mode: OFFLINE (deterministic responses)", "WARN")
    else:
        log(f"LLM mode: LIVE ({model_str})", "OK")

    log(f"Bots: {args.bots}, Rounds: {args.rounds}, Pace: {args.pace}s", "INFO")
    print()

    server_proc = None
    try:
        # ── Start server ─────────────────────────────────────
        server_proc = start_server(num_members, args.pace, max_rounds=args.rounds + 5)
        if not wait_for_server():
            log("Server failed to start!", "FAIL")
            return 1
        log("Server is up!", "OK")
        print()

        # ── Register Clawbots ────────────────────────────────
        log("Registering Clawbot agents...", "STEP")
        bots: List[ClawbotAgent] = []
        for i in range(args.bots):
            bot = ClawbotAgent(
                name=f"Clawbot-{i}",
                server_url=BASE_URL,
                llm_client=llm_client,
                model=model_str,
            )
            if bot.register():
                log(f"  {bot.name}: member_id={bot.member_id}, key={bot.api_key[:12]}...", "OK")
                bots.append(bot)
            else:
                log(f"  {bot.name}: registration failed", "FAIL")

        if not bots:
            log("No bots registered, aborting", "FAIL")
            return 1

        # ── Get initial world state ──────────────────────────
        world = requests.get(f"{BASE_URL}/v1/world").json()
        log(f"World: {world['world_id'][:16]}..., members={world['member_count']}, pubkey={world['world_public_key'][:16]}...", "OK")
        print()

        # ── Play rounds ──────────────────────────────────────
        rounds_played = 0
        for round_num in range(1, args.rounds + 1):
            log(f"=== ROUND {round_num} ===", "STEP")

            # Wait for submission window
            deadline = wait_for_accepting(timeout=args.pace + 10)
            if not deadline:
                log(f"No submission window for round {round_num}", "WARN")
                continue

            round_id = deadline["round_id"]
            remaining = deadline["seconds_remaining"]
            log(f"Round {round_id} open, {remaining:.1f}s remaining", "INFO")

            # All bots act concurrently
            threads = []
            results = {}

            def bot_act(bot: ClawbotAgent, rid: int):
                status = bot.play_round(rid)
                results[bot.name] = status

            for bot in bots:
                t = threading.Thread(target=bot_act, args=(bot, round_id))
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=10)

            # Report
            shown_code = False
            for bot in bots:
                status = results.get(bot.name, "timeout")
                level = "OK" if status == "accepted" else "WARN"
                code_preview = bot.last_code.split("\n")[1].strip()[:60] if bot.last_code and "\n" in bot.last_code else "n/a"
                log(f"  {bot.name}: {status} | code: {code_preview}", level)
                # Show full LLM-generated code for first successful bot per round
                if status == "accepted" and bot.last_code and not shown_code:
                    for cline in bot.last_code.strip().split("\n"):
                        log(f"    | {cline}", "LLM")
                    shown_code = True

            # Wait for settlement
            receipt = wait_for_settled(round_id, timeout=args.pace + 10)
            if receipt:
                m = receipt.get("round_metrics", {})
                log(f"  Settled: pop={m.get('population', '?')}, gini={m.get('gini_coefficient', 0):.3f}, vitality={m.get('total_vitality', '?')}", "OK")
                if receipt.get("oracle_signature"):
                    log(f"  Signed: sig={receipt['oracle_signature'][:20]}...", "OK")
            else:
                log(f"  Round {round_id} did not settle in time", "WARN")

            rounds_played += 1
            print()

        # ── Moderator test ───────────────────────────────────
        log("=== MODERATOR TEST ===", "STEP")
        mod_headers = {"X-API-Key": MODERATOR_KEY}

        # Pause
        r = requests.post(f"{BASE_URL}/v1/admin/pause", headers=mod_headers)
        log(f"Pause: {r.json()}", "OK")

        # Ban first bot
        if bots:
            ban_id = bots[0].member_id
            r = requests.post(f"{BASE_URL}/v1/admin/ban/{ban_id}", headers=mod_headers)
            log(f"Ban {bots[0].name} (member {ban_id}): {r.json()}", "OK")

            # Verify ban
            r = requests.get(f"{BASE_URL}/v1/admin/status", headers=mod_headers)
            status = r.json()
            log(f"Status: paused={status['paused']}, banned={status['banned_agents']}", "OK")

            # Unban
            r = requests.post(f"{BASE_URL}/v1/admin/unban/{ban_id}", headers=mod_headers)
            log(f"Unban: {r.json()}", "OK")

        # Resume
        r = requests.post(f"{BASE_URL}/v1/admin/resume", headers=mod_headers)
        log(f"Resume: {r.json()}", "OK")
        print()

        # ── Event log summary ────────────────────────────────
        log("=== EVENT LOG ===", "STEP")
        r = requests.get(f"{BASE_URL}/v1/world/events")
        events = r.json()
        event_types = {}
        for e in events:
            t = e["event_type"]
            event_types[t] = event_types.get(t, 0) + 1
        for t, count in sorted(event_types.items()):
            log(f"  {t}: {count}", "OK")
        print()

        # ── Final stats ──────────────────────────────────────
        print("=" * 60)
        print("  CLAWBOT STATS")
        print("=" * 60)
        print(f"  {'Bot':<14} {'Member':>6} {'Rounds':>7} {'Accept':>7} {'Reject':>7} {'LLM':>5} {'Fail':>5}")
        print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5}")
        for bot in bots:
            s = bot.stats_dict()
            print(f"  {s['name']:<14} {s['member_id']:>6} {s['rounds_acted']:>7} "
                  f"{s['accepted']:>7} {s['rejected']:>7} {s['llm_calls']:>5} {s['llm_failures']:>5}")
        print()

        total_accepted = sum(b.actions_accepted for b in bots)
        total_llm = sum(b.llm_calls for b in bots)
        total_failures = sum(b.llm_failures for b in bots)

        print(f"  LLM mode: {'OFFLINE' if is_offline else 'LIVE'} ({model_str})")
        print(f"  Rounds played: {rounds_played}/{args.rounds}")
        print(f"  Actions accepted: {total_accepted}/{total_llm}")
        print(f"  LLM failures: {total_failures}/{total_llm}")

        # Verdict
        print()
        if rounds_played > 0 and total_accepted > 0:
            print("  SMOKE TEST PASSED!")
        else:
            print("  SMOKE TEST FAILED (no actions accepted)")
            return 1

        print("=" * 60 + "\n")
        return 0

    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
        return 1
    except Exception as e:
        log(f"UNEXPECTED ERROR: {type(e).__name__}: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if server_proc and server_proc.poll() is None:
            log("Shutting down server...", "STEP")
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=10)
                log("Server stopped cleanly", "OK")
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
                log("Server killed", "WARN")


if __name__ == "__main__":
    sys.exit(main())
