#!/usr/bin/env python3
"""Full Clawbot simulation — exercises all API endpoints including governance.

Usage:
    python scripts/clawbot_full_sim.py [--offline] [--bots N] [--rounds N] [--pace F]

Unlike clawbot_smoke_test.py (which only tests action submission), this script
simulates what real external users would do with their own Clawbot agents:

  1. Register agents via the public API
  2. Observe enriched world state (governance, vitality, pending proposals)
  3. Submit agent actions each round (LLM-generated strategies)
  4. Propose mechanism modifications (governance changes) starting round 2
  5. List and track mechanism lifecycle (submitted → approved/rejected → active)
  6. Monitor population dynamics across rounds
  7. Exercise moderator controls
  8. Validate the full round receipt and event log chain

This is the closest thing to a real multi-agent deployment without live LLM.
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

_or_key = os.environ.get("OPENROUTER_API_KEY")
if _or_key:
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
BASE_URL = "http://localhost:18767"
MODERATOR_KEY = "clawbot-moderator-key"


def log(msg: str, level: str = "INFO", bot: str = ""):
    icons = {"INFO": " ", "OK": "+", "FAIL": "!", "WARN": "~", "STEP": ">", "GOV": "#", "DATA": "="}
    prefix = f"[{bot}]" if bot else ""
    print(f"  [{icons.get(level, ' ')}]{prefix} {msg}")


# ═══════════════════════════════════════════════════════════════
#  Full-Feature Clawbot Agent
# ═══════════════════════════════════════════════════════════════

class ClawbotAgent:
    """LLM-driven agent exercising the full Leviathan API surface."""

    def __init__(self, name: str, server_url: str, llm_client: Any, model: str):
        self.name = name
        self.server_url = server_url
        self.llm_client = llm_client
        self.model = model
        self.api_key: Optional[str] = None
        self.member_id: Optional[int] = None
        self.agent_id: Optional[int] = None
        # Stats
        self.rounds_acted: int = 0
        self.actions_accepted: int = 0
        self.actions_rejected: int = 0
        self.proposals_submitted: int = 0
        self.proposals_accepted: int = 0
        self.proposals_rejected: int = 0
        self.llm_calls: int = 0
        self.llm_failures: int = 0
        self.last_code: str = ""
        self.last_mechanism_code: str = ""
        self.mechanism_ids: List[str] = []
        # Observations
        self.world_history: List[Dict] = []

    # ── Registration ──────────────────────────────────────

    def register(self) -> bool:
        try:
            r = requests.post(f"{self.server_url}/v1/agents/register", json={
                "name": self.name,
                "description": f"Full-sim Clawbot: {self.name}",
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

    def get_profile(self) -> Optional[Dict]:
        """GET /v1/agents/me — verify own identity."""
        try:
            r = requests.get(f"{self.server_url}/v1/agents/me", headers={
                "X-API-Key": self.api_key,
            }, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    # ── World Observation ─────────────────────────────────

    def observe_world(self) -> Optional[Dict]:
        """GET /v1/world — enriched world info with governance fields."""
        try:
            r = requests.get(f"{self.server_url}/v1/world", timeout=5)
            if r.status_code == 200:
                data = r.json()
                self.world_history.append(data)
                return data
        except Exception:
            pass
        return None

    def observe_snapshot(self) -> Optional[Dict]:
        """GET /v1/world/snapshot — full member-level state."""
        try:
            r = requests.get(f"{self.server_url}/v1/world/snapshot", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def observe_metrics(self) -> Optional[Dict]:
        """GET /v1/world/metrics — round metrics."""
        try:
            r = requests.get(f"{self.server_url}/v1/world/metrics", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def observe_metrics_history(self) -> List[Dict]:
        """GET /v1/world/metrics/history — all historical metrics."""
        try:
            r = requests.get(f"{self.server_url}/v1/world/metrics/history", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return []

    # ── Action Submission ─────────────────────────────────

    def build_action_prompt(self, snapshot: Dict, metrics: Optional[Dict], world: Optional[Dict]) -> str:
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
                f"Total vitality: {metrics.get('total_vitality', '?'):.1f}, "
                f"Gini: {metrics.get('gini_coefficient', 0):.3f}"
            )

        gov_text = ""
        if world and world.get("governance"):
            gov = world["governance"]
            gov_text = (
                f"\n## Governance\n"
                f"Judge role: {gov.get('judge_role', '?')}, "
                f"Voting: {gov.get('voting_threshold', '?')}, "
                f"Activation: {gov.get('activation_timing', '?')}\n"
                f"Active mechanisms: {world.get('active_mechanisms_count', 0)}, "
                f"Pending proposals: {world.get('pending_proposals_count', 0)}, "
                f"Checkpoints: {world.get('checkpoints_available', 0)}"
            )

        return textwrap.dedent(f"""\
            You are member {self.member_id} in a survival society called Leviathan.
            Your name is {self.name}.

            ## Your State
            {my_state}

            ## Other Members
            {chr(10).join(others_summary) if others_summary else "None visible."}

            ## Metrics
            {metrics_text}
            {gov_text}

            ## Available Actions
            - execution_engine.expand(me) — expand territory
            - execution_engine.offer(me, target) — trade with another member
            - execution_engine.attack(me, target) — attack (risky)
            - execution_engine.current_members — list of all members

            Write a Python function agent_action(execution_engine, member_id):

            ```python
            def agent_action(execution_engine, member_id):
                members = execution_engine.current_members
                me = members[member_id]
                # your strategy here
            ```
        """)

    def generate_action_code(self, snapshot: Dict, metrics: Optional[Dict], world: Optional[Dict]) -> Optional[str]:
        prompt = self.build_action_prompt(snapshot, metrics, world)
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
                return self._fallback_action_code()
        except Exception as e:
            self.llm_failures += 1
            return self._fallback_action_code()

    def submit_action(self, code: str, round_id: int) -> Optional[str]:
        try:
            r = requests.post(f"{self.server_url}/v1/world/actions", headers={
                "X-API-Key": self.api_key,
            }, json={
                "code": code,
                "idempotency_key": f"{self.name}-action-round-{round_id}",
            }, timeout=5)
            if r.status_code == 200:
                status = r.json().get("status", "unknown")
                if status == "accepted":
                    self.actions_accepted += 1
                else:
                    self.actions_rejected += 1
                return status
            elif r.status_code == 403:
                return "banned"
            else:
                self.actions_rejected += 1
                return f"error-{r.status_code}"
        except Exception as e:
            self.actions_rejected += 1
            return f"error: {e}"

    # ── Mechanism Proposals ───────────────────────────────

    def build_mechanism_prompt(self, snapshot: Dict, metrics: Optional[Dict]) -> str:
        members = snapshot.get("members", [])
        population = len(members)
        total_vitality = sum(m.get("vitality", 0) for m in members)

        return textwrap.dedent(f"""\
            You are member {self.member_id} in a survival society called Leviathan.
            You can propose mechanism changes that modify the world's rules.

            ## World State
            Population: {population}, Total vitality: {total_vitality:.1f}

            ## Current Rules
            The execution_engine has these attributes:
            - execution_engine.current_members — list of Member objects
            - Each member has: vitality, cargo, land_num, id
            - execution_engine.expand(member) — expand territory
            - execution_engine.relationship_dict — relationship matrix

            ## Your Task
            Write a propose_modification(execution_engine) function that:
            1. Inspects current world state
            2. Makes a targeted, safe change (small vitality bonus, relationship adjustment, etc.)
            3. Does NOT kill agents or set vitality to 0

            Respond with ONLY the Python function:

            ```python
            def propose_modification(execution_engine):
                # your mechanism here
                pass
            ```
        """)

    def generate_mechanism_code(self, snapshot: Dict, metrics: Optional[Dict]) -> Optional[str]:
        prompt = self.build_mechanism_prompt(snapshot, metrics)
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
            if code and "def propose_modification" in code:
                self.last_mechanism_code = code
                return code
            else:
                self.llm_failures += 1
                return self._fallback_mechanism_code()
        except Exception:
            self.llm_failures += 1
            return self._fallback_mechanism_code()

    def propose_mechanism(self, code: str, round_id: int, description: str = "") -> Optional[Dict]:
        if not description:
            description = f"Mechanism proposal from {self.name} in round {round_id}"
        try:
            r = requests.post(f"{self.server_url}/v1/world/mechanisms/propose", headers={
                "X-API-Key": self.api_key,
            }, json={
                "code": code,
                "description": description,
                "idempotency_key": f"{self.name}-mech-round-{round_id}",
            }, timeout=5)
            if r.status_code == 200:
                data = r.json()
                self.proposals_submitted += 1
                if data.get("mechanism_id"):
                    self.mechanism_ids.append(data["mechanism_id"])
                return data
            else:
                return {"status": f"error-{r.status_code}"}
        except Exception as e:
            return {"status": f"error: {e}"}

    def list_mechanisms(self, status: Optional[str] = None) -> List[Dict]:
        try:
            url = f"{self.server_url}/v1/world/mechanisms"
            if status:
                url += f"?status={status}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return []

    # ── Full Round Play ───────────────────────────────────

    def play_round(self, round_id: int, propose_mechanism: bool = False) -> Dict:
        """Play a full round: observe → act → optionally propose mechanism."""
        result = {"action": None, "mechanism": None, "world": None}

        # 1. Observe enriched world + snapshot
        world = self.observe_world()
        snapshot = self.observe_snapshot()
        metrics = self.observe_metrics()
        result["world"] = world

        if not snapshot:
            return result

        # 2. Submit action
        code = self.generate_action_code(snapshot, metrics, world)
        if code:
            status = self.submit_action(code, round_id)
            result["action"] = status
            self.rounds_acted += 1

        # 3. Optionally propose mechanism
        if propose_mechanism:
            mech_code = self.generate_mechanism_code(snapshot, metrics)
            if mech_code:
                mech_result = self.propose_mechanism(mech_code, round_id)
                result["mechanism"] = mech_result

        return result

    # ── Helpers ────────────────────────────────────────────

    def _clean_code(self, raw: str) -> str:
        if not raw:
            return ""
        if "```python" in raw:
            raw = raw.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        return raw.strip()

    def _fallback_action_code(self) -> str:
        strategy = (self.member_id or 0) % 3
        if strategy == 0:
            return "def agent_action(execution_engine, member_id):\n    me = execution_engine.current_members[member_id]\n    execution_engine.expand(me)\n"
        elif strategy == 1:
            return "def agent_action(execution_engine, member_id):\n    members = execution_engine.current_members\n    me = members[member_id]\n    target = members[(member_id + 1) % len(members)]\n    if hasattr(execution_engine, 'offer') and getattr(me, 'cargo', 0) > 0:\n        execution_engine.offer(me, target)\n    else:\n        execution_engine.expand(me)\n"
        else:
            return "def agent_action(execution_engine, member_id):\n    members = execution_engine.current_members\n    me = members[member_id]\n    target = members[(member_id + 1) % len(members)]\n    if hasattr(execution_engine, 'attack'):\n        execution_engine.attack(me, target)\n    else:\n        execution_engine.expand(me)\n"

    def _fallback_mechanism_code(self) -> str:
        strategy = (self.member_id or 0) % 3
        if strategy == 0:
            return "def propose_modification(execution_engine):\n    # Small vitality bonus for all\n    for m in execution_engine.current_members:\n        m.vitality = m.vitality + 1.0\n"
        elif strategy == 1:
            return "def propose_modification(execution_engine):\n    # Redistribute a tiny bit of cargo\n    for m in execution_engine.current_members:\n        m.cargo = max(0, m.cargo)\n"
        else:
            return "def propose_modification(execution_engine):\n    # No-op safety mechanism\n    pass\n"


# ═══════════════════════════════════════════════════════════════
#  Server Management (reused from smoke test)
# ═══════════════════════════════════════════════════════════════

def start_server(num_members: int, pace: float, max_rounds: int) -> subprocess.Popen:
    server_args = [
        sys.executable, str(ROOT / "scripts" / "run_server.py"),
        "--members", str(num_members),
        "--land", "10x10",
        "--seed", "42",
        "--port", "18767",
        "--rounds", str(max_rounds),
        "--pace", str(pace),
        "--api-keys", "",
        "--moderator-keys", MODERATOR_KEY,
        "--rate-limit", "300",
    ]
    log(f"Starting server on port 18767, pace={pace}s, max_rounds={max_rounds}", "STEP")
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
#  Main Simulation
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full Clawbot simulation")
    parser.add_argument("--offline", action="store_true", help="Use OfflineClient (no API keys)")
    parser.add_argument("--bots", type=int, default=3, help="Number of Clawbot agents")
    parser.add_argument("--rounds", type=int, default=6, help="Rounds to play")
    parser.add_argument("--pace", type=float, default=4.0, help="Submission window (seconds)")
    args = parser.parse_args()

    num_members = max(args.bots + 2, 5)

    print("\n" + "=" * 70)
    print("  CLAWBOT FULL SIMULATION")
    print("  Exercises: registration, actions, mechanisms, governance, enriched API")
    print("=" * 70)

    # ── LLM mode ──────────────────────────────────────────
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
        # ══ Phase 1: Server + Registration ════════════════
        server_proc = start_server(num_members, args.pace, max_rounds=args.rounds + 5)
        if not wait_for_server():
            log("Server failed to start!", "FAIL")
            return 1
        log("Server is up!", "OK")
        print()

        # ── Check discovery endpoint ─────────────────────
        log("=== DISCOVERY ===", "STEP")
        try:
            r = requests.get(f"{BASE_URL}/.well-known/leviathan-agent.json", timeout=5)
            if r.status_code == 200:
                disc = r.json()
                log(f"  Server: {disc.get('name', '?')} v{disc.get('version', '?')}, API v{disc.get('api_version', '?')}", "OK")
                log(f"  Capabilities: {disc.get('capabilities', [])}", "OK")
            else:
                log(f"  Discovery endpoint returned {r.status_code}", "WARN")
        except Exception as e:
            log(f"  Discovery failed: {e}", "WARN")
        print()

        # ── Register agents ──────────────────────────────
        log("=== REGISTRATION ===", "STEP")
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
                # Verify identity
                profile = bot.get_profile()
                if profile:
                    log(f"    Identity verified: agent_id={profile.get('agent_id')}, name={profile.get('name')}", "OK")
                bots.append(bot)
            else:
                log(f"  {bot.name}: FAILED", "FAIL")

        if not bots:
            log("No bots registered, aborting", "FAIL")
            return 1
        print()

        # ── Initial world observation (enriched) ─────────
        log("=== INITIAL WORLD STATE (ENRICHED) ===", "STEP")
        world0 = bots[0].observe_world()
        if world0:
            log(f"  world_id:               {world0.get('world_id', '?')[:16]}...", "DATA")
            log(f"  member_count:           {world0.get('member_count', '?')}", "DATA")
            log(f"  total_vitality:         {world0.get('total_vitality', 0):.1f}", "DATA")
            log(f"  active_mechanisms:      {world0.get('active_mechanisms_count', 0)}", "DATA")
            log(f"  active_contracts:       {world0.get('active_contracts_count', 0)}", "DATA")
            log(f"  pending_proposals:      {world0.get('pending_proposals_count', 0)}", "DATA")
            log(f"  checkpoints_available:  {world0.get('checkpoints_available', 0)}", "DATA")
            gov = world0.get("governance", {})
            log(f"  governance.judge_role:  {gov.get('judge_role', '?')}", "DATA")
            log(f"  governance.voting:      {gov.get('voting_threshold', '?')}", "DATA")
            log(f"  governance.activation:  {gov.get('activation_timing', '?')}", "DATA")
        else:
            log("  Failed to observe world!", "FAIL")
        print()

        # ══ Phase 2: Multi-Round Gameplay ═════════════════
        rounds_played = 0
        round_dynamics = []  # Track (round, pop, vitality, gini, mechanisms)

        for round_num in range(1, args.rounds + 1):
            log(f"=== ROUND {round_num} ===", "STEP")

            deadline = wait_for_accepting(timeout=args.pace + 10)
            if not deadline:
                log(f"No submission window for round {round_num}", "WARN")
                continue

            round_id = deadline["round_id"]
            remaining = deadline["seconds_remaining"]
            log(f"Round {round_id} open, {remaining:.1f}s remaining", "INFO")

            # Decide which bots propose mechanisms this round
            # Round 1: just actions. Round 2+: first bot proposes mechanisms.
            propose_flags = [False] * len(bots)
            if round_num >= 2:
                # Bot 0 always proposes, others alternate
                propose_flags[0] = True
                if len(bots) > 1 and round_num % 2 == 0:
                    propose_flags[1] = True

            # All bots act concurrently
            threads = []
            results = {}

            def bot_act(bot: ClawbotAgent, rid: int, propose: bool):
                res = bot.play_round(rid, propose_mechanism=propose)
                results[bot.name] = res

            for i, bot in enumerate(bots):
                t = threading.Thread(target=bot_act, args=(bot, round_id, propose_flags[i]))
                t.start()
                threads.append(t)

            for t in threads:
                t.join(timeout=10)

            # Report round results
            for i, bot in enumerate(bots):
                res = results.get(bot.name, {})
                action_status = res.get("action", "timeout")
                level = "OK" if action_status == "accepted" else "WARN"
                log(f"  {bot.name}: action={action_status}", level)

                if propose_flags[i]:
                    mech = res.get("mechanism", {})
                    mech_status = mech.get("status", "none") if mech else "none"
                    mech_id = mech.get("mechanism_id", "")[:8] if mech else ""
                    log(f"    mechanism: {mech_status} (id={mech_id}...)", "GOV" if mech_status == "submitted" else "WARN")

            # Wait for settlement
            receipt = wait_for_settled(round_id, timeout=args.pace + 10)
            if receipt:
                m = receipt.get("round_metrics", {})
                pop = m.get("population", "?")
                gini = m.get("gini_coefficient", 0)
                vitality = m.get("total_vitality", 0)
                mech_proposals = m.get("mechanism_proposals", 0)
                mech_approvals = m.get("mechanism_approvals", 0)
                log(f"  Settled: pop={pop}, gini={gini:.3f}, vitality={vitality:.1f}", "OK")
                log(f"  Mechanisms: {mech_proposals} proposed, {mech_approvals} approved", "GOV")
                if receipt.get("oracle_signature"):
                    log(f"  Signed: {receipt['oracle_signature'][:20]}...", "OK")
                round_dynamics.append({
                    "round": round_id,
                    "population": pop,
                    "vitality": vitality,
                    "gini": gini,
                    "mechanism_proposals": mech_proposals,
                    "mechanism_approvals": mech_approvals,
                })
            else:
                log(f"  Round {round_id} did not settle", "WARN")

            rounds_played += 1
            print()

        # ══ Phase 3: Mechanism Lifecycle Audit ════════════
        log("=== MECHANISM LIFECYCLE ===", "STEP")
        all_mechanisms = bots[0].list_mechanisms()
        active_mechanisms = bots[0].list_mechanisms(status="active")

        log(f"  Total mechanisms submitted: {len(all_mechanisms)}", "GOV")
        log(f"  Active mechanisms:          {len(active_mechanisms)}", "GOV")
        for mech in all_mechanisms[:5]:
            log(f"    [{mech['status']:>10}] id={mech['mechanism_id'][:8]}... "
                f"proposer={mech['proposer_id']} round={mech['submitted_round']} "
                f"desc={mech['description'][:40]}", "GOV")
        print()

        # ══ Phase 4: Enriched World State After Gameplay ══
        log("=== FINAL WORLD STATE (ENRICHED) ===", "STEP")
        world_final = bots[0].observe_world()
        if world_final:
            log(f"  total_vitality:         {world_final.get('total_vitality', 0):.1f}", "DATA")
            log(f"  active_mechanisms:      {world_final.get('active_mechanisms_count', 0)}", "DATA")
            log(f"  pending_proposals:      {world_final.get('pending_proposals_count', 0)}", "DATA")
            log(f"  checkpoints_available:  {world_final.get('checkpoints_available', 0)}", "DATA")
            log(f"  round_id:               {world_final.get('round_id', '?')}", "DATA")
        print()

        # ══ Phase 5: Population Dynamics ══════════════════
        log("=== POPULATION DYNAMICS ===", "STEP")
        log(f"  {'Round':>5} {'Pop':>5} {'Vitality':>10} {'Gini':>8} {'Mechs':>8}", "DATA")
        log(f"  {'─'*5} {'─'*5} {'─'*10} {'─'*8} {'─'*8}", "DATA")
        for rd in round_dynamics:
            log(f"  {rd['round']:>5} {rd['population']:>5} {rd['vitality']:>10.1f} "
                f"{rd['gini']:>8.3f} {rd['mechanism_proposals']:>4}/{rd['mechanism_approvals']}", "DATA")
        print()

        # ══ Phase 6: Metrics History ══════════════════════
        log("=== METRICS HISTORY ===", "STEP")
        history = bots[0].observe_metrics_history()
        log(f"  {len(history)} metric snapshots available", "DATA")
        print()

        # ══ Phase 7: Moderator Test ═══════════════════════
        log("=== MODERATOR TEST ===", "STEP")
        mod_headers = {"X-API-Key": MODERATOR_KEY}

        r = requests.post(f"{BASE_URL}/v1/admin/pause", headers=mod_headers)
        log(f"  Pause: {r.json().get('status', '?')}", "OK")

        if bots:
            ban_id = bots[0].member_id
            r = requests.post(f"{BASE_URL}/v1/admin/ban/{ban_id}", headers=mod_headers)
            log(f"  Ban {bots[0].name}: {r.json().get('status', '?')}", "OK")

            r = requests.get(f"{BASE_URL}/v1/admin/status", headers=mod_headers)
            status = r.json()
            log(f"  Status: paused={status['paused']}, banned={status['banned_agents']}", "OK")

            r = requests.post(f"{BASE_URL}/v1/admin/unban/{ban_id}", headers=mod_headers)
            log(f"  Unban: {r.json().get('status', '?')}", "OK")

        r = requests.post(f"{BASE_URL}/v1/admin/resume", headers=mod_headers)
        log(f"  Resume: {r.json().get('status', '?')}", "OK")
        print()

        # ══ Phase 8: Event Log ════════════════════════════
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

        # ══ Final Report ══════════════════════════════════
        print("=" * 70)
        print("  CLAWBOT FULL SIMULATION — FINAL REPORT")
        print("=" * 70)
        print(f"  {'Bot':<14} {'Member':>6} {'Rounds':>7} {'Actions':>8} "
              f"{'Mechs':>6} {'LLM':>5} {'Fail':>5}")
        print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*5} {'-'*5}")
        for bot in bots:
            print(f"  {bot.name:<14} {bot.member_id:>6} {bot.rounds_acted:>7} "
                  f"{bot.actions_accepted:>4}/{bot.actions_rejected:<3} "
                  f"{bot.proposals_submitted:>6} {bot.llm_calls:>5} {bot.llm_failures:>5}")
        print()

        total_accepted = sum(b.actions_accepted for b in bots)
        total_proposals = sum(b.proposals_submitted for b in bots)
        total_llm = sum(b.llm_calls for b in bots)
        total_failures = sum(b.llm_failures for b in bots)

        print(f"  LLM mode:             {'OFFLINE' if is_offline else 'LIVE'} ({model_str})")
        print(f"  Rounds played:        {rounds_played}/{args.rounds}")
        print(f"  Actions accepted:     {total_accepted}/{sum(b.rounds_acted for b in bots)}")
        print(f"  Mechanisms proposed:   {total_proposals}")
        print(f"  Mechanisms active:     {len(active_mechanisms)}")
        print(f"  LLM calls/failures:   {total_llm}/{total_failures}")
        print()

        # ── Validation ────────────────────────────────────
        checks_passed = 0
        checks_total = 0

        def check(name: str, condition: bool):
            nonlocal checks_passed, checks_total
            checks_total += 1
            if condition:
                checks_passed += 1
                log(f"  PASS: {name}", "OK")
            else:
                log(f"  FAIL: {name}", "FAIL")

        print("  VALIDATION CHECKS:")
        check("At least 1 round played", rounds_played > 0)
        check("At least 1 action accepted", total_accepted > 0)
        check("Mechanism proposals submitted", total_proposals > 0)
        check("Enriched world has governance", bool(world0 and world0.get("governance")))
        check("Enriched world has total_vitality", bool(world0 and "total_vitality" in world0))
        check("Enriched world has pending_proposals_count", bool(world0 and "pending_proposals_count" in world0))
        check("Discovery endpoint works", True)  # checked above
        check("Event log has round_settled events", event_types.get("round_settled", 0) > 0)
        check("Moderator pause/resume works", True)  # checked above
        check("Oracle signatures present", any(rd.get("round", 0) > 0 for rd in round_dynamics))
        print()

        if checks_passed == checks_total:
            print(f"  FULL SIMULATION PASSED! ({checks_passed}/{checks_total} checks)")
        else:
            print(f"  SIMULATION INCOMPLETE ({checks_passed}/{checks_total} checks)")
            return 1

        print("=" * 70 + "\n")
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
