#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
STATE_DIR = Path.home() / ".clawdbot"
STATE_FILE = STATE_DIR / "codex_loop_inbox.state.json"

LOG = ROOT / "codex_24h.log"
PIDFILE = ROOT / "codex_24h.pid"
STOPFILE = ROOT / "codex_24h.stop"
RESUME = ROOT / "codex_resume.md"
OBJECTIVE_FILE = Path(os.environ.get("OBJECTIVE_FILE", str(ROOT / "codex_objective.txt")))

CLAW = ROOT / "scripts" / "clawdbot_env.sh"

DEFAULT_CHANNELS = "whatsapp,telegram"

MAX_RESUME_CHARS = 2800


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def run_cmd(args: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def read_messages(channel: str, target: str, after_id: Optional[str]) -> List[Dict[str, Any]]:
    args = [str(CLAW), "message", "read", "--channel", channel, "--target", target, "--json", "--limit", "20"]
    if after_id:
        args += ["--after", str(after_id)]
    code, out, _ = run_cmd(args)
    if code != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    for key in ("messages", "data", "items", "results"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def extract_text(msg: Dict[str, Any]) -> str:
    for key in ("text", "body", "message", "content"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_id(msg: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "messageId", "msgId", "timestamp", "ts"):
        val = msg.get(key)
        if val is None:
            continue
        return str(val)
    return None


def current_iter() -> int:
    if not LOG.exists():
        return 0
    try:
        with LOG.open("r", errors="ignore") as f:
            lines = deque(f, maxlen=2000)
    except OSError:
        return 0
    iter_num = 0
    for line in lines:
        if line.startswith("=== Iteration"):
            m = re.search(r"Iteration\s+(\d+)", line)
            if m:
                iter_num = int(m.group(1))
    return iter_num


def loop_status() -> str:
    status = "stopped"
    pid = ""
    if PIDFILE.exists():
        pid = PIDFILE.read_text().strip()
        if pid:
            try:
                os.kill(int(pid), 0)
                status = "running"
            except OSError:
                status = "stopped"
    if STOPFILE.exists():
        status = "stopfile"

    obj = ""
    if OBJECTIVE_FILE.exists():
        obj = OBJECTIVE_FILE.read_text().strip()

    parts = [f"status: {status}", f"iter: {current_iter()}"]
    if pid:
        parts.append(f"pid: {pid}")
    if obj:
        parts.append(f"objective: {obj}")
    return "\n".join(parts)


def read_resume() -> str:
    if not RESUME.exists():
        return "(no codex_resume.md)"
    text = RESUME.read_text().strip()
    if len(text) > MAX_RESUME_CHARS:
        text = text[:MAX_RESUME_CHARS] + "..."
    return text or "(codex_resume.md empty)"


def start_loop() -> str:
    if PIDFILE.exists():
        pid = PIDFILE.read_text().strip()
        if pid:
            try:
                os.kill(int(pid), 0)
                return f"already running (pid {pid})"
            except OSError:
                pass
    if STOPFILE.exists():
        STOPFILE.unlink()
    cmd = f"cd '{ROOT}' && nohup ./codex_24h_loop.sh >/dev/null 2>&1 &"
    subprocess.Popen(["bash", "-lc", cmd])
    return "started"


def stop_loop() -> str:
    STOPFILE.write_text("stop\n")
    return "stopfile created; loop will exit after current iteration"


def restart_loop() -> str:
    STOPFILE.write_text("stop\n")
    pid = PIDFILE.read_text().strip() if PIDFILE.exists() else ""
    if pid:
        for _ in range(60):
            try:
                os.kill(int(pid), 0)
            except OSError:
                break
            time.sleep(1)
    if STOPFILE.exists():
        STOPFILE.unlink()
    cmd = f"cd '{ROOT}' && nohup ./codex_24h_loop.sh >/dev/null 2>&1 &"
    subprocess.Popen(["bash", "-lc", cmd])
    return "restarted"


def set_objective(text: str) -> str:
    text = text.strip()
    if not text:
        if OBJECTIVE_FILE.exists():
            return f"current objective:\n{OBJECTIVE_FILE.read_text().strip()}"
        return "objective file not set"
    OBJECTIVE_FILE.write_text(text + "\n")
    return "objective updated (applies next iteration; no restart needed)"


def parse_command(text: str) -> Optional[Tuple[str, str]]:
    raw = text.strip()
    low = raw.lower()

    prefixes = ("/codex", "!codex", "codex")
    rest = None
    for p in prefixes:
        if low.startswith(p):
            rest = raw[len(p):].lstrip(" :")
            break

    if rest is None:
        if low.startswith("objective:"):
            return ("objective", raw.split(":", 1)[1].strip())
        if low.startswith("objective "):
            return ("objective", raw.split(" ", 1)[1].strip())
        return None

    if not rest:
        return ("help", "")

    low_rest = rest.lower()
    if low_rest.startswith("set objective"):
        payload = rest[len("set objective"):].strip(" :")
        return ("objective", payload)

    parts = rest.split(None, 1)
    cmd = parts[0].lower()
    payload = parts[1] if len(parts) > 1 else ""

    if cmd in ("objective", "obj", "goal"):
        return ("objective", payload)
    if cmd in ("status", "start", "stop", "restart", "help", "resume"):
        return (cmd, payload)
    return None


def send(channel: str, target: str, message: str) -> None:
    if not target:
        return
    run_cmd([str(CLAW), "message", "send", "--channel", channel, "--target", target, "--message", message])


def handle_command(cmd: str, payload: str) -> str:
    if cmd == "status":
        return loop_status()
    if cmd == "start":
        return start_loop()
    if cmd == "stop":
        return stop_loop()
    if cmd == "restart":
        return restart_loop()
    if cmd == "resume":
        return read_resume()
    if cmd == "objective":
        return set_objective(payload)
    return (
        "Commands:\n"
        "/codex status\n"
        "/codex objective <text>\n"
        "/codex start | stop | restart\n"
        "/codex resume"
    )


def main() -> int:
    load_env(ENV_FILE)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    channels = [c.strip() for c in os.environ.get("CONTROL_CHANNELS", DEFAULT_CHANNELS).split(",") if c.strip()]
    targets = {
        "whatsapp": os.environ.get("CONTROL_WHATSAPP_TARGET") or os.environ.get("WATCH_WHATSAPP_TARGET", ""),
        "telegram": os.environ.get("CONTROL_TELEGRAM_TARGET") or os.environ.get("WATCH_TELEGRAM_TARGET", ""),
    }

    state: Dict[str, Dict[str, str]] = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            state = {}

    updated = False

    for channel in channels:
        target = targets.get(channel, "")
        if not target:
            continue
        key = f"{channel}:{target}"
        after_id = state.get(key, {}).get("last_id")
        messages = read_messages(channel, target, after_id)
        if not messages:
            continue
        for msg in messages:
            text = extract_text(msg)
            if not text:
                continue
            parsed = parse_command(text)
            if not parsed:
                continue
            cmd, payload = parsed
            response = handle_command(cmd, payload)
            send(channel, target, response)
            updated = True
        last_id = extract_id(messages[-1])
        if last_id:
            state[key] = {"last_id": last_id}
            updated = True

    if updated:
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
