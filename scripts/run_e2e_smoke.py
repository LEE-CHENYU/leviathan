#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDEALOGY_TEXT = """
[Island Ideology]

Island is a place of abundant resources and opportunity. Agents can:
- Extract resources from land based on land quality
- Build businesses to transform resources into products
- Create contracts with other agents for trade and services
- Propose physical constraints and economic mechanisms
- Form supply chains through interconnected contracts

The economy is entirely agent-driven. Agents write code to:
- Define what resources exist and how they're extracted
- Create markets and pricing mechanisms
- Build production chains and businesses
- Establish labor markets and specialization

Success depends on creating realistic, mutually beneficial economic systems.
""".strip()

CONFIG_PATH = ROOT / "config" / "models.yaml"


def parse_land_shape(value: str) -> Tuple[int, int]:
    value = value.strip().lower().replace("x", ",")
    parts = [p for p in value.split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("land must be like '6x6' or '6,6'")
    return int(parts[0]), int(parts[1])


def _load_default_model_id() -> str:
    try:
        import yaml
        if not CONFIG_PATH.exists():
            return "gpt-5.2"
        config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        default = config.get("default", {}) if isinstance(config, dict) else {}
        return default.get("model_id") or "gpt-5.2"
    except Exception:
        return "gpt-5.2"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _offline_mode_enabled() -> bool:
    if _is_truthy(os.environ.get("E2E_OFFLINE")):
        return True
    if _is_truthy(os.environ.get("LLM_OFFLINE")):
        return True
    provider = os.environ.get("E2E_PROVIDER", "").strip().lower()
    return provider in {"offline", "mock", "stub"}


def _enable_offline_mode(reason: str) -> None:
    os.environ.setdefault("E2E_OFFLINE", "1")
    os.environ.setdefault("LLM_OFFLINE", "1")
    if reason:
        print(f"[e2e] Offline mode enabled: {reason}")


def _configure_e2e_provider() -> None:
    provider = os.environ.get("E2E_PROVIDER", "openrouter").strip().lower()
    e2e_model = os.environ.get("E2E_MODEL") or _load_default_model_id()
    if not os.environ.get("LLM_MAX_TOKENS"):
        os.environ["LLM_MAX_TOKENS"] = "1024"

    if _offline_mode_enabled():
        _enable_offline_mode("offline flag set")
        return

    if provider in {"offline", "mock", "stub"}:
        _enable_offline_mode(f"E2E_PROVIDER={provider}")
        return

    if provider == "openrouter":
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            if not os.environ.get("OPENAI_API_KEY"):
                _enable_offline_mode("OPENROUTER_API_KEY missing")
                return
            provider = "openai"
        else:
            os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            # Use OpenRouter key just for this process
            os.environ["OPENAI_API_KEY"] = openrouter_key
            os.environ.setdefault("MODEL_ROUTE_DEFAULT_PROVIDER", "openai")
            os.environ.setdefault("MODEL_ROUTE_DEFAULT_MODEL_ID", e2e_model)
            os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_PROVIDER", "openai")
            os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_MODEL_ID", e2e_model)
            return

    if not os.environ.get("OPENAI_API_KEY"):
        _enable_offline_mode("OPENAI_API_KEY missing")
        return
    os.environ.setdefault("MODEL_ROUTE_DEFAULT_PROVIDER", "openai")
    os.environ.setdefault("MODEL_ROUTE_DEFAULT_MODEL_ID", e2e_model or "gpt-5.2")
    os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_PROVIDER", "openai")
    os.environ.setdefault("MODEL_ROUTE_DEEPSEEK_MODEL_ID", e2e_model or "gpt-5.2")


async def run(rounds: int, members: int, land: Tuple[int, int], seed: int, output_dir: Path) -> dict:
    from MetaIsland.metaIsland import IslandExecution

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    exec = IslandExecution(members, land, str(run_dir), seed)
    IslandExecution._RECORD_PERIOD = 1
    exec.island_ideology = IDEALOGY_TEXT

    for _ in range(rounds):
        await exec.run_round_with_graph()

    round_record = exec.execution_history['rounds'][-1] if exec.execution_history.get('rounds') else {}
    round_metrics = round_record.get('round_metrics', {})
    round_end_metrics = round_record.get('round_end_metrics', {})

    exec_history_file = None
    history_dir = Path(exec.execution_history_path)
    if history_dir.exists():
        files = sorted(history_dir.glob("execution_history_*.json"), key=lambda p: p.stat().st_mtime)
        if files:
            exec_history_file = str(files[-1])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "rounds": rounds,
        "members": members,
        "land_shape": list(land),
        "seed": seed,
        "offline_mode": _offline_mode_enabled(),
        "round_number": round_record.get("round_number"),
        "round_metrics": round_metrics,
        "round_end_metrics": round_end_metrics,
        "contract_stats": round_record.get("contract_stats", {}),
        "physics_stats": round_record.get("physics_stats", {}),
        "execution_history_file": exec_history_file,
        "save_path": str(run_dir),
    }

    latest_path = output_dir / "latest_summary.json"
    stamped_path = output_dir / f"summary_{run_id}.json"
    latest_path.write_text(json.dumps(summary, indent=2) + "\n")
    stamped_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("E2E smoke run summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal end-to-end MetaIsland smoke test.")
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("E2E_ROUNDS", 1)))
    parser.add_argument("--members", type=int, default=int(os.environ.get("E2E_MEMBERS", 3)))
    parser.add_argument("--land", type=parse_land_shape, default=parse_land_shape(os.environ.get("E2E_LAND", "6x6")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("E2E_SEED", 2023)))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("E2E_OUTPUT_DIR", "execution_histories/e2e_smoke")))
    parser.add_argument("--offline", action="store_true", help="Use offline stub LLM responses.")
    args = parser.parse_args()

    load_dotenv()
    if args.offline:
        _enable_offline_mode("cli flag")
    _configure_e2e_provider()

    asyncio.run(run(args.rounds, args.members, args.land, args.seed, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
