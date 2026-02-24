#!/usr/bin/env python3
"""Run the Leviathan Read API server with a background simulation loop."""

import argparse
import dataclasses
import logging
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Set

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import create_app
from api.models import EventEnvelope
from kernel.schemas import WorldConfig
from kernel.world_kernel import WorldKernel

logger = logging.getLogger(__name__)


def build_app(
    members: int,
    land_w: int,
    land_h: int,
    seed: int,
    api_keys: Optional[Set[str]] = None,
    rate_limit: int = 60,
) -> FastAPI:
    """Create a WorldKernel and return a fully configured FastAPI app.

    This is the testable entry point -- no threads, no uvicorn,
    just the wired-up application ready for TestClient or production.
    """
    save_path = tempfile.mkdtemp(prefix="leviathan_")
    config = WorldConfig(
        init_member_number=members,
        land_shape=(land_w, land_h),
        random_seed=seed,
    )
    kernel = WorldKernel(config, save_path=save_path)
    return create_app(kernel, api_keys=api_keys, rate_limit=rate_limit)


def _simulation_loop(
    kernel: WorldKernel,
    event_log: List[EventEnvelope],
    round_state,
    pace: float,
    max_rounds: int,
    stop_event: threading.Event,
) -> None:
    """Background thread that advances the simulation.

    Each iteration runs one full round with a submission window:
      begin_round -> open_submissions -> sleep(pace) ->
      close_submissions -> execute_actions -> settle_round -> append_event

    The loop exits cleanly when *stop_event* is set or
    *max_rounds* is reached (0 means unlimited).
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

        # Execute pending actions through SubprocessSandbox
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


def _parse_land(value: str):
    """Parse a 'WxH' string into (width, height)."""
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("land must be like '20x20' or '20,20'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    """CLI entry point: parse args, start simulation thread, run uvicorn."""
    parser = argparse.ArgumentParser(
        description="Run the Leviathan Read API server.",
    )
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--land", type=_parse_land, default="20x20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=0, help="0 = infinite")
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument(
        "--api-keys",
        type=str,
        default="",
        help="Comma-separated API keys (empty = open access)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Max requests per minute per client IP",
    )
    args = parser.parse_args()

    land_w, land_h = args.land
    api_keys: Optional[Set[str]] = None
    if args.api_keys:
        api_keys = {k.strip() for k in args.api_keys.split(",") if k.strip()}

    app = build_app(
        members=args.members,
        land_w=land_w,
        land_h=land_h,
        seed=args.seed,
        api_keys=api_keys,
        rate_limit=args.rate_limit,
    )

    kernel = app.state.leviathan["kernel"]
    event_log = app.state.leviathan["event_log"]
    round_state = app.state.leviathan["round_state"]
    stop_event = threading.Event()

    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(kernel, event_log, round_state, args.pace, args.rounds, stop_event),
        daemon=True,
        name="sim-loop",
    )

    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    sim_thread.start()

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)

    # After uvicorn exits, ensure the sim thread stops
    stop_event.set()
    sim_thread.join(timeout=5)


if __name__ == "__main__":
    main()
