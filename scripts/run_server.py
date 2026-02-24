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
from typing import List

from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.app import create_app
from api.models import EventEnvelope
from kernel.schemas import WorldConfig
from kernel.world_kernel import WorldKernel

logger = logging.getLogger(__name__)


def build_app(members: int, land_w: int, land_h: int, seed: int) -> FastAPI:
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
    return create_app(kernel)


def _simulation_loop(
    kernel: WorldKernel,
    event_log: List[EventEnvelope],
    pace: float,
    max_rounds: int,
    stop_event: threading.Event,
) -> None:
    """Background thread that advances the simulation.

    Each iteration runs one full round (begin + settle), appends an
    EventEnvelope to the shared event log, then sleeps for *pace*
    seconds.  The loop exits cleanly when *stop_event* is set or
    *max_rounds* is reached (0 means unlimited).
    """
    rounds_completed = 0
    while not stop_event.is_set():
        kernel.begin_round()
        receipt = kernel.settle_round(seed=kernel.round_id)
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

        # Sleep interruptibly so SIGINT/SIGTERM can stop us quickly
        stop_event.wait(timeout=pace)


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
    args = parser.parse_args()

    land_w, land_h = args.land
    app = build_app(members=args.members, land_w=land_w, land_h=land_h, seed=args.seed)

    kernel = app.state.leviathan["kernel"]
    event_log = app.state.leviathan["event_log"]
    stop_event = threading.Event()

    sim_thread = threading.Thread(
        target=_simulation_loop,
        args=(kernel, event_log, args.pace, args.rounds, stop_event),
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
