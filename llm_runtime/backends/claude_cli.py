"""Claude CLI backend."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from typing import Any

from llm_runtime.types import GenerationRequest, GenerationResponse


def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _build_env(strip_env: list[str]) -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in set(strip_env)}


class ClaudeCLIBackend:
    """Shells out to `claude -p`."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        model = request.model or self.settings.get("model")
        if not model:
            raise ValueError("Claude backend requires a model")

        timeout = int(request.timeout or self.settings.get("timeout", 300))
        binary = self.settings.get("command", "claude")
        output_format = self.settings.get("output_format", "json")
        no_session_persistence = bool(self.settings.get("no_session_persistence", True))
        env = _build_env(list(self.settings.get("strip_env", [])))

        cmd = [binary, "-p", "--model", model]
        if output_format:
            cmd.extend(["--output-format", output_format])
        if no_session_persistence:
            cmd.append("--no-session-persistence")

        start = time.time()
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=request.prompt.encode()),
                timeout=timeout,
            )
            elapsed = time.time() - start

            if proc.returncode != 0:
                err_msg = (stderr or b"").decode()[:300]
                if not err_msg.strip():
                    err_msg = (stdout or b"").decode()[:300]
                return GenerationResponse(
                    text=None,
                    elapsed=elapsed,
                    info={"status": "error", "error": err_msg or f"exit code {proc.returncode}"},
                )

            payload = json.loads(stdout.decode())
            usage = payload.get("usage", {})
            info = {"status": "success", **usage}
            return GenerationResponse(
                text=payload.get("result", ""),
                elapsed=elapsed,
                info=info,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            if proc is not None:
                _kill_process_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            return GenerationResponse(
                text=None,
                elapsed=elapsed,
                info={"status": "timeout", "error": "TIMEOUT"},
            )
        except Exception as exc:
            elapsed = time.time() - start
            if proc is not None and proc.returncode is None:
                _kill_process_tree(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
            return GenerationResponse(
                text=None,
                elapsed=elapsed,
                info={"status": "error", "error": str(exc)},
            )
