"""Codex CLI backend."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
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


class CodexCLIBackend:
    """Shells out to `codex exec`."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = dict(settings)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        model = request.model or self.settings.get("model")
        if not model:
            raise ValueError("Codex backend requires a model")

        reasoning = request.reasoning or self.settings.get("reasoning", "xhigh")
        timeout = int(request.timeout or self.settings.get("timeout", 300))
        binary = self.settings.get("command", "codex")
        full_auto = bool(self.settings.get("full_auto", True))
        ephemeral = bool(self.settings.get("ephemeral", True))
        env = _build_env(list(self.settings.get("strip_env", [])))

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            prefix="codex_out_",
        )
        outfile = tmp.name
        tmp.close()

        cmd = [binary, "exec", "-m", model]
        if reasoning:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
        if full_auto:
            cmd.append("--full-auto")
        if ephemeral:
            cmd.append("--ephemeral")
        cmd.extend(["-o", outfile, "-"])

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
                err_msg = (stderr or b"").decode()[:500]
                if not err_msg.strip():
                    err_msg = (stdout or b"").decode()[:500]
                return GenerationResponse(
                    text=None,
                    elapsed=elapsed,
                    info={
                        "status": "error",
                        "error": err_msg or f"exit code {proc.returncode}",
                    },
                )

            try:
                with open(outfile, "r") as handle:
                    result_text = handle.read().strip()
            except (FileNotFoundError, OSError) as exc:
                result_text = (stdout or b"").decode().strip()
                if not result_text:
                    return GenerationResponse(
                        text=None,
                        elapsed=elapsed,
                        info={
                            "status": "error",
                            "error": f"output file missing: {exc}",
                        },
                    )

            return GenerationResponse(
                text=result_text or None,
                elapsed=elapsed,
                info={"status": "success"},
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
        finally:
            try:
                os.unlink(outfile)
            except OSError:
                pass
