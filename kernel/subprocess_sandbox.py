"""SubprocessSandbox — runs agent code in a child process with resource limits."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from kernel.execution_sandbox import SandboxContext, SandboxResult

_ENGINE_PROXY_PATH = Path(__file__).parent / "engine_proxy.py"


class SubprocessSandbox:
    """Runs agent code in a child process with timeout enforcement."""

    def __init__(self, timeout: int = 5) -> None:
        self.timeout = timeout

    def _get_preexec_fn(self):
        """Return preexec_fn for subprocess resource limits (Unix only)."""
        try:
            import resource
        except ImportError:
            return None
        def _set_limits():
            # 5 seconds CPU time
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
            # 10 MB max file size
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            # 256 MB virtual memory — applied last as it can interfere with setup
            try:
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
            except (ValueError, OSError):
                pass  # Some platforms may need more for Python interpreter startup
            # Limit child processes
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            except (ValueError, OSError):
                pass  # macOS may not support low RLIMIT_NPROC
        return _set_limits

    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult:
        state = self._extract_state(context)
        return self._run_in_subprocess(code, state, context.member_index, "agent_action")

    def execute_mechanism_code(self, code: str, context: SandboxContext) -> SandboxResult:
        state = self._extract_state(context)
        return self._run_in_subprocess(code, state, context.member_index, "propose_modification")

    def _extract_state(self, context: SandboxContext) -> dict:
        engine = context.execution_engine
        members = []
        for m in engine.current_members:
            members.append({"id": m.id, "vitality": float(m.vitality), "cargo": float(m.cargo), "land_num": int(m.land_num)})
        return {"members": members, "land": {"shape": list(engine.land.shape)}}

    def _run_in_subprocess(self, code: str, state: dict, member_index: int, entry_point: str) -> SandboxResult:
        wrapper = self._build_wrapper(code, state, member_index, entry_point)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            script_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=self.timeout,
                preexec_fn=self._get_preexec_fn(),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(success=False, error=f"Timed out after {self.timeout}s")
        finally:
            Path(script_path).unlink(missing_ok=True)

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return SandboxResult(success=False, error=stderr or f"Process exited with code {proc.returncode}", traceback_str=stderr)

        try:
            output = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError) as e:
            return SandboxResult(success=False, error=f"Failed to parse subprocess output: {e}")

        if output.get("error"):
            return SandboxResult(success=False, error=output["error"], traceback_str=output.get("traceback"))

        return SandboxResult(success=True, intended_actions=output.get("actions", []))

    def _build_wrapper(self, code: str, state: dict, member_index: int, entry_point: str) -> str:
        state_json = json.dumps(state)
        repo_root = str(_ENGINE_PROXY_PATH.parent.parent)

        lines = [
            "import json",
            "import sys",
            "import traceback",
            "",
            f"sys.path.insert(0, {repr(repo_root)})",
            "from kernel.engine_proxy import EngineProxy",
            "",
            f"_AGENT_CODE = {repr(code)}",
            f"_STATE = json.loads({repr(state_json)})",
            "",
            "def main():",
            "    proxy = EngineProxy(_STATE)",
            "    try:",
            "        compiled = compile(_AGENT_CODE, '<agent>', 'exec')",
            "        ns = {}",
            "        import builtins as _builtins",
            "        _BLOCKED = frozenset({'open', 'exec', 'eval', '__import__', 'compile',",
            "                    'globals', 'locals', 'breakpoint', 'exit', 'quit',",
            "                    'input', 'help', 'memoryview'})",
            "        _safe = {k: getattr(_builtins, k) for k in dir(_builtins)",
            "                 if not k.startswith('_') and k not in _BLOCKED}",
            "        _safe['__name__'] = '__main__'",
            "        _safe['__builtins__'] = _safe",
            "        _safe['print'] = lambda *a, **kw: None",
            "        ns['__builtins__'] = _safe",
            "        _run = getattr(_builtins, 'exec')",
            "        _run(compiled, ns)",
            "    except SyntaxError as e:",
            "        print(json.dumps({'error': f'SyntaxError: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            "    except Exception as e:",
            "        print(json.dumps({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            f"    fn = ns.get({repr(entry_point)})",
            "    if fn is None or not callable(fn):",
            f"        print(json.dumps({{'error': \"Code did not define callable '{entry_point}'\"}}))",
            "        return",
            "    try:",
        ]
        if entry_point == "agent_action":
            lines.append(f"        fn(proxy, {member_index})")
        else:
            lines.append("        fn(proxy)")
        lines.extend([
            "    except Exception as e:",
            "        print(json.dumps({'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}))",
            "        return",
            "    print(json.dumps({'actions': proxy.actions}))",
            "",
            "main()",
        ])
        return "\n".join(lines) + "\n"
