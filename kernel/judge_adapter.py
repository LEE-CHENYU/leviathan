"""JudgeAdapter — wraps the MetaIsland Judge in a subprocess for safe evaluation."""

import json
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class JudgmentResult:
    """Result of evaluating a proposal."""
    approved: bool
    reason: str
    latency_ms: float
    error: Optional[str] = None


class DummyJudge:
    """Always approves. For testing and fast iteration."""
    def evaluate(self, code: str, proposer_id: int, proposal_type: str, context: Optional[dict] = None) -> JudgmentResult:
        return JudgmentResult(approved=True, reason="dummy-approved", latency_ms=0.0)


class JudgeAdapter:
    """Runs the MetaIsland Judge in a subprocess with timeout.
    On timeout or crash, defaults to reject (fail-closed)."""

    def __init__(self, timeout: float = 30.0, use_dummy: bool = False) -> None:
        self.timeout = timeout
        self.use_dummy = use_dummy
        self._dummy = DummyJudge() if use_dummy else None

    def evaluate(self, code: str, proposer_id: int, proposal_type: str, context: Optional[dict] = None) -> JudgmentResult:
        if self._dummy:
            return self._dummy.evaluate(code, proposer_id, proposal_type, context)
        start = time.monotonic()
        try:
            result = self._run_in_subprocess(code, proposer_id, proposal_type, context)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return JudgmentResult(approved=False, reason=f"Judge error: {e}", latency_ms=elapsed, error=str(e))
        elapsed = (time.monotonic() - start) * 1000
        result.latency_ms = elapsed
        return result

    def _run_in_subprocess(self, code: str, proposer_id: int, proposal_type: str, context: Optional[dict]) -> JudgmentResult:
        repo_root = str(Path(__file__).resolve().parents[1])
        script = self._build_judge_script(code, proposer_id, proposal_type, context, repo_root)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name
        try:
            proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return JudgmentResult(approved=False, reason=f"Judge timed out after {self.timeout}s", latency_ms=0.0, error="timeout")
        finally:
            Path(script_path).unlink(missing_ok=True)
        if proc.returncode != 0:
            return JudgmentResult(approved=False, reason=f"Judge subprocess failed: {proc.stderr.strip()[:200]}", latency_ms=0.0, error=proc.stderr.strip()[:200])
        try:
            output = json.loads(proc.stdout.strip())
        except (json.JSONDecodeError, ValueError) as e:
            return JudgmentResult(approved=False, reason=f"Failed to parse judge output: {e}", latency_ms=0.0, error=str(e))
        return JudgmentResult(approved=output.get("approved", False), reason=output.get("reason", "unknown"), latency_ms=0.0)

    def _build_judge_script(self, code: str, proposer_id: int, proposal_type: str, context: Optional[dict], repo_root: str) -> str:
        context_json = json.dumps(context or {})
        lines = [
            "import json", "import sys", "import traceback", "",
            f"sys.path.insert(0, {repr(repo_root)})", "",
            "def main():",
            "    try:",
            "        from MetaIsland.judge import Judge",
            "        judge = Judge()",
            f"        approved, reason = judge.judge_proposal(",
            f"            code={repr(code)},",
            f"            proposer_id={proposer_id},",
            f"            proposal_type={repr(proposal_type)},",
            f"            context=json.loads({repr(context_json)}),",
            "        )",
            '        print(json.dumps({"approved": approved, "reason": reason}))',
            "    except Exception as e:",
            '        print(json.dumps({"approved": False, "reason": f"Judge error: {e}"}))',
            "", "main()",
        ]
        return "\n".join(lines) + "\n"
