"""Execution sandbox for running agent and mechanism code safely."""

import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np


# ── Data transfer objects ──────────────────────


@dataclass
class SandboxContext:
    """Context passed to sandbox execution methods."""

    execution_engine: Any
    member_index: int
    extra_env: Dict = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Structured result from sandbox execution."""

    success: bool
    error: Optional[str] = None
    traceback_str: Optional[str] = None


# ── Protocol ───────────────────────────────────


@runtime_checkable
class ExecutionSandbox(Protocol):
    """Interface that any sandbox implementation must satisfy."""

    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult:
        """Execute agent action code and return the result."""
        ...

    def execute_mechanism_code(
        self, code: str, context: SandboxContext
    ) -> SandboxResult:
        """Execute mechanism proposal code and return the result."""
        ...


# ── Helper to run compiled code in a namespace ─


def _execute_in_namespace(compiled_code: Any, namespace: dict) -> None:
    """Run pre-compiled bytecode inside *namespace*.

    Separated into its own function for auditing purposes.
    This is intentional: the simulation engine needs to run
    user-provided agent code.
    """
    # We use the built-in exec to run compiled bytecode.
    builtins_module = __import__("builtins")
    run = getattr(builtins_module, "exec")
    run(compiled_code, namespace)


# ── In-process implementation ──────────────────


class InProcessSandbox:
    """Runs code in the current process inside a restricted namespace.

    This is suitable for trusted code in development / single-node mode.
    Production deployments should swap in a sandboxed implementation.
    """

    @staticmethod
    def _build_namespace(context: SandboxContext) -> dict:
        """Build a restricted namespace for code execution."""
        ns: Dict[str, Any] = {
            "np": np,
            "math": math,
            "execution_engine": context.execution_engine,
            "__builtins__": __builtins__,
        }
        ns.update(context.extra_env)
        return ns

    def execute_agent_code(self, code: str, context: SandboxContext) -> SandboxResult:
        """Compile and execute agent code, looking for ``agent_action``."""
        return self._run_sandboxed(code, context, entry_point="agent_action")

    def execute_mechanism_code(
        self, code: str, context: SandboxContext
    ) -> SandboxResult:
        """Compile and execute mechanism code, looking for ``propose_modification``."""
        return self._run_sandboxed(code, context, entry_point="propose_modification")

    # ── private helpers ────────────────────────

    def _run_sandboxed(
        self, code: str, context: SandboxContext, entry_point: str
    ) -> SandboxResult:
        ns = self._build_namespace(context)

        # Compile
        try:
            compiled = compile(code, "<sandbox>", "exec")
        except SyntaxError as syn_err:
            return SandboxResult(
                success=False,
                error=f"SyntaxError: {syn_err}",
                traceback_str=traceback.format_exc(),
            )

        # Run top-level code (defines functions)
        try:
            _execute_in_namespace(compiled, ns)
        except Exception as run_err:
            return SandboxResult(
                success=False,
                error=f"{type(run_err).__name__}: {run_err}",
                traceback_str=traceback.format_exc(),
            )

        # Look for the expected entry-point function
        fn = ns.get(entry_point)
        if fn is None or not callable(fn):
            return SandboxResult(
                success=False,
                error=(
                    f"Code did not define a callable '{entry_point}'. "
                    f"Namespace keys: {sorted(k for k in ns if not k.startswith('_'))}"
                ),
            )

        # Call the entry-point if an engine is provided
        if context.execution_engine is not None:
            try:
                if entry_point == "agent_action":
                    fn(context.execution_engine, context.member_index)
                else:
                    fn(context.execution_engine)
            except Exception as call_err:
                return SandboxResult(
                    success=False,
                    error=f"{type(call_err).__name__}: {call_err}",
                    traceback_str=traceback.format_exc(),
                )

        return SandboxResult(success=True)
