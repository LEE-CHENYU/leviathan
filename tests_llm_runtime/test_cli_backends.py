"""Direct tests for CLI backends."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_runtime.backends.aisuite_api import AisuiteAPIBackend
from llm_runtime.backends.claude_cli import ClaudeCLIBackend
from llm_runtime.backends.codex_cli import CodexCLIBackend
from llm_runtime.types import GenerationRequest


class _FakeProcess:
    """Minimal stand-in for asyncio subprocess process."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int | None = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 12345

    async def communicate(self, input: bytes | None = None):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode


class TestCodexCLIBackend(unittest.IsolatedAsyncioTestCase):
    """Codex backend should preserve success and timeout behavior."""

    @patch("llm_runtime.backends.codex_cli.asyncio.create_subprocess_exec")
    async def test_success_reads_output_file(self, mock_exec):
        fake_proc = _FakeProcess(stdout=b"", returncode=0)
        mock_exec.return_value = fake_proc

        backend = CodexCLIBackend({"command": "codex", "timeout": 300})
        request = GenerationRequest(prompt="hello", model="gpt-5.4", timeout=1)

        import builtins

        real_open = builtins.open

        def _fake_open(path, *args, **kwargs):
            if "codex_out_" in str(path):
                handle = MagicMock()
                handle.__enter__ = lambda s: MagicMock(read=lambda: "Hello from Codex")
                handle.__exit__ = lambda s, *a: None
                return handle
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_fake_open):
            response = await backend.generate(request)

        self.assertEqual(response.text, "Hello from Codex")
        self.assertEqual(response.info["status"], "success")

    @patch("llm_runtime.backends.codex_cli.asyncio.create_subprocess_exec")
    async def test_timeout_returns_timeout_status(self, mock_exec):
        fake_proc = _FakeProcess(returncode=None)

        async def _hang(input=None):
            await asyncio.sleep(60)
            return b"", b""

        fake_proc.communicate = _hang
        mock_exec.return_value = fake_proc

        backend = CodexCLIBackend({"command": "codex", "timeout": 1})
        request = GenerationRequest(prompt="hello", model="gpt-5.4", timeout=1)

        with patch("os.getpgid", return_value=99999), patch("os.killpg"):
            response = await backend.generate(request)

        self.assertIsNone(response.text)
        self.assertEqual(response.info["status"], "timeout")


class TestClaudeCLIBackend(unittest.IsolatedAsyncioTestCase):
    """Claude backend should parse structured JSON responses."""

    @patch("llm_runtime.backends.claude_cli.asyncio.create_subprocess_exec")
    async def test_success_parses_json_payload(self, mock_exec):
        payload = {
            "result": "Claude result",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        fake_proc = _FakeProcess(stdout=json.dumps(payload).encode(), returncode=0)
        mock_exec.return_value = fake_proc

        backend = ClaudeCLIBackend({"command": "claude", "timeout": 300})
        request = GenerationRequest(prompt="hello", model="claude-sonnet-4-6", timeout=1)
        response = await backend.generate(request)

        self.assertEqual(response.text, "Claude result")
        self.assertEqual(response.info["status"], "success")
        self.assertEqual(response.info["input_tokens"], 10)
        self.assertEqual(response.info["output_tokens"], 20)


class TestAisuiteAPIBackend(unittest.IsolatedAsyncioTestCase):
    """Aisuite backend should normalize provider-prefixed model names."""

    async def test_adds_provider_prefix_for_bare_model_ids(self):
        captured = {}

        class _FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content='{"ok": true}')
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice])

        fake_ai = SimpleNamespace(
            Client=lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=_FakeCompletions())
            )
        )

        with patch.dict(sys.modules, {"aisuite": fake_ai}):
            backend = AisuiteAPIBackend({"provider": "openai", "timeout": 30})
            response = await backend.generate(
                GenerationRequest(prompt="hello", model="gpt-5.4", timeout=1)
            )

        self.assertEqual(response.info["status"], "success")
        self.assertEqual(captured["model"], "openai:gpt-5.4")
