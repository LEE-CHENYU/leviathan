"""Tests for YAML-backed LLM service profile resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_runtime.service import LLMService
from llm_runtime.types import GenerationResponse


class _FakeBackend:
    """Backend double that records the merged request."""

    last_request = None

    def __init__(self, settings):
        self.settings = dict(settings)

    async def generate(self, request):
        type(self).last_request = request
        return GenerationResponse(text="ok", elapsed=0.01, info={"status": "success"})


class TestLLMService(unittest.IsolatedAsyncioTestCase):
    """Service should merge profile YAML and runtime overrides correctly."""

    async def test_profile_config_and_overrides_are_merged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            backends_dir = root / "backends"
            profiles_dir = root / "profiles"
            backends_dir.mkdir()
            profiles_dir.mkdir()

            (backends_dir / "fake.yaml").write_text("type: fake\n")
            (profiles_dir / "demo.yaml").write_text(
                "\n".join(
                    [
                        "backend: fake",
                        "model: gpt-5.4",
                        "temperature: 0.7",
                        "max_tokens: 4096",
                        "timeout: 300",
                        "reasoning: xhigh",
                    ]
                )
            )

            with patch.dict("llm_runtime.service.BACKEND_TYPES", {"fake": _FakeBackend}, clear=False):
                service = LLMService(backends_dir=backends_dir, profiles_dir=profiles_dir)
                response = await service.generate(
                    prompt="hello",
                    profile="demo",
                    temperature=0.2,
                    timeout=30,
                )

        self.assertEqual(response.text, "ok")
        self.assertEqual(response.info["status"], "success")
        self.assertEqual(_FakeBackend.last_request.prompt, "hello")
        self.assertEqual(_FakeBackend.last_request.model, "gpt-5.4")
        self.assertEqual(_FakeBackend.last_request.temperature, 0.2)
        self.assertEqual(_FakeBackend.last_request.max_tokens, 4096)
        self.assertEqual(_FakeBackend.last_request.timeout, 30)
        self.assertEqual(_FakeBackend.last_request.reasoning, "xhigh")
