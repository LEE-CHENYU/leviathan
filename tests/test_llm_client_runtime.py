from __future__ import annotations

from llm_runtime.types import GenerationResponse
from MetaIsland.llm_client import get_llm_client


class _FakeService:
    def __init__(self):
        self.calls: list[dict] = []

    def get_profile(self, name: str) -> dict:
        return {"backend": "codex_cli", "model": "gpt-5.4"}

    async def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return GenerationResponse(
            text='{"ok": true}',
            elapsed=0.01,
            info={
                "status": "success",
                "input_tokens": 12,
                "output_tokens": 5,
            },
        )


def test_runtime_client_uses_shared_llm_runtime(monkeypatch):
    fake_service = _FakeService()
    monkeypatch.delenv("LLM_OFFLINE", raising=False)
    monkeypatch.setenv("METAISLAND_LLM_PROFILE", "codex_cli_default")
    monkeypatch.setattr("MetaIsland.llm_client.get_default_service", lambda: fake_service)

    client = get_llm_client()
    completion = client.chat.completions.create(
        model="openai:gpt-5.4",
        messages=[
            {"role": "system", "content": "You are strict."},
            {"role": "user", "content": "Return JSON."},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=64,
        temperature=0.2,
    )

    assert fake_service.calls == [
        {
            "prompt": "[SYSTEM]\nYou are strict.\n\n[USER]\nReturn JSON.",
            "profile": "codex_cli_default",
            "model": "gpt-5.4",
            "temperature": 0.2,
            "max_tokens": 64,
            "timeout": None,
            "json_output": True,
        }
    ]
    assert completion.choices[0].message.content == '{"ok": true}'
    assert completion.usage.prompt_tokens == 12
    assert completion.usage.completion_tokens == 5
    assert completion.usage.total_tokens == 17
