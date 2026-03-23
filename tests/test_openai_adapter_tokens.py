from __future__ import annotations

import sys
import types
from dataclasses import dataclass


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _RecorderCompletions:
    def __init__(self, *, fail_on_max_completion_tokens: bool = False) -> None:
        self.calls: list[dict] = []
        self._fail_on_max_completion_tokens = fail_on_max_completion_tokens

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_on_max_completion_tokens and "max_completion_tokens" in kwargs:
            raise Exception("Unsupported parameter: 'max_completion_tokens'")
        return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content='{"ok": true}'))])


class _FakeOpenAI:
    def __init__(self, *, fail_on_max_completion_tokens: bool = False):
        self.chat = types.SimpleNamespace(
            completions=_RecorderCompletions(
                fail_on_max_completion_tokens=fail_on_max_completion_tokens
            )
        )


def _install_fake_openai(monkeypatch, *, fail_on_max_completion_tokens: bool = False):
    created_clients: list[_FakeOpenAI] = []

    def _factory(**_kwargs):
        client = _FakeOpenAI(
            fail_on_max_completion_tokens=fail_on_max_completion_tokens
        )
        created_clients.append(client)
        return client

    fake_mod = types.SimpleNamespace(OpenAI=_factory)
    monkeypatch.setitem(sys.modules, "openai", fake_mod)
    return created_clients


def test_openai_adapter_uses_max_completion_tokens_for_gpt5(monkeypatch) -> None:
    from llm_synthesis.adapter import OpenAILLMAdapter

    clients = _install_fake_openai(monkeypatch)
    adapter = OpenAILLMAdapter(model="gpt-5.4", max_tokens=321, api_key="test-key")
    raw = adapter.generate("hello")

    assert raw == '{"ok": true}'
    assert len(clients) == 1
    calls = clients[0].chat.completions.calls
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 321
    assert "max_tokens" not in calls[0]


def test_openai_adapter_falls_back_to_max_tokens_when_unsupported(monkeypatch) -> None:
    from llm_synthesis.adapter import OpenAILLMAdapter

    clients = _install_fake_openai(monkeypatch, fail_on_max_completion_tokens=True)
    adapter = OpenAILLMAdapter(model="gpt-4o", max_tokens=123, api_key="test-key")
    raw = adapter.generate("hello")

    assert raw == '{"ok": true}'
    assert len(clients) == 1
    calls = clients[0].chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_completion_tokens"] == 123
    assert calls[1]["max_tokens"] == 123
