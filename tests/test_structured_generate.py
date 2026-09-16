"""Unit tests for structured_generate.generate_structured(): validate, retry
once, fail gracefully -- with ollama.generate mocked so no server is needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structured_generate
from schemas import StructuredAnswer
from structured_generate import build_prompt, build_retry_prompt, generate_structured

VALID_JSON = '{"answer": "Paris", "confidence": 0.9, "reasoning": "It just is."}'
INVALID_JSON = "not json at all"
SCHEMA_MISMATCH_JSON = '{"answer": "Paris"}'  # missing confidence/reasoning


def _fake_generate_sequence(responses):
    """Returns a stand-in for ollama.generate that yields each response in
    order on successive calls (as if the model 'response' text is `responses[i]`).
    """
    calls = {"n": 0}

    def fake_generate(model, prompt, format=None, stream=False):
        text = responses[calls["n"]]
        calls["n"] += 1
        return {"response": text}

    return fake_generate


def test_success_on_first_try(monkeypatch):
    monkeypatch.setattr(structured_generate.ollama, "generate", _fake_generate_sequence([VALID_JSON]))

    result = generate_structured(
        "fake-tag", "What is the capital of France?",
        schema=StructuredAnswer, prompt_builder=build_prompt, retry_prompt_builder=build_retry_prompt,
    )

    assert result["success"] is True
    assert result["attempts"] == 1
    assert result["parsed"]["answer"] == "Paris"
    assert result["error"] is None


def test_invalid_then_valid_on_retry(monkeypatch):
    monkeypatch.setattr(
        structured_generate.ollama, "generate",
        _fake_generate_sequence([INVALID_JSON, VALID_JSON]),
    )

    result = generate_structured(
        "fake-tag", "What is the capital of France?",
        schema=StructuredAnswer, prompt_builder=build_prompt, retry_prompt_builder=build_retry_prompt,
    )

    assert result["success"] is True
    assert result["attempts"] == 2
    assert result["parsed"]["answer"] == "Paris"


def test_fails_gracefully_after_max_retries(monkeypatch):
    monkeypatch.setattr(
        structured_generate.ollama, "generate",
        _fake_generate_sequence([SCHEMA_MISMATCH_JSON, SCHEMA_MISMATCH_JSON]),
    )

    result = generate_structured(
        "fake-tag", "What is the capital of France?",
        schema=StructuredAnswer, prompt_builder=build_prompt, retry_prompt_builder=build_retry_prompt,
        max_retries=1,
    )

    assert result["success"] is False
    assert result["attempts"] == 2
    assert result["parsed"] is None
    assert "ValidationError" in result["error"]


def test_ollama_call_error_is_captured_not_raised(monkeypatch):
    def raising_generate(model, prompt, format=None, stream=False):
        raise ConnectionError("no server")

    monkeypatch.setattr(structured_generate.ollama, "generate", raising_generate)

    result = generate_structured(
        "fake-tag", "What is the capital of France?",
        schema=StructuredAnswer, prompt_builder=build_prompt, retry_prompt_builder=build_retry_prompt,
    )

    assert result["success"] is False
    assert result["parsed"] is None
    assert "OllamaCallError" in result["error"]
