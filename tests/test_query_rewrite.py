"""Unit tests for query_rewrite.rewrite_query(): self-contained rewriting of
ambiguous follow-ups, with graceful fallback to the original question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structured_generate
from query_rewrite import rewrite_query

VALID_REWRITE_JSON = '{"query": "What are the advantages of the Agile model?"}'
INVALID_JSON = "not json"


def test_no_history_returns_question_unchanged(monkeypatch):
    def should_not_be_called(*a, **kw):
        raise AssertionError("ollama.generate should not be called with no history")

    monkeypatch.setattr(structured_generate.ollama, "generate", should_not_be_called)

    result = rewrite_query("fake-tag", "What about Agile?", history=[])
    assert result == "What about Agile?"


def test_successful_rewrite_returns_rewritten_query(monkeypatch):
    def fake_generate(model, prompt, format=None, stream=False):
        return {"response": VALID_REWRITE_JSON}

    monkeypatch.setattr(structured_generate.ollama, "generate", fake_generate)

    history = [("What is the V-Model?", "A sequential SDLC model."),
               ("What are its advantages?", "Clear phase separation.")]
    result = rewrite_query("fake-tag", "What about Agile?", history)

    assert result == "What are the advantages of the Agile model?"


def test_failed_rewrite_falls_back_to_original_question(monkeypatch):
    def fake_generate(model, prompt, format=None, stream=False):
        return {"response": INVALID_JSON}

    monkeypatch.setattr(structured_generate.ollama, "generate", fake_generate)

    history = [("What is the V-Model?", "A sequential SDLC model.")]
    result = rewrite_query("fake-tag", "What about Agile?", history)

    # Falls back rather than propagating a failure -- never blocks the pipeline.
    assert result == "What about Agile?"


def test_ollama_error_falls_back_to_original_question(monkeypatch):
    def raising_generate(model, prompt, format=None, stream=False):
        raise ConnectionError("no server")

    monkeypatch.setattr(structured_generate.ollama, "generate", raising_generate)

    history = [("What is the V-Model?", "A sequential SDLC model.")]
    result = rewrite_query("fake-tag", "What about Agile?", history)

    assert result == "What about Agile?"
