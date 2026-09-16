"""Unit tests for rag_prompts.py and its use with generate_structured()
against the RAGAnswer schema -- ollama.generate mocked, no server needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structured_generate
from rag_prompts import RAG_ANSWER_SPEC, build_rag_prompt, build_rag_retry_prompt
from schemas import RAGAnswer
from structured_generate import generate_structured

VALID_RAG_JSON = (
    '{"answer": "Reranking improved precision.", '
    '"citations": ["eval/results.md"], "confidence": 0.85}'
)


def test_build_rag_prompt_embeds_question_context_and_spec():
    prompt = build_rag_prompt("What did reranking do?", "[Source: eval/results.md]\nRerank raised precision.")

    assert "What did reranking do?" in prompt
    assert "[Source: eval/results.md]" in prompt
    assert "Rerank raised precision." in prompt
    assert RAG_ANSWER_SPEC in prompt


def test_build_rag_retry_prompt_includes_error_and_previous_output():
    prompt = build_rag_retry_prompt(
        "What did reranking do?", "[Source: eval/results.md]\ncontext",
        previous_output="not json", error="JSONDecodeError: bad",
    )

    assert "not json" in prompt
    assert "JSONDecodeError: bad" in prompt


def test_generate_structured_with_rag_answer_schema(monkeypatch):
    def fake_generate(model, prompt, format=None, stream=False):
        return {"response": VALID_RAG_JSON}

    monkeypatch.setattr(structured_generate.ollama, "generate", fake_generate)

    context = "[Source: eval/results.md]\nReranking raised precision by 12%."

    def prompt_builder(q: str) -> str:
        return build_rag_prompt(q, context)

    def retry_prompt_builder(q: str, prev: str, err: str) -> str:
        return build_rag_retry_prompt(q, context, prev, err)

    result = generate_structured(
        "fake-tag", "What did reranking do?",
        schema=RAGAnswer, prompt_builder=prompt_builder, retry_prompt_builder=retry_prompt_builder,
    )

    assert result["success"] is True
    assert result["parsed"]["citations"] == ["eval/results.md"]
    assert 0.0 <= result["parsed"]["confidence"] <= 1.0
