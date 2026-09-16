"""Integration: rewrite an ambiguous follow-up into a self-contained query
before retrieval, using conversation history.

Fixes a real failure found during hardening: a vague follow-up like "What
about Agile?" after several turns about a different SDLC model retrieves
fine (the word "Agile" is right there), but llama3.2-3b sometimes picks the
wrong chunk among several plausible ones and answers at full confidence
regardless. A prompt-only fix (an ambiguity-resolution instruction added
directly to the answer prompt) did not reliably fix this -- confirmed by
re-testing the same failing turn, and by cross-checking that mistral-7b
handles the same case correctly, meaning it's a capability limit, not a
wording problem.

Decoupling "figure out what's actually being asked" from "answer it" into
two focused steps -- rather than asking one prompt to disambiguate, answer,
and calibrate confidence all at once -- is the standard fix for this pattern
in RAG systems (query condensation). This module is that step. It reuses
the same generate_structured() validate/retry/fail-gracefully path as every
other structured call in this project; on any failure it falls back to the
original question rather than blocking the pipeline.
"""

from schemas import RewrittenQuery
from structured_generate import generate_structured

REWRITE_SPEC = (
    'Fields:\n'
    '  query (string): the standalone, self-contained rewrite\n'
    'Example: history mentions "the V-Model", current question is '
    '"what about Agile?" -> {"query": "What are the characteristics of the Agile model?"}'
)

REWRITE_RULE = (
    "Rewrite ONLY the CURRENT QUESTION below into a fully self-contained "
    "question a reader with no other context could understand -- resolve "
    "any pronoun or vague reference ('it', 'that', 'what about X') using the "
    "conversation history. If the current question is already "
    "self-contained, return it unchanged. Do not answer the question, only "
    "rewrite it. If prior questions asked about a specific aspect (e.g. "
    "advantages, comparisons), prefer continuing that aspect when the "
    "current question is vague about which aspect it means."
)


def build_rewrite_prompt(history_and_question: str) -> str:
    return (
        f"{history_and_question}\n\n"
        f"{REWRITE_RULE}\n\n"
        f"Respond with ONLY a JSON object with exactly this field (no extra "
        f"text, no markdown code fences, no nested wrapper key):\n{REWRITE_SPEC}\n"
    )


def build_rewrite_retry_prompt(history_and_question: str, previous_output: str, error: str) -> str:
    return (
        f"{history_and_question}\n\n"
        f"{REWRITE_RULE}\n\n"
        f"Your previous response was invalid:\n{previous_output}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Respond again with ONLY a corrected JSON object with exactly this "
        f"field (no extra text, no markdown code fences, no nested wrapper key):\n{REWRITE_SPEC}\n"
    )


def rewrite_query(tag: str, question: str, history: list[tuple[str, str]]) -> str:
    """Returns a self-contained rewrite of `question` using `history`, or the
    original `question` unchanged if there's no history or rewriting fails.
    Never raises.
    """
    if not history:
        return question

    turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    history_and_question = f"Conversation so far:\n{turns}\n\nCURRENT QUESTION: {question}"

    def prompt_builder(_: str) -> str:
        return build_rewrite_prompt(history_and_question)

    def retry_prompt_builder(_: str, prev: str, err: str) -> str:
        return build_rewrite_retry_prompt(history_and_question, prev, err)

    result = generate_structured(
        tag, question,
        schema=RewrittenQuery,
        prompt_builder=prompt_builder,
        retry_prompt_builder=retry_prompt_builder,
    )

    if result["success"] and result["parsed"]["query"].strip():
        return result["parsed"]["query"].strip()
    return question
