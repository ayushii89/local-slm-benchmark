"""Integration: prompt builders for grounded RAG question-answering.

Used with the generalized generate_structured() in structured_generate.py,
scoring against the RAGAnswer schema. Mirrors judge_prompts.py's plain
field-spec-plus-example pattern (a raw JSON Schema dump broke mistral-7b
during Phase 2 testing -- see structured_generate.py).
"""

RAG_ANSWER_SPEC = (
    'Fields:\n'
    '  answer (string): the answer to the question, grounded only in the context below\n'
    '  citations (array of strings): the exact [Source: ...] label(s) the answer drew from, '
    'copied character-for-character from the context -- never shortened, never just a number\n'
    '  confidence (number, 0.0 to 1.0): how confident you are\n'
    'Example: given context containing "[Source: 1 various sdlc models.pdf]", '
    'a correct citation is "1 various sdlc models.pdf", NOT "1".\n'
    'Full example: {"answer": "Reranking improved precision by reordering the fused candidates.", '
    '"citations": ["eval/results.md"], "confidence": 0.9}'
)

GROUNDING_RULE = (
    "Answer ONLY using the context below. If the context does not contain the "
    "answer, say so plainly in the answer field (and use a low confidence) "
    "rather than guessing. Cite every source label you actually used, each "
    "label at most once, copied exactly as it appears after 'Source:' -- even "
    "if the label itself starts with a digit, copy the whole label, not just "
    "the digit."
)


HISTORY_AMBIGUITY_RULE = (
    "The current question may be a vague continuation (e.g. 'what about X?'). "
    "If so, the context usually contains several distinct facts about the "
    "topic (a definition, advantages, when to use it, etc.) -- prefer "
    "whichever one matches the *kind* of thing asked about in the "
    "conversation so far (e.g. if prior turns asked about advantages, lean "
    "toward advantages, not an unrelated fact). If it's still genuinely "
    "unclear which aspect is meant, lower your confidence to reflect that "
    "ambiguity instead of guessing at full confidence."
)


def format_history(history: list[tuple[str, str]]) -> str:
    """Renders prior (question, answer) turns for follow-up questions like
    'what about its disadvantages?' that only make sense with context.
    Empty string if there's no history (first turn / one-shot mode).
    """
    if not history:
        return ""
    turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    return f"Conversation so far:\n{turns}\n\n{HISTORY_AMBIGUITY_RULE}\n\n"


def build_rag_prompt(question: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    return (
        f"{format_history(history)}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{GROUNDING_RULE}\n\n"
        f"Respond with ONLY a JSON object with exactly these fields (no extra "
        f"text, no markdown code fences, no nested wrapper key):\n{RAG_ANSWER_SPEC}\n"
    )


def build_rag_retry_prompt(
    question: str, context: str, previous_output: str, error: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    return (
        f"{format_history(history)}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{GROUNDING_RULE}\n\n"
        f"Your previous response was invalid:\n{previous_output}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Respond again with ONLY a corrected JSON object with exactly these "
        f"fields (no extra text, no markdown code fences, no nested wrapper key):\n{RAG_ANSWER_SPEC}\n"
    )
