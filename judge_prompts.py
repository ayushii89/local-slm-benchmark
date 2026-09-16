"""Phase 3: prompt builders for the LLM-as-judge quality scoring step.

Used with the generalized `generate_structured()` in structured_generate.py,
scoring a candidate model's answer against the QualityScore schema.
"""

# A plain field spec + concrete example, NOT a raw JSON Schema dump: dumping
# model_json_schema() confused mistral-7b into echoing the schema itself
# instead of producing an instance (see structured_generate.py). Same fix here.
QUALITY_SCORE_SPEC = (
    'Fields:\n'
    '  score (integer, 1 to 5): the quality score\n'
    '  justification (string): a brief explanation for the score\n'
    'Example: {"score": 4, "justification": "Correct and clearly explained."}'
)

RUBRIC = (
    "Score the ANSWER to the QUESTION on a 1-5 scale:\n"
    "  1 = wrong or incoherent\n"
    "  2 = mostly wrong, or correct but very unclear\n"
    "  3 = partially correct or correct but poorly explained\n"
    "  4 = correct and reasonably clear\n"
    "  5 = fully correct, clear, and well-explained\n"
)


def build_judge_prompt(question: str, answer: str) -> str:
    return (
        f"You are grading an AI assistant's answer to a question.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"{RUBRIC}\n"
        f"Respond with ONLY a JSON object with exactly these fields (no extra "
        f"text, no markdown code fences, no nested wrapper key):\n{QUALITY_SCORE_SPEC}\n"
    )


def build_judge_retry_prompt(question: str, answer: str, previous_output: str, error: str) -> str:
    return (
        f"You are grading an AI assistant's answer to a question.\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"{RUBRIC}\n"
        f"Your previous response was invalid:\n{previous_output}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Respond again with ONLY a corrected JSON object with exactly these "
        f"fields (no extra text, no markdown code fences, no nested wrapper key):\n{QUALITY_SCORE_SPEC}\n"
    )
