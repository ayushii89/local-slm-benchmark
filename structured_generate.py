"""Core structured-generation logic: force a model to answer in a fixed JSON
schema, validate with Pydantic, and retry once before failing gracefully.

Generalized across Phase 2 (StructuredAnswer) and Phase 3 (QualityScore judge
calls) -- callers pass the Pydantic schema and the prompt builders to use.
"""

import json
from typing import Callable

import ollama
from pydantic import BaseModel, ValidationError

from schemas import StructuredAnswer

# A plain field spec + concrete example, NOT a raw JSON Schema dump: dumping
# model_json_schema() confused mistral-7b into echoing the schema itself
# (or wrapping the answer in a `"StructuredAnswer"` key) instead of producing
# an instance -- 8/8 failures in testing. An example fixes this reliably.
STRUCTURED_ANSWER_SPEC = (
    'Fields:\n'
    '  answer (string): the answer to the question\n'
    '  confidence (number, 0.0 to 1.0): how confident you are\n'
    '  reasoning (string): a brief explanation\n'
    'Example: {"answer": "Paris", "confidence": 0.95, "reasoning": "Paris is the capital of France."}'
)


def build_prompt(question: str) -> str:
    return (
        f"Answer the following question.\n\n"
        f"Question: {question}\n\n"
        f"Respond with ONLY a JSON object with exactly these fields (no extra "
        f"text, no markdown code fences, no nested wrapper key):\n{STRUCTURED_ANSWER_SPEC}\n"
    )


def build_retry_prompt(question: str, previous_output: str, error: str) -> str:
    return (
        f"Answer the following question.\n\n"
        f"Question: {question}\n\n"
        f"Your previous response was invalid:\n{previous_output}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Respond again with ONLY a corrected JSON object with exactly these "
        f"fields (no extra text, no markdown code fences, no nested wrapper key):\n{STRUCTURED_ANSWER_SPEC}\n"
    )


def _try_parse(raw_text: str, schema: type[BaseModel]) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"

    try:
        validated = schema.model_validate(data)
    except ValidationError as e:
        return None, f"ValidationError: {e}"

    return validated.model_dump(), None


def generate_structured(
    tag: str,
    question: str,
    schema: type[BaseModel],
    prompt_builder: Callable[[str], str],
    retry_prompt_builder: Callable[[str, str, str], str],
    max_retries: int = 1,
) -> dict:
    """Generate a `schema`-validated response for `question` using Ollama model `tag`.

    Args:
        prompt_builder: (question) -> initial prompt text.
        retry_prompt_builder: (question, previous_output, error) -> retry prompt text.

    Returns a diagnostics dict:
        {
          "success": bool,
          "attempts": int,
          "parsed": dict | None,
          "raw_response": str,
          "error": str | None,
        }
    Never raises -- validation/parse failures are captured, not propagated.
    """
    prompt = prompt_builder(question)
    raw_text = ""
    error = None

    for attempt in range(1, max_retries + 2):  # e.g. max_retries=1 -> attempts 1, 2
        try:
            response = ollama.generate(model=tag, prompt=prompt, format="json", stream=False)
            raw_text = response.get("response", "")
        except Exception as e:
            error = f"OllamaCallError: {e}"
            break

        parsed, error = _try_parse(raw_text, schema)
        if parsed is not None:
            return {
                "success": True,
                "attempts": attempt,
                "parsed": parsed,
                "raw_response": raw_text,
                "error": None,
            }

        if attempt <= max_retries:
            prompt = retry_prompt_builder(question, raw_text, error)

    return {
        "success": False,
        "attempts": min(attempt, max_retries + 1),
        "parsed": None,
        "raw_response": raw_text,
        "error": error,
    }
