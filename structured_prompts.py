"""Phase 2 prompt set -- questions used to test structured-output reliability.

Each is a question with a checkable-by-eye correct answer, so a spot-check of
`parsed.answer` can confirm the model is both valid AND correct, not just valid.
"""

STRUCTURED_PROMPTS = [
    {"id": "sq-1", "question": "What is the capital of Japan?"},
    {"id": "sq-2", "question": "What is 12 multiplied by 8?"},
    {"id": "sq-3", "question": "Who developed the theory of general relativity?"},
    {"id": "sq-4", "question": "What is the chemical symbol for gold?"},
    {"id": "sq-5", "question": "How many continents are there on Earth?"},
    {"id": "sq-6", "question": "What is the largest planet in our solar system?"},
    {"id": "sq-7", "question": "In which year did World War II end?"},
    {"id": "sq-8", "question": "What is the boiling point of water in Celsius at sea level?"},
]
