"""Phase 1 prompt set -- performance measurement only, not the Phase 3 quality eval set.

12 prompts across 4 categories, chosen to vary expected generation length so
timing/throughput numbers aren't all measured on one kind of response.
"""

PHASE1_PROMPTS = [
    # -- factual: short, low generation-length ---------------------------------
    {"id": "factual-1", "category": "factual", "prompt": "What is the capital of France?"},
    {"id": "factual-2", "category": "factual", "prompt": "What year did the first man land on the moon?"},
    {"id": "factual-3", "category": "factual", "prompt": "Who wrote the novel Pride and Prejudice?"},

    # -- reasoning/explanatory: medium length -----------------------------------
    {"id": "reasoning-1", "category": "reasoning", "prompt": "Explain how a hash map works, in 3 sentences."},
    {"id": "reasoning-2", "category": "reasoning", "prompt": "Why is the sky blue? Explain briefly."},
    {"id": "reasoning-3", "category": "reasoning", "prompt": "Explain the difference between TCP and UDP in a few sentences."},

    # -- code generation: medium-to-long ------------------------------------------
    {"id": "code-1", "category": "code", "prompt": "Write a Python function to reverse a singly linked list."},
    {"id": "code-2", "category": "code", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"id": "code-3", "category": "code", "prompt": "Write a Python function to find the nth Fibonacci number using memoization."},

    # -- long-form/open-ended: longest expected generations -----------------------
    {"id": "longform-1", "category": "longform", "prompt": "Write a short paragraph about the tradeoffs of local vs cloud LLM inference."},
    {"id": "longform-2", "category": "longform", "prompt": "Describe three practical use cases for an offline AI assistant."},
    {"id": "longform-3", "category": "longform", "prompt": "Write a short paragraph explaining why quantization matters for running LLMs on consumer hardware."},
]
