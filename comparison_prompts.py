"""Phase 3 standardized prompt set (~40 prompts) for the full model comparison
study: memory usage, tokens/sec, and LLM-judged output quality.

Extends Phase 1's category taxonomy (factual, reasoning, code, longform) to
~10 prompts each for broader, more "standardized" coverage.
"""

COMPARISON_PROMPTS = [
    # -- factual ------------------------------------------------------------
    {"id": "cf-1", "category": "factual", "prompt": "What is the capital of France?"},
    {"id": "cf-2", "category": "factual", "prompt": "What year did the first man land on the moon?"},
    {"id": "cf-3", "category": "factual", "prompt": "Who wrote the novel Pride and Prejudice?"},
    {"id": "cf-4", "category": "factual", "prompt": "What is the capital of Japan?"},
    {"id": "cf-5", "category": "factual", "prompt": "What is the chemical symbol for gold?"},
    {"id": "cf-6", "category": "factual", "prompt": "How many continents are there on Earth?"},
    {"id": "cf-7", "category": "factual", "prompt": "What is the largest planet in our solar system?"},
    {"id": "cf-8", "category": "factual", "prompt": "In which year did World War II end?"},
    {"id": "cf-9", "category": "factual", "prompt": "Who painted the Mona Lisa?"},
    {"id": "cf-10", "category": "factual", "prompt": "What is the boiling point of water in Celsius at sea level?"},

    # -- reasoning/explanatory ------------------------------------------------
    {"id": "cr-1", "category": "reasoning", "prompt": "Explain how a hash map works, in 3 sentences."},
    {"id": "cr-2", "category": "reasoning", "prompt": "Why is the sky blue? Explain briefly."},
    {"id": "cr-3", "category": "reasoning", "prompt": "Explain the difference between TCP and UDP in a few sentences."},
    {"id": "cr-4", "category": "reasoning", "prompt": "Explain why a binary search is faster than a linear search."},
    {"id": "cr-5", "category": "reasoning", "prompt": "Why do objects in orbit appear weightless?"},
    {"id": "cr-6", "category": "reasoning", "prompt": "Explain the difference between supervised and unsupervised learning, briefly."},
    {"id": "cr-7", "category": "reasoning", "prompt": "Why does ice float on water?"},
    {"id": "cr-8", "category": "reasoning", "prompt": "Explain what a race condition is in concurrent programming."},
    {"id": "cr-9", "category": "reasoning", "prompt": "Why is Big-O notation useful when analyzing algorithms?"},
    {"id": "cr-10", "category": "reasoning", "prompt": "Explain the difference between a stack and a queue."},

    # -- code generation ------------------------------------------------------
    {"id": "cc-1", "category": "code", "prompt": "Write a Python function to reverse a singly linked list."},
    {"id": "cc-2", "category": "code", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"id": "cc-3", "category": "code", "prompt": "Write a Python function to find the nth Fibonacci number using memoization."},
    {"id": "cc-4", "category": "code", "prompt": "Write a Python function to check if a number is prime."},
    {"id": "cc-5", "category": "code", "prompt": "Write a Python function to merge two sorted lists into one sorted list."},
    {"id": "cc-6", "category": "code", "prompt": "Write a Python function to count word frequency in a string."},
    {"id": "cc-7", "category": "code", "prompt": "Write a Python function to flatten a nested list."},
    {"id": "cc-8", "category": "code", "prompt": "Write a Python function to remove duplicates from a list while preserving order."},
    {"id": "cc-9", "category": "code", "prompt": "Write a Python function that returns the factorial of a number using recursion."},
    {"id": "cc-10", "category": "code", "prompt": "Write a Python function to find the maximum subarray sum (Kadane's algorithm)."},

    # -- long-form/open-ended ----------------------------------------------------
    {"id": "cl-1", "category": "longform", "prompt": "Write a short paragraph about the tradeoffs of local vs cloud LLM inference."},
    {"id": "cl-2", "category": "longform", "prompt": "Describe three practical use cases for an offline AI assistant."},
    {"id": "cl-3", "category": "longform", "prompt": "Write a short paragraph explaining why quantization matters for running LLMs on consumer hardware."},
    {"id": "cl-4", "category": "longform", "prompt": "Write a short paragraph on the privacy benefits of running an AI assistant fully offline."},
    {"id": "cl-5", "category": "longform", "prompt": "Describe the main challenges of deploying AI models at the edge."},
    {"id": "cl-6", "category": "longform", "prompt": "Write a short paragraph comparing small language models to large language models."},
    {"id": "cl-7", "category": "longform", "prompt": "Explain, in a short paragraph, what makes a good system prompt for a local assistant."},
    {"id": "cl-8", "category": "longform", "prompt": "Write a short paragraph about how caching can improve LLM application latency."},
    {"id": "cl-9", "category": "longform", "prompt": "Describe how a developer might decide between three candidate local models for a product."},
    {"id": "cl-10", "category": "longform", "prompt": "Write a short paragraph on the risks of hallucination in local AI assistants and how to mitigate them."},
]
