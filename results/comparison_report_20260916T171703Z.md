# Phase 3: Model Comparison Study

Source data: `comparison_20260916T163623Z.jsonl`

*Quality scores are the mean of judgments from the other two models (never a model judging its own answer) -- a single local LLM judge showed inconsistent scoring in testing, so this averages across two independent judges rather than trusting one.*

## Summary

| Model | Mean tok/s | Mean VRAM (MB) | Mean quality (1-5) | Scored below 3 |
|---|---|---|---|---|
| llama3.2-3b | 34.2 | 2226 | 4.71 | 0/40 |
| mistral-7b | 15.8 | 4482 | 4.75 | 0/40 |
| phi4-mini | 29.6 | 2713 | 4.65 | 0/40 |

## Per-model analysis

**llama3.2-3b** averaged 34.2 tokens/sec and 2226MB VRAM across 40 prompts, with a mean quality score of 4.71/5 (0 of 40 judged responses scored below 3). By category: code: 4.45, factual: 4.95, longform: 4.55, reasoning: 4.90.

**mistral-7b** averaged 15.8 tokens/sec and 4482MB VRAM across 40 prompts, with a mean quality score of 4.75/5 (0 of 40 judged responses scored below 3). By category: code: 4.85, factual: 4.90, longform: 4.65, reasoning: 4.60.

**phi4-mini** averaged 29.6 tokens/sec and 2713MB VRAM across 40 prompts, with a mean quality score of 4.65/5 (0 of 40 judged responses scored below 3). By category: code: 4.55, factual: 5.00, longform: 4.40, reasoning: 4.65.

## Recommendation

Highest judged quality: **mistral-7b** (4.75/5). Fastest: **llama3.2-3b** (34.2 tok/s). Smallest memory footprint: **llama3.2-3b** (2226MB). Choose based on which constraint (quality, latency, or memory) matters most for the target deployment.
