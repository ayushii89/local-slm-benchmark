# local-slm-benchmark

[![Tests](https://github.com/ayushii89/local-slm-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/ayushii89/local-slm-benchmark/actions/workflows/tests.yml)

Foundation work for an offline local AI assistant: benchmarking small local
language models served via [Ollama](https://ollama.com), in three phases.

## Models benchmarked

- `llama3.2:3b`
- `phi4-mini`
- `mistral:7b`

See [models.py](models.py).

## Setup

```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1: Measurement

Raw inference speed, computed from Ollama's own response timing fields
(nanoseconds), not client-side wall-clock timing:

- **TTFT (time to first token, ms)** = `prompt_eval_duration / 1e6`
- **Tokens/sec** = `eval_count / (eval_duration / 1e9)`
- **Total latency (ms)** = `total_duration / 1e6`
- **Cold-start load duration (ms)** = `load_duration / 1e6`, recorded once per
  model from a warm-up call, kept separate from steady-state numbers.

```bash
python benchmark.py                          # all models, 3 repeats x 12 prompts
python benchmark.py --models llama3.2-3b --repeats 5
python report.py                              # console table from the latest results file
python report.py results/20260101T000000Z.jsonl
```

## Phase 2: Structure & Determinism

Forces JSON output matching a Pydantic schema (`schemas.StructuredAnswer`),
validates it, and retries once with the validation error fed back before
failing gracefully. Core logic in `structured_generate.generate_structured()`,
generalized (schema + prompt builders as arguments) so Phase 3's judge step
reuses the same validate/retry/fail-gracefully path.

```bash
python run_structured_eval.py
python structured_report.py                   # success (1st try) / after-retry / failed, per model
```

## Phase 3: Model Comparison Study

Speed + peak memory + LLM-judged output quality across a standardized
40-prompt set (`comparison_prompts.py`, 10 each of factual/reasoning/code/
longform). Each candidate's answer is judged by the *other two* models only
(never itself) and `quality_score` is the mean of those two independent
judgments. Memory is the *peak* VRAM/RAM sampled every 200ms during
generation (`memory_utils.PeakMemoryTracker`), not a single post-call snapshot.

```bash
python run_comparison.py --models llama3.2-3b   # smoke-test one model first
python run_comparison.py                        # full 3-model sweep (slow -- mistral-7b is both a
                                                  # candidate and a judge for every prompt)
python comparison_report.py                      # console table + writes a Markdown technical report
```

## Integration: Local RAG Assistant

Reuses `~/rag-project`'s existing retrieval pipeline (hybrid semantic + BM25,
RRF fusion, cross-encoder reranking, already-ingested `chroma_index/`) but
swaps its cloud (Groq) generation step for a local Ollama model, validated
against `schemas.RAGAnswer` (answer, citations, confidence) through the same
`generate_structured()` used in Phase 2/3. `~/rag-project`'s own files are
not modified, only imported (read-only). Model defaults to `llama3.2-3b`,
per Phase 3's finding: at this prompt set, quality across all three models is
within judge noise (4.65-4.75/5), so llama3.2-3b's ~2x speed and half the
VRAM of mistral-7b is the better tradeoff, not a quality compromise.

Run with the **system Python** (`python3`, not this project's `.venv`) --
it already has `rag-project`'s langchain/chroma/sentence-transformers stack
installed; only `pip install ollama` was needed there. The `.venv` stays
deliberately free of that heavier dependency set.

```bash
python3 assistant.py "What does the eval say about reranking?"   # one-shot
python3 assistant.py                                              # interactive loop
python3 assistant.py --model mistral-7b --k 8 "..."
python3 assistant.py --corpus ~/rag-project/chroma_db "..."      # answer over a different index
```

Citations are validated after generation, not just trusted from the prompt:
the model's raw citations are matched against the sources actually retrieved
(normalized against `"Source: "`/`[...]` wrapping, then a substring fallback
for a truncated filename like `"Agile model (1).pdf"` against the real
`"2 Incremental SDLC Model, ..., Agile model (1).pdf"`), and anything that
still doesn't match is dropped and reported rather than shown as a real
citation. Pydantic alone can't catch this -- `citations` is just `list[str]`,
so a shortened, wrapped, or fabricated source label passes schema validation
but is still wrong.

Interactive mode keeps the last `--history-turns` Q&A pairs (default 3) so
follow-ups like "what are its disadvantages?" resolve against the prior
question. A genuinely ambiguous follow-up (e.g. "what about Agile?" after
several turns about a different SDLC model, where the retrieved context
contains multiple distinct facts about Agile) used to make `llama3.2-3b`
pick a narrow or unhelpful chunk and stay overconfident about it -- a prompt
instruction alone didn't fix this (confirmed by re-testing after adding one,
and by checking `mistral-7b` handled the same case correctly, meaning it was
a capability limit, not a wording problem). The actual fix, in
`query_rewrite.py`: before retrieval, an ambiguous follow-up is first
rewritten into a self-contained question ("what about Agile?" ->
"What are the characteristics of the Agile model?") using conversation
history, through the same generate_structured() validate/retry/
fail-gracefully path as everything else -- decoupling "figure out what's
being asked" from "answer it" instead of asking one prompt to do both.
Falls back to the original question if rewriting itself fails, so it never
blocks the pipeline. Re-verified against the exact failing case after the
fix: correct, fully-grounded answer instead of a one-word non-answer.

`--corpus` must point at a directory (or a path that doesn't exist yet --
Chroma creates it); pointing it at an existing file gives a clean error
instead of a raw Chroma stack trace.

**Known architectural limitation, not a bug**: this is a document-grounded
RAG system, not a conversational-memory system. A meta-question about the
conversation itself ("summarize everything we've discussed," "which model
did we cover first?," "how many have we covered?") retrieves from the
document corpus like any other question -- which has no good match for a
question about *the conversation* -- and/or exceeds the capped
`--history-turns` window, so it answers confidently and wrong rather than
recognizing it can't answer that class of question. Tested with an 11-turn
conversation: turns asking about SDLC content were all correct; three
meta-questions at the end were all wrong. A real fix would mean detecting
meta-questions and routing them to a path that reads full conversation
history directly instead of going through retrieval -- a feature addition,
not a patch, so it's documented here rather than rushed.

Concurrent requests to the same Ollama server were tested (two threads
calling `ollama.generate` simultaneously) -- no crash or deadlock, but
Ollama serializes them rather than running true parallel inference, so this
assistant isn't built for concurrent multi-user load.

## Tests

```bash
pytest tests/
```

Covers `benchmark.extract_metrics()` (the ns-to-ms/tok-per-s math) and
`structured_generate.generate_structured()` (validate/retry/fail-gracefully),
with `ollama.generate` mocked -- no Ollama server needed to run these.

## Results

All sweeps write to `results/*.jsonl`, one JSON record per line, flushed
immediately after each call (a crash mid-sweep doesn't lose completed runs).
Reports default to the most recently modified matching file, or take an
explicit path as an argument.
