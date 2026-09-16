# local-slm-benchmark

[![Tests](https://github.com/ayushii89/local-slm-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/ayushii89/local-slm-benchmark/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A local, fully-offline RAG assistant -- built on a rigorous evaluation of
*which* small language model should power it. Everything runs through
[Ollama](https://ollama.com) on-device: no cloud API calls, no data leaving
the machine.

**Models compared**: `llama3.2:3b` · `phi4-mini` · `mistral:7b`

**The short version**: 3 phases of testing (speed, reliability, then a full
comparison), followed by a working assistant built on the winner.
`llama3.2:3b` won -- fastest and smallest, and just as good quality-wise as
the other two on this test set. The assistant defaults to it.

| | [Phase 1: Speed](#phase-1-speed) | [Phase 2: Reliability](#phase-2-reliability) | [Phase 3: Comparison](#phase-3-comparison) | [The Assistant](#the-assistant) |
|---|---|---|---|---|
| llama3.2-3b | 53.6 tok/s | 8/8 valid | 34.2 tok/s · 2.2GB · 4.71/5 | ✅ default |
| phi4-mini | 30.2 tok/s | 8/8 valid | 29.6 tok/s · 2.7GB · 4.65/5 | `--model phi4-mini` |
| mistral-7b | 15.2 tok/s | 8/8 valid | 15.8 tok/s · 4.5GB · 4.75/5 | `--model mistral-7b` |

Full technical report: [results/comparison_report_20260916T171703Z.md](results/comparison_report_20260916T171703Z.md)

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

## Phase 1: Speed

How fast does each model actually respond? Measures tokens/sec, time-to-first-token,
and total latency -- read straight from Ollama's own timing data, not a stopwatch
around the API call.

```bash
python benchmark.py          # run all 3 models
python report.py             # see the results as a table
```

## Phase 2: Reliability

Can each model reliably return well-formed, structured data (not just free
text)? Forces JSON output, validates it against a schema, and gives the model
one retry before giving up cleanly.

```bash
python run_structured_eval.py
python structured_report.py   # success rate per model
```

## Phase 3: Comparison

The full picture: speed + memory usage + answer quality, across 40 varied
questions. Quality is judged by having each model's answers scored by the
*other two* models (never judging itself), so no single model's opinion
decides the outcome.

```bash
python run_comparison.py --models llama3.2-3b   # try one model first (faster)
python run_comparison.py                        # all 3 (slow, several minutes)
python comparison_report.py                      # table + writes a full report
```

## The Assistant

The point of all that testing: a working command-line assistant that answers
questions grounded in real documents, using the model the testing recommended.

```bash
python3 assistant.py "What is the Spiral model?"   # one-shot
python3 assistant.py                                # chat interactively
python3 assistant.py --model mistral-7b "..."       # try a different model
```

It answers only from retrieved documents (no making things up -- if the
documents don't have the answer, it says so), cites its sources, remembers
recent turns of the conversation for natural follow-ups, and reuses an
existing retrieval pipeline from a separate project (`~/rag-project`) rather
than rebuilding one from scratch.

> Run this one with plain `python3` (not the `.venv`) -- it needs a
> retrieval stack (`langchain`/`chroma`) that's already installed system-wide
> from the other project, and keeping it out of `.venv` keeps this repo's own
> dependencies light.

## Tests

```bash
pytest tests/
```

14 tests, no Ollama server required (the model calls are mocked). Runs
automatically on every push via GitHub Actions.

---

## Engineering notes: what actually broke, and how it got fixed

The short version above is the pitch. This section is the detail, for anyone
who wants to see the actual debugging, not just the finished product.

**A prompt bug broke structured output for one model entirely.** Early on,
dumping the raw JSON Schema into the prompt made `mistral-7b` echo the schema
back instead of answering it -- 0/8 valid responses. Replacing the schema
dump with a plain field list plus one example fixed it completely (8/8,
first try, across all three models).

**A tag-matching bug made memory readings silently wrong.** Ollama reports
running models with a `:latest` suffix (`phi4-mini:latest`) even when asked
for a bare name (`phi4-mini`); an exact-match check meant `phi4-mini`'s
memory usage always read as 0. Fixed by also matching on the name with the
tag stripped.

**A single LLM judge gave inconsistent scores** -- including once scoring a
factually correct answer as wrong, with a self-contradictory explanation.
Fixed by having each model's answers judged by the *other two* models only,
averaging their scores, so no one judge's inconsistency (or self-judging
bias) decides the result.

**Citations needed a second layer of checking beyond the schema.** A model
can return a citation that's valid *JSON* but not a *real* source -- a
Pydantic schema can't catch that. The assistant now cross-checks every
citation against the documents actually retrieved, dropping and flagging
anything that doesn't match (handling both wrapped labels like
`"Source: ..."` and truncated filenames).

**A vague follow-up question could get a wrong, overconfident answer.**
"What about Agile?" after a few turns about a different topic retrieved the
right documents just fine, but the model sometimes picked the wrong fact
among several in the context and answered at full confidence anyway. A
prompt instruction alone didn't reliably fix this -- confirmed by testing
that a bigger model (`mistral-7b`) handled the same case correctly, meaning
it was a capability limit, not a wording problem. The real fix: rewrite
ambiguous follow-ups into a self-contained question *before* retrieval,
using the conversation so far, so retrieval and answering aren't both
guessing at the same time.

**One thing that's a real limitation, not a bug**: this assistant answers
questions *about the documents*, not questions *about the conversation
itself*. Asking it to "summarize everything we've discussed" gets treated
like any other document search -- and document search has nothing useful to
return for a question about the conversation, so it answers confidently and
wrong. Fixing this properly means detecting that class of question and
routing it to conversation history directly, which is a real feature to
build, not a quick patch -- so for now it's documented rather than papered
over with a half-fix.

## Results

All runs write to `results/*.jsonl`, one line per prompt, saved as it goes
(a crash mid-run doesn't lose what's already done). Reports read the most
recent matching file by default, or you can point at a specific one.
