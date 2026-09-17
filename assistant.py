"""Integration: a local, fully-offline RAG assistant.

Reuses ~/rag-project's existing hybrid retriever (semantic + BM25 + RRF +
cross-encoder rerank, already-populated Chroma index) but swaps its cloud
(Groq) generation step for a local Ollama model + Pydantic-validated
structured output (schemas.RAGAnswer), via the same generate_structured()
used throughout Phase 2/3 -- no new retry/validation logic.

Run with the SYSTEM python3 (already has rag-project's langchain/chroma
stack + pydantic/rich; only needed `pip install ollama` there), not this
project's .venv, which intentionally stays lightweight:

    python3 assistant.py "What does the eval say about reranking?"   # one-shot
    python3 assistant.py                                              # interactive
    python3 assistant.py --model mistral-7b --k 8 "..."
"""

import argparse
import sys
import time
from pathlib import Path

RAG_PROJECT_ROOT = Path.home() / "rag-project"
sys.path.insert(0, str(RAG_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import check_ollama_available  # noqa: E402
from models import MODELS  # noqa: E402
from query_rewrite import rewrite_query  # noqa: E402
from rag_prompts import build_rag_prompt, build_rag_retry_prompt  # noqa: E402
from schemas import RAGAnswer  # noqa: E402
from structured_generate import generate_structured  # noqa: E402

from src.generation.chain import format_docs  # noqa: E402
from src.ingestion.store import get_embeddings, get_vector_store  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402


def answer_question(
    retriever: HybridRetriever, tag: str, question: str,
    history: list[tuple[str, str]] | None = None,
) -> dict:
    """Answers `question`, optionally informed by prior (question, answer)
    turns for follow-ups. Pure data -- no printing, so both the CLI
    (print_answer()) and server.py's /api/chat can use it. Returns:
        {
          "success": bool,
          "answer": str | None,
          "citations": list[str],
          "dropped_citations": list[str],
          "confidence": float | None,
          "attempts": int,
          "rewritten_query": str | None,   # None if unchanged / no history
          "n_retrieved": int,
          "retrieve_mode": str,
          "retrieval_ms": float,   # total retrieval time (fuse + rerank)
          "generation_ms": float,
          "error": str | None,
        }

    When history is present, `question` is first rewritten into a
    self-contained query (query_rewrite.rewrite_query) before retrieval and
    generation -- fixes a failure found during hardening where a vague
    follow-up ("what about Agile?") retrieved fine but the small model
    picked the wrong chunk among several plausible ones and stayed
    overconfident about it; decoupling disambiguation from answering fixes
    what a prompt-only instruction on the answer step did not.
    """
    effective_question = rewrite_query(tag, question, history) if history else question
    rewritten_query = effective_question if effective_question != question else None

    documents, trace = retriever.retrieve(effective_question, trace=True)

    if not documents:
        return {
            "success": False, "answer": None, "citations": [], "dropped_citations": [],
            "confidence": None, "attempts": 0, "rewritten_query": rewritten_query,
            "n_retrieved": 0, "retrieve_mode": trace.mode, "retrieval_ms": trace.total_ms,
            "generation_ms": 0.0,
            "error": "No documents in the index at all -- nothing to retrieve from.",
        }

    context = format_docs(documents)

    def prompt_builder(q: str) -> str:
        return build_rag_prompt(q, context, history=history)

    def retry_prompt_builder(q: str, prev: str, err: str) -> str:
        return build_rag_retry_prompt(q, context, prev, err, history=history)

    gen_started = time.perf_counter()
    result = generate_structured(
        tag, effective_question,
        schema=RAGAnswer,
        prompt_builder=prompt_builder,
        retry_prompt_builder=retry_prompt_builder,
    )
    gen_ms = (time.perf_counter() - gen_started) * 1000

    if not result["success"]:
        return {
            "success": False, "answer": None, "citations": [], "dropped_citations": [],
            "confidence": None, "attempts": result["attempts"], "rewritten_query": rewritten_query,
            "n_retrieved": len(documents), "retrieve_mode": trace.mode, "retrieval_ms": trace.total_ms,
            "generation_ms": gen_ms,
            "error": f"{result['error']} -- raw output: {result['raw_response']}",
        }

    parsed = result["parsed"]
    # Dedupe citations, preserving first-seen order -- the model sometimes
    # repeats the same [Source: ...] label when it draws on it more than once.
    raw_citations = list(dict.fromkeys(parsed["citations"]))

    # Validate against the sources actually retrieved: the model isn't always
    # reliable about copying a label verbatim -- seen both shortening a
    # filename that starts with a digit down to just the digit, and keeping
    # a "Source: " / "[Source: ...]" wrapper the raw metadata doesn't have.
    # Pydantic can't catch either -- citations is just list[str] -- so
    # cross-check against ground truth (normalized) instead of trusting
    # prompt compliance alone.
    def normalize(label: str) -> str:
        label = label.strip()
        if label.startswith("[") and label.endswith("]"):
            label = label[1:-1].strip()
        if label.lower().startswith("source:"):
            label = label[len("source:"):].strip()
        return label

    def find_match(citation: str, sources_normalized: dict[str, str]) -> str | None:
        norm = normalize(citation)
        if norm in sources_normalized:
            return sources_normalized[norm]
        # Fallback: substring containment either direction, for a truncated
        # citation like "Agile model (1).pdf" against the real, longer
        # "2 Incremental SDLC Model, ..., Agile model (1).pdf". Guarded by a
        # minimum length so short/generic strings can't spuriously match.
        if len(norm) >= 8:
            for source_norm, source_original in sources_normalized.items():
                if norm.lower() in source_norm.lower() or source_norm.lower() in norm.lower():
                    return source_original
        return None

    real_sources = {doc.metadata.get("source", "unknown") for doc in documents}
    real_sources_normalized = {normalize(s): s for s in real_sources}

    citations, dropped = [], []
    for c in raw_citations:
        match = find_match(c, real_sources_normalized)
        (citations if match else dropped).append(match or c)
    citations = list(dict.fromkeys(citations))

    return {
        "success": True, "answer": parsed["answer"], "citations": citations,
        "dropped_citations": dropped, "confidence": parsed["confidence"],
        "attempts": result["attempts"], "rewritten_query": rewritten_query,
        "n_retrieved": len(documents), "retrieve_mode": trace.mode, "retrieval_ms": trace.total_ms,
        "generation_ms": gen_ms, "error": None,
    }


def print_answer(result: dict) -> None:
    """CLI-only formatting of an answer_question() result dict."""
    if result["rewritten_query"]:
        print(f'(rewritten for retrieval: "{result["rewritten_query"]}")')

    print(f"\nRetrieved {result['n_retrieved']} chunks in {result['retrieval_ms']:.0f}ms "
          f"(mode={result['retrieve_mode']})")

    if not result["success"]:
        print(f"\nFAILED: {result['error']}\n")
        return

    print(f"\nAnswer ({result['attempts']} attempt(s)):\n  {result['answer']}")
    print(f"\nCitations: {', '.join(result['citations']) if result['citations'] else '(none)'}")
    if result["dropped_citations"]:
        print(f"  (dropped unverifiable citation(s) from model output: {result['dropped_citations']})")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Generation: {result['generation_ms']:.0f}ms wall time ({result['attempts']} model call(s))\n")


DEFAULT_HISTORY_TURNS = 3


def main():
    parser = argparse.ArgumentParser(description="Local offline RAG assistant")
    parser.add_argument("question", nargs="?", default=None, help="One-shot question (omit for interactive mode)")
    parser.add_argument("--model", type=str, default="llama3.2-3b", help="Friendly model name (default: llama3.2-3b)")
    parser.add_argument("--k", type=int, default=6, help="Final number of retrieved chunks (default: 6)")
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to a Chroma persist_dir to answer over (default: ~/rag-project/chroma_index)",
    )
    parser.add_argument(
        "--history-turns", type=int, default=DEFAULT_HISTORY_TURNS,
        help=f"Prior Q&A turns to keep for follow-ups in interactive mode (default: {DEFAULT_HISTORY_TURNS}, 0 to disable)",
    )
    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"ERROR: unknown model {args.model!r}. Known: {list(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)
    tag = MODELS[args.model]

    check_ollama_available()

    corpus_dir = Path(args.corpus).expanduser() if args.corpus else RAG_PROJECT_ROOT / "chroma_index"
    if corpus_dir.exists() and not corpus_dir.is_dir():
        print(
            f"ERROR: --corpus {corpus_dir} exists but is not a directory. "
            f"A Chroma persist_dir must be a directory (it doesn't need to exist yet -- "
            f"a new one is created if missing).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading index from {corpus_dir} ...")
    embeddings = get_embeddings()
    db = get_vector_store(persist_dir=str(corpus_dir), embeddings=embeddings)
    from src.retrieval.hybrid import RetrievalConfig
    retriever = HybridRetriever(db, RetrievalConfig(final_k=args.k))
    print(f"Ready. Model: {args.model} ({tag})\n")

    if args.question:
        print_answer(answer_question(retriever, tag, args.question))
        return

    print("Interactive mode. Type a question, or 'quit'/'exit' to stop.\n")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        result = answer_question(retriever, tag, question, history=history or None)
        print_answer(result)
        if result["success"] and args.history_turns > 0:
            history.append((question, result["answer"]))
            history = history[-args.history_turns:]


if __name__ == "__main__":
    main()
