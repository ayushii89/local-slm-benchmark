"""Phase 3 CLI sweep: for each model x standardized prompt, measures tokens/sec,
peak memory usage, and LLM-judged output quality.

Quality judging uses the *other two* models as judges (never a model judging
its own answer), and quality_score is the mean of the judges that returned a
valid score -- this avoids both self-judging bias and relying on a single
judge's (sometimes inconsistent) call.

Usage:
    python run_comparison.py [--models llama3.2-3b,phi4-mini,mistral-7b] [--out results/foo.jsonl]

Note: a full 3-model x 40-prompt sweep is substantial (120 generations + up
to 240 judge calls, each with up to 2 judges). Test with a single model
first (--models llama3.2-3b) before running the full set.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import ollama

from benchmark import check_ollama_available, extract_metrics, now_iso
from comparison_prompts import COMPARISON_PROMPTS
from judge_prompts import build_judge_prompt, build_judge_retry_prompt
from memory_utils import PeakMemoryTracker
from models import MODELS
from schemas import QualityScore
from structured_generate import generate_structured


def judge_answer(judge_tag: str, question: str, answer: str) -> dict:
    def prompt_builder(q: str) -> str:
        return build_judge_prompt(q, answer)

    def retry_prompt_builder(q: str, prev: str, err: str) -> str:
        return build_judge_retry_prompt(q, answer, prev, err)

    return generate_structured(
        judge_tag,
        question,
        schema=QualityScore,
        prompt_builder=prompt_builder,
        retry_prompt_builder=retry_prompt_builder,
    )


def judge_with_other_models(candidate_model: str, question: str, answer: str) -> list[dict]:
    """Judges `answer` with every model in MODELS except `candidate_model`."""
    judge_names = [name for name in MODELS if name != candidate_model]
    results = []
    for judge_name in judge_names:
        judge_tag = MODELS[judge_name]
        judge_result = judge_answer(judge_tag, question, answer)
        results.append(
            {
                "judge_model": judge_name,
                "score": judge_result["parsed"]["score"] if judge_result["success"] else None,
                "justification": (
                    judge_result["parsed"]["justification"]
                    if judge_result["success"]
                    else judge_result["error"]
                ),
                "success": judge_result["success"],
            }
        )
    return results


def run_sweep(model_names: list[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    total_calls = len(model_names) * len(COMPARISON_PROMPTS)

    with out_path.open("w") as f:
        for model_name in model_names:
            tag = MODELS[model_name]
            judge_names = [name for name in MODELS if name != model_name]
            print(f"\n=== {model_name} ({tag}) -- judges: {judge_names} ===")

            for i, prompt_entry in enumerate(COMPARISON_PROMPTS, start=1):
                print(f"  [{model_name}] {prompt_entry['id']} ({i}/{len(COMPARISON_PROMPTS)})")
                try:
                    with PeakMemoryTracker(tag) as tracker:
                        response = ollama.generate(model=tag, prompt=prompt_entry["prompt"], stream=False)
                except Exception as e:
                    print(f"    ERROR generating: {e} -- skipping", file=sys.stderr)
                    continue

                answer_text = response.get("response", "")
                speed_metrics = extract_metrics(response)
                memory = tracker.peak

                judge_scores = judge_with_other_models(model_name, prompt_entry["prompt"], answer_text)
                successful_scores = [j["score"] for j in judge_scores if j["success"]]
                quality_score = statistics.mean(successful_scores) if successful_scores else None

                record = {
                    "model": model_name,
                    "ollama_tag": tag,
                    "prompt_id": prompt_entry["id"],
                    "category": prompt_entry["category"],
                    "timestamp": now_iso(),
                    "tokens_per_sec": speed_metrics["tokens_per_sec"],
                    "ttft_ms": speed_metrics["ttft_ms"],
                    "total_latency_ms": speed_metrics["total_latency_ms"],
                    "size_bytes": memory["size_bytes"],
                    "size_vram_bytes": memory["size_vram_bytes"],
                    "answer_text": answer_text,
                    "judge_scores": judge_scores,
                    "quality_score": quality_score,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                records_written += 1

    print(f"\nDone. {records_written}/{total_calls} runs recorded to {out_path}")
    return records_written


def main():
    parser = argparse.ArgumentParser(description="Phase 3 model comparison study")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated friendly model names to evaluate (default: all in models.py)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .jsonl path (default: results/comparison_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    check_ollama_available()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_names if m not in MODELS]
    if unknown:
        print(f"ERROR: unknown model name(s) {unknown}. Known: {list(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("results") / f"comparison_{timestamp}.jsonl"

    run_sweep(model_names, out_path)


if __name__ == "__main__":
    main()
