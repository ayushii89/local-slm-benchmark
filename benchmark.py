"""Phase 1 benchmarking harness: measures raw inference performance
(TTFT, tokens/sec, total latency) for local models served via Ollama.

Usage:
    python benchmark.py [--models llama3.2-3b,phi4-mini,mistral-7b] [--repeats 3] [--out results/foo.jsonl]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import ollama

from models import MODELS
from prompts import PHASE1_PROMPTS

WARMUP_PROMPT = "Say hello."


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_metrics(response: dict) -> dict:
    total_duration = response.get("total_duration", 0)
    load_duration = response.get("load_duration", 0)
    prompt_eval_count = response.get("prompt_eval_count", 0)
    prompt_eval_duration = response.get("prompt_eval_duration", 0)
    eval_count = response.get("eval_count", 0)
    eval_duration = response.get("eval_duration", 0)

    ttft_ms = prompt_eval_duration / 1e6
    tokens_per_sec = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0.0
    total_latency_ms = total_duration / 1e6
    load_duration_ms = load_duration / 1e6

    return {
        "ttft_ms": round(ttft_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "load_duration_ms": round(load_duration_ms, 3),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def run_sweep(model_names: list[str], repeats: int, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    total_calls = len(model_names) * len(PHASE1_PROMPTS) * repeats

    with out_path.open("w") as f:
        for model_name in model_names:
            tag = MODELS[model_name]
            print(f"\n=== {model_name} ({tag}) ===")

            # Warm-up call: forces model load outside the timed loop.
            try:
                warmup = ollama.generate(model=tag, prompt=WARMUP_PROMPT, stream=False)
                warmup_metrics = extract_metrics(warmup)
                print(f"  warm-up: load_duration_ms={warmup_metrics['load_duration_ms']:.1f}")
                cold_start_record = {
                    "model": model_name,
                    "ollama_tag": tag,
                    "record_type": "cold_start",
                    "load_duration_ms": warmup_metrics["load_duration_ms"],
                    "timestamp": now_iso(),
                }
                f.write(json.dumps(cold_start_record) + "\n")
                f.flush()
            except Exception as e:
                print(f"  ERROR during warm-up for {model_name}: {e}", file=sys.stderr)
                print(f"  Skipping {model_name} entirely.", file=sys.stderr)
                continue

            call_idx = 0
            for prompt_entry in PHASE1_PROMPTS:
                for repeat in range(1, repeats + 1):
                    call_idx += 1
                    print(
                        f"  [{model_name}] prompt {prompt_entry['id']} "
                        f"repeat {repeat}/{repeats} ({call_idx}/{len(PHASE1_PROMPTS) * repeats})"
                    )
                    try:
                        response = ollama.generate(
                            model=tag, prompt=prompt_entry["prompt"], stream=False
                        )
                    except Exception as e:
                        print(f"    ERROR: {e} -- skipping this run", file=sys.stderr)
                        continue

                    metrics = extract_metrics(response)
                    record = {
                        "model": model_name,
                        "ollama_tag": tag,
                        "record_type": "run",
                        "prompt_id": prompt_entry["id"],
                        "category": prompt_entry["category"],
                        "repeat": repeat,
                        "timestamp": now_iso(),
                        **metrics,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    records_written += 1

    print(f"\nDone. {records_written}/{total_calls} runs recorded to {out_path}")
    return records_written


def check_ollama_available() -> None:
    try:
        ollama.list()
    except Exception as e:
        print(
            "ERROR: could not reach the Ollama server.\n"
            "  Make sure it's running: `brew services start ollama` or `ollama serve`.\n"
            f"  Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 SLM inference benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated friendly model names (default: all in models.py)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per prompt (default: 3)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .jsonl path (default: results/<timestamp>.jsonl)",
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
        out_path = Path("results") / f"{timestamp}.jsonl"

    run_sweep(model_names, args.repeats, out_path)


if __name__ == "__main__":
    main()
