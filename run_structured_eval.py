"""Phase 2 CLI sweep: runs generate_structured() across models x prompts and
records success/retry/failure diagnostics to a .jsonl results file.

Usage:
    python run_structured_eval.py [--models llama3.2-3b,phi4-mini,mistral-7b] [--out results/foo.jsonl]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark import check_ollama_available, now_iso
from models import MODELS
from schemas import StructuredAnswer
from structured_generate import build_prompt, build_retry_prompt, generate_structured
from structured_prompts import STRUCTURED_PROMPTS


def run_sweep(model_names: list[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0
    total_calls = len(model_names) * len(STRUCTURED_PROMPTS)

    with out_path.open("w") as f:
        for model_name in model_names:
            tag = MODELS[model_name]
            print(f"\n=== {model_name} ({tag}) ===")

            for i, prompt_entry in enumerate(STRUCTURED_PROMPTS, start=1):
                print(f"  [{model_name}] {prompt_entry['id']} ({i}/{len(STRUCTURED_PROMPTS)})")
                result = generate_structured(
                    tag,
                    prompt_entry["question"],
                    schema=StructuredAnswer,
                    prompt_builder=build_prompt,
                    retry_prompt_builder=build_retry_prompt,
                )

                status = "OK" if result["success"] else "FAIL"
                print(f"    {status} in {result['attempts']} attempt(s)")

                record = {
                    "model": model_name,
                    "ollama_tag": tag,
                    "prompt_id": prompt_entry["id"],
                    "question": prompt_entry["question"],
                    "timestamp": now_iso(),
                    **result,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                records_written += 1

    print(f"\nDone. {records_written}/{total_calls} runs recorded to {out_path}")
    return records_written


def main():
    parser = argparse.ArgumentParser(description="Phase 2 structured-output reliability eval")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODELS.keys()),
        help="Comma-separated friendly model names (default: all in models.py)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .jsonl path (default: results/structured_<timestamp>.jsonl)",
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
        out_path = Path("results") / f"structured_{timestamp}.jsonl"

    run_sweep(model_names, out_path)


if __name__ == "__main__":
    main()
