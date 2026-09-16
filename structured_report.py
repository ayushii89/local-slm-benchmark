"""Prints a reliability summary table from a run_structured_eval.py results file.

Usage:
    python structured_report.py [results/structured_foo.jsonl]   # defaults to most recent
"""

import json
import statistics
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


def find_latest_results() -> Path:
    results_dir = Path("results")
    files = sorted(results_dir.glob("structured_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No structured results files found in results/", file=sys.stderr)
        sys.exit(1)
    return files[-1]


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_results()
    print(f"Reading {path}")

    records = load_records(path)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    models = sorted({r["model"] for r in records})

    console = Console()
    table = Table(title=f"Phase 2 Structured Output Reliability -- {path.name}")
    table.add_column("Model", style="bold")
    table.add_column("Success (1st try)")
    table.add_column("Success (after retry)")
    table.add_column("Failed")
    table.add_column("Mean confidence\n(successful only)")

    for model in models:
        model_records = [r for r in records if r["model"] == model]
        total = len(model_records)
        first_try = sum(1 for r in model_records if r["success"] and r["attempts"] == 1)
        after_retry = sum(1 for r in model_records if r["success"] and r["attempts"] > 1)
        failed = sum(1 for r in model_records if not r["success"])
        confidences = [
            r["parsed"]["confidence"]
            for r in model_records
            if r["success"] and r.get("parsed")
        ]
        mean_conf = f"{statistics.mean(confidences):.2f}" if confidences else "n/a"

        table.add_row(
            model,
            f"{first_try}/{total}",
            f"{after_retry}/{total}",
            f"{failed}/{total}",
            mean_conf,
        )

    console.print(table)
    print(f"\n{len(records)} total runs across {len(models)} model(s).")


if __name__ == "__main__":
    main()
