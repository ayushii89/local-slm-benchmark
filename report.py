"""Prints a summary table (mean/median/min/max) from a benchmark.py results file.

Usage:
    python report.py [results/foo.jsonl]   # defaults to most recent file in results/
"""

import json
import statistics
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

METRICS = ["ttft_ms", "tokens_per_sec", "total_latency_ms"]


def find_latest_results() -> Path:
    # Phase 1's default filenames are bare timestamps ("20260916T124147Z.jsonl"),
    # unlike Phase 2/3's "structured_*"/"comparison_*" prefixes. An unscoped
    # "*.jsonl" glob here would match those too, and since they share field
    # names (ttft_ms, tokens_per_sec), a mismatch wouldn't even crash -- it
    # would silently produce a Phase 1 report built from Phase 3 data.
    results_dir = Path("results")
    files = sorted(results_dir.glob("[0-9]*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No results files found in results/", file=sys.stderr)
        sys.exit(1)
    return files[-1]


def load_records(path: Path) -> tuple[list[dict], list[dict]]:
    runs, cold_starts = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") == "cold_start":
                cold_starts.append(record)
            else:
                runs.append(record)
    return runs, cold_starts


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_results()
    print(f"Reading {path}")

    runs, cold_starts = load_records(path)
    if not runs:
        print("No run records found.", file=sys.stderr)
        sys.exit(1)

    models = sorted({r["model"] for r in runs})
    cold_start_by_model = {
        c["model"]: c["load_duration_ms"] for c in cold_starts
    }

    console = Console()
    table = Table(title=f"Phase 1 Inference Benchmark -- {path.name}")
    table.add_column("Model", style="bold")
    for metric in METRICS:
        table.add_column(f"{metric}\n(mean / median / min-max)")
    table.add_column("Cold-start load (ms)")

    for model in models:
        model_runs = [r for r in runs if r["model"] == model]
        row = [model]
        for metric in METRICS:
            values = [r[metric] for r in model_runs if metric in r]
            s = summarize(values)
            row.append(
                f"{s['mean']:.1f} / {s['median']:.1f} / {s['min']:.1f}-{s['max']:.1f}"
            )
        cold_start = cold_start_by_model.get(model)
        row.append(f"{cold_start:.1f}" if cold_start is not None else "n/a")
        table.add_row(*row)

    console.print(table)
    print(f"\n{len(runs)} total runs across {len(models)} model(s).")


if __name__ == "__main__":
    main()
