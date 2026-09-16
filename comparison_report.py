"""Phase 3: reads a run_comparison.py results file and produces
(1) a console summary table and (2) a written Markdown technical report.

Usage:
    python comparison_report.py [results/comparison_foo.jsonl]   # defaults to most recent
"""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table


def find_latest_results() -> Path:
    results_dir = Path("results")
    files = sorted(results_dir.glob("comparison_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("No comparison results files found in results/", file=sys.stderr)
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


def model_stats(records: list[dict], model: str) -> dict:
    model_records = [r for r in records if r["model"] == model]
    tok_s = [r["tokens_per_sec"] for r in model_records if r.get("tokens_per_sec")]
    vram_mb = [r["size_vram_bytes"] / (1024 * 1024) for r in model_records if r.get("size_vram_bytes")]
    scores = [r["quality_score"] for r in model_records if r.get("quality_score") is not None]
    below_3 = sum(1 for s in scores if s < 3)

    by_category: dict[str, list[int]] = {}
    for r in model_records:
        if r.get("quality_score") is not None:
            by_category.setdefault(r["category"], []).append(r["quality_score"])

    return {
        "n": len(model_records),
        "mean_tok_s": statistics.mean(tok_s) if tok_s else 0.0,
        "mean_vram_mb": statistics.mean(vram_mb) if vram_mb else 0.0,
        "mean_quality": statistics.mean(scores) if scores else 0.0,
        "n_scored": len(scores),
        "n_below_3": below_3,
        "by_category": {
            cat: statistics.mean(vals) for cat, vals in by_category.items()
        },
    }


def print_console_table(console: Console, path: Path, models: list[str], stats: dict[str, dict]) -> None:
    table = Table(title=f"Phase 3 Model Comparison -- {path.name}")
    table.add_column("Model", style="bold")
    table.add_column("Mean tok/s")
    table.add_column("Mean VRAM (MB)")
    table.add_column("Mean quality (1-5)")
    table.add_column("Scored below 3")

    for model in models:
        s = stats[model]
        table.add_row(
            model,
            f"{s['mean_tok_s']:.1f}",
            f"{s['mean_vram_mb']:.0f}",
            f"{s['mean_quality']:.2f}",
            f"{s['n_below_3']}/{s['n_scored']}",
        )
    console.print(table)


def write_markdown_report(path: Path, models: list[str], stats: dict[str, dict], out_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"comparison_report_{timestamp}.md"

    lines = [
        "# Phase 3: Model Comparison Study",
        "",
        f"Source data: `{path.name}`",
        "",
        "*Quality scores are the mean of judgments from the other two models "
        "(never a model judging its own answer) -- a single local LLM judge "
        "showed inconsistent scoring in testing, so this averages across two "
        "independent judges rather than trusting one.*",
        "",
        "## Summary",
        "",
        "| Model | Mean tok/s | Mean VRAM (MB) | Mean quality (1-5) | Scored below 3 |",
        "|---|---|---|---|---|",
    ]
    for model in models:
        s = stats[model]
        lines.append(
            f"| {model} | {s['mean_tok_s']:.1f} | {s['mean_vram_mb']:.0f} | "
            f"{s['mean_quality']:.2f} | {s['n_below_3']}/{s['n_scored']} |"
        )

    lines += ["", "## Per-model analysis", ""]
    for model in models:
        s = stats[model]
        cat_summary = ", ".join(f"{cat}: {v:.2f}" for cat, v in sorted(s["by_category"].items()))
        lines.append(
            f"**{model}** averaged {s['mean_tok_s']:.1f} tokens/sec and "
            f"{s['mean_vram_mb']:.0f}MB VRAM across {s['n']} prompts, with a mean "
            f"quality score of {s['mean_quality']:.2f}/5 ({s['n_below_3']} of "
            f"{s['n_scored']} judged responses scored below 3). By category: "
            f"{cat_summary}."
        )
        lines.append("")

    best_quality = max(models, key=lambda m: stats[m]["mean_quality"]) if models else None
    fastest = max(models, key=lambda m: stats[m]["mean_tok_s"]) if models else None
    smallest = min(models, key=lambda m: stats[m]["mean_vram_mb"] if stats[m]["mean_vram_mb"] else float("inf")) if models else None

    lines += [
        "## Recommendation",
        "",
        f"Highest judged quality: **{best_quality}** "
        f"({stats[best_quality]['mean_quality']:.2f}/5). "
        f"Fastest: **{fastest}** ({stats[fastest]['mean_tok_s']:.1f} tok/s). "
        f"Smallest memory footprint: **{smallest}** ({stats[smallest]['mean_vram_mb']:.0f}MB). "
        "Choose based on which constraint (quality, latency, or memory) matters most "
        "for the target deployment.",
        "",
    ]

    out_path.write_text("\n".join(lines))
    return out_path


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_results()
    print(f"Reading {path}")

    records = load_records(path)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    models = sorted({r["model"] for r in records})
    stats = {model: model_stats(records, model) for model in models}

    console = Console()
    print_console_table(console, path, models, stats)

    md_path = write_markdown_report(path, models, stats, out_dir=path.parent)
    print(f"\nWritten Markdown report: {md_path}")


if __name__ == "__main__":
    main()
