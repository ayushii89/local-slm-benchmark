"""Pure, JSON-serializable summaries of the latest Phase 1/2/3 results.

No langchain, no FastAPI -- reuses report.py/structured_report.py/
comparison_report.py's load/aggregate functions (not their find_latest_results(),
which calls sys.exit(1) on no files -- fine for a CLI, not for a server
process). Testable under the lightweight .venv; imported by both server.py
(the FastAPI app) and its own test file.
"""

from pathlib import Path

import comparison_report
import report
import structured_report

RESULTS_DIR = Path("results")


def _latest(glob_pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def get_phase1_summary() -> dict:
    path = _latest("[0-9]*.jsonl")
    if path is None:
        return {"available": False}

    runs, cold_starts = report.load_records(path)
    if not runs:
        return {"available": False, "source_file": path.name}

    cold_start_by_model = {c["model"]: c["load_duration_ms"] for c in cold_starts}
    models = sorted({r["model"] for r in runs})

    per_model = {}
    for model in models:
        model_runs = [r for r in runs if r["model"] == model]
        metrics = {}
        for metric in report.METRICS:
            values = [r[metric] for r in model_runs if metric in r]
            if values:
                metrics[metric] = report.summarize(values)
        per_model[model] = {
            "metrics": metrics,
            "cold_start_ms": cold_start_by_model.get(model),
            "n": len(model_runs),
        }

    return {"available": True, "source_file": path.name, "per_model": per_model}


def get_phase2_summary() -> dict:
    path = _latest("structured_*.jsonl")
    if path is None:
        return {"available": False}

    records = structured_report.load_records(path)
    if not records:
        return {"available": False, "source_file": path.name}

    models = sorted({r["model"] for r in records})
    per_model = {}
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        total = len(model_records)
        first_try = sum(1 for r in model_records if r["success"] and r["attempts"] == 1)
        after_retry = sum(1 for r in model_records if r["success"] and r["attempts"] > 1)
        failed = sum(1 for r in model_records if not r["success"])
        confidences = [
            r["parsed"]["confidence"] for r in model_records if r["success"] and r.get("parsed")
        ]
        per_model[model] = {
            "total": total,
            "first_try": first_try,
            "after_retry": after_retry,
            "failed": failed,
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        }

    return {"available": True, "source_file": path.name, "per_model": per_model}


def get_phase3_summary() -> dict:
    path = _latest("comparison_*.jsonl")
    if path is None:
        return {"available": False}

    records = comparison_report.load_records(path)
    if not records:
        return {"available": False, "source_file": path.name}

    models = sorted({r["model"] for r in records})
    per_model = {model: comparison_report.model_stats(records, model) for model in models}

    return {"available": True, "source_file": path.name, "per_model": per_model}
