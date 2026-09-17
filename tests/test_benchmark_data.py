"""Unit tests for benchmark_data.py against small fixture .jsonl files.

No langchain, no FastAPI, no Ollama server -- these run in the lightweight
.venv just like everything else in tests/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import benchmark_data


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(benchmark_data, "RESULTS_DIR", d)
    return d


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_phase1_summary_no_files_returns_unavailable(results_dir):
    result = benchmark_data.get_phase1_summary()
    assert result == {"available": False}


def test_phase1_summary_ignores_prefixed_phase2_phase3_files(results_dir):
    # A structured_*/comparison_* file with a LATER mtime must not be picked
    # up by Phase 1's glob -- this is exactly the bug found and fixed in
    # report.py's find_latest_results() before this module was written.
    write_jsonl(results_dir / "20260101T000000Z.jsonl", [
        {"record_type": "run", "model": "llama3.2-3b", "ttft_ms": 10.0,
         "tokens_per_sec": 50.0, "total_latency_ms": 100.0},
    ])
    write_jsonl(results_dir / "comparison_20260102T000000Z.jsonl", [
        {"model": "llama3.2-3b", "tokens_per_sec": 999.0},
    ])

    result = benchmark_data.get_phase1_summary()

    assert result["available"] is True
    assert result["source_file"] == "20260101T000000Z.jsonl"
    assert result["per_model"]["llama3.2-3b"]["metrics"]["tokens_per_sec"]["mean"] == 50.0


def test_phase2_summary_computes_rates(results_dir):
    write_jsonl(results_dir / "structured_20260101T000000Z.jsonl", [
        {"model": "llama3.2-3b", "success": True, "attempts": 1, "parsed": {"confidence": 0.9}},
        {"model": "llama3.2-3b", "success": True, "attempts": 2, "parsed": {"confidence": 0.7}},
        {"model": "llama3.2-3b", "success": False, "attempts": 2, "parsed": None},
    ])

    result = benchmark_data.get_phase2_summary()

    assert result["available"] is True
    m = result["per_model"]["llama3.2-3b"]
    assert m["total"] == 3
    assert m["first_try"] == 1
    assert m["after_retry"] == 1
    assert m["failed"] == 1
    assert m["mean_confidence"] == pytest.approx(0.8)


def test_phase3_summary_uses_comparison_report_model_stats(results_dir):
    write_jsonl(results_dir / "comparison_20260101T000000Z.jsonl", [
        {"model": "llama3.2-3b", "category": "factual", "tokens_per_sec": 40.0,
         "size_vram_bytes": 2_000_000_000, "quality_score": 4.5},
    ])

    result = benchmark_data.get_phase3_summary()

    assert result["available"] is True
    assert "llama3.2-3b" in result["per_model"]
    assert result["per_model"]["llama3.2-3b"]["n"] == 1
