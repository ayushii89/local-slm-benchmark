"""Unit tests for benchmark.extract_metrics() -- the ns-to-ms / tokens-per-sec math."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import extract_metrics


def test_extract_metrics_basic_computation():
    response = {
        "total_duration": 5_000_000_000,      # 5s
        "load_duration": 1_000_000_000,       # 1s
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200_000_000,  # 0.2s
        "eval_count": 100,
        "eval_duration": 2_000_000_000,       # 2s -> 50 tok/s
    }
    metrics = extract_metrics(response)

    assert metrics["ttft_ms"] == 200.0
    assert metrics["tokens_per_sec"] == 50.0
    assert metrics["total_latency_ms"] == 5000.0
    assert metrics["load_duration_ms"] == 1000.0
    assert metrics["prompt_eval_count"] == 10
    assert metrics["eval_count"] == 100


def test_extract_metrics_zero_eval_duration_does_not_raise():
    response = {
        "total_duration": 1_000_000_000,
        "load_duration": 0,
        "prompt_eval_count": 5,
        "prompt_eval_duration": 100_000_000,
        "eval_count": 0,
        "eval_duration": 0,
    }
    metrics = extract_metrics(response)

    assert metrics["tokens_per_sec"] == 0.0


def test_extract_metrics_missing_fields_default_to_zero():
    metrics = extract_metrics({})

    assert metrics["ttft_ms"] == 0.0
    assert metrics["tokens_per_sec"] == 0.0
    assert metrics["total_latency_ms"] == 0.0
    assert metrics["load_duration_ms"] == 0.0
    assert metrics["prompt_eval_count"] == 0
    assert metrics["eval_count"] == 0
