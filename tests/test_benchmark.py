"""Regression tests for benchmark measurement isolation."""

import importlib.util
import sys
import tracemalloc
import weakref
from pathlib import Path

import pytest

_HELPER = Path(__file__).parents[1] / "scripts" / "benchmark_support.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_support", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
support = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = support
_SPEC.loader.exec_module(support)


def test_bench_releases_previous_output_before_next_call():
    class Output:
        pass

    references = []

    def allocate():
        assert all(reference() is None for reference in references)
        result = Output()
        references.append(weakref.ref(result))
        return result

    _, last = support.bench(allocate, warmup=2, repeats=3)
    assert len(references) == 5
    assert references[-1]() is last


def test_bench_even_repeats_use_statistical_median(monkeypatch):
    times = iter([0.0, 0.002, 1.0, 1.004])
    monkeypatch.setattr(support.time, "perf_counter", lambda: next(times))
    elapsed, _ = support.bench(lambda: None, warmup=0, repeats=2)
    assert elapsed == pytest.approx(3.0)


@pytest.mark.parametrize(("warmup", "repeats"), [(-1, 1), (0, 0)])
def test_bench_rejects_invalid_repetition_counts(warmup, repeats):
    with pytest.raises(ValueError, match=r"warmup.*repeats"):
        support.bench(lambda: None, warmup=warmup, repeats=repeats)


def test_heap_tracing_stops_when_workload_fails():
    def fail():
        msg = "workload failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="workload failed"):
        support.python_heap_peak_mb(fail)
    assert not tracemalloc.is_tracing()
