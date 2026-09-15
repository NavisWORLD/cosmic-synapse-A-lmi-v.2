"""Measured local software benchmarks for reproducibility and soak checks.

This module reports observations from the current machine. It defines no
performance threshold and makes no hardware, production, or comparative claim.
"""

from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path
from typing import Any

from cosmic_synapse.cst_state import CSTEngine, CSTEvent

from .continuity import (
    append_memory_record,
    export_bundle,
    import_bundle,
    initialize_workspace,
    inspect_workspace,
    verify_bundle,
)


def _rate(operations: int, elapsed: float) -> float:
    if elapsed <= 0.0:
        return 0.0
    return float(operations) / elapsed


def benchmark_core(*, iterations: int = 1000, records: int = 100, seed: int = 2026) -> dict[str, Any]:
    """Measure deterministic CST and portable-continuity operations locally.

    ``iterations`` and ``records`` are bounded by the caller/environment rather
    than interpreted as a benchmark target. The returned values are evidence of
    this execution only.
    """

    iterations = int(iterations)
    records = int(records)
    seed = int(seed)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if records < 0:
        raise ValueError("records must be non-negative")
    if iterations > 10_000_000:
        raise ValueError("iterations exceeds bounded local benchmark limit")
    if records > 1_000_000:
        raise ValueError("records exceeds bounded local benchmark limit")

    errors = 0

    engine = CSTEngine(seed=seed)
    event = CSTEvent(dt=0.01, omega=0.5, audio_energy=0.1, neighbor_phases=())
    cst_start = time.perf_counter()
    snapshots = engine.replay(event for _ in range(iterations))
    cst_elapsed = time.perf_counter() - cst_start
    if len(snapshots) != iterations or engine.state.step_index != iterations:
        errors += 1

    continuity_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cosmic-benchmark-") as tmp:
        root = Path(tmp)
        source = root / "source"
        restored = root / "restored"
        bundle = root / "state.cosmos"
        initialize_workspace(source, name="Benchmark Cosmos", seed=seed)
        for index in range(records):
            append_memory_record(
                source,
                {
                    "role": "benchmark",
                    "index": index,
                    "content": f"record-{index}",
                },
            )
        export_bundle(source, bundle)
        verified = verify_bundle(bundle)
        import_bundle(bundle, restored)
        restored_summary = inspect_workspace(restored)
        bundle_bytes = bundle.stat().st_size
        if not verified["valid"] or restored_summary["memory_records"] != records:
            errors += 1
    continuity_elapsed = time.perf_counter() - continuity_start

    return {
        "classification": "local-software-benchmark",
        "note": (
            "Measured on the current execution environment only; no production, "
            "comparative, hardware, or minimum-performance claim is implied."
        ),
        "seed": seed,
        "errors": errors,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "cst": {
            "operations": iterations,
            "elapsed_seconds": cst_elapsed,
            "ops_per_second": _rate(iterations, cst_elapsed),
            "final_step": engine.state.step_index,
        },
        "continuity": {
            "records": records,
            "elapsed_seconds": continuity_elapsed,
            "operations_per_second": _rate(max(records, 1), continuity_elapsed),
            "verified": bool(verified["valid"]),
            "restored_memory_records": int(restored_summary["memory_records"]),
            "bundle_bytes": int(bundle_bytes),
        },
    }
