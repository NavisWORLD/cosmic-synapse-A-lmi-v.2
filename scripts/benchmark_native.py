#!/usr/bin/env python3
"""Compare Python reference and Rust native library operations without marketing thresholds."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from a_lmi.continuity import (
    append_memory_record,
    export_bundle,
    import_bundle,
    initialize_workspace,
    verify_bundle,
)
from cosmic_synapse.cst_state import CSTState


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int((fraction * len(ordered) + 0.999999)) - 1))
    return ordered[rank]


def summarize(samples: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(samples),
        "median_us": statistics.median(samples) if samples else 0.0,
        "p95_us": percentile(samples, 0.95),
    }


def timed(operation: Callable[[], object]) -> float:
    start = time.perf_counter_ns()
    operation()
    return (time.perf_counter_ns() - start) / 1_000.0


def scan_ledger(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def python_metrics(iterations: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="almi-python-bench-") as temp:
        root = Path(temp)
        metrics: dict[str, object] = {}

        samples = []
        for index in range(iterations):
            workspace = root / f"init-{index}"
            samples.append(
                timed(lambda workspace=workspace, index=index: initialize_workspace(
                    workspace, name="Python Benchmark", seed=index
                ))
            )
        metrics["workspace_init"] = summarize(samples)

        source = root / "source"
        initialize_workspace(source, name="Python Benchmark", seed=7)
        samples = []
        for index in range(iterations):
            bundle = root / f"export-{index}.cosmos"
            samples.append(timed(lambda bundle=bundle: export_bundle(source, bundle)))
        metrics["bundle_export"] = summarize(samples)

        verification_bundle = root / "verify.cosmos"
        export_bundle(source, verification_bundle)
        metrics["bundle_verify"] = summarize(
            [timed(lambda: verify_bundle(verification_bundle)) for _ in range(iterations)]
        )

        samples = []
        for index in range(iterations):
            destination = root / f"import-{index}"
            samples.append(
                timed(lambda destination=destination: import_bundle(verification_bundle, destination))
            )
        metrics["bundle_import"] = summarize(samples)

        memory_workspace = root / "memory"
        initialize_workspace(memory_workspace, name="Memory Benchmark", seed=8)
        samples = []
        for index in range(iterations):
            record = {"version": 1, "role": "user", "content": f"synthetic benchmark record {index}"}
            samples.append(timed(lambda record=record: append_memory_record(memory_workspace, record)))
        metrics["memory_append"] = summarize(samples)

        scan_workspace = root / "scan"
        initialize_workspace(scan_workspace, name="Scan Benchmark", seed=9)
        for index in range(100):
            append_memory_record(
                scan_workspace,
                {"version": 1, "role": "user", "content": f"synthetic benchmark record {index}"},
            )
        ledger = scan_workspace / "memory" / "ledger.jsonl"
        metrics["memory_scan_100"] = summarize(
            [timed(lambda: scan_ledger(ledger)) for _ in range(iterations)]
        )

        state = CSTState(seed=10)
        metrics["state_serialize"] = summarize(
            [timed(state.to_json) for _ in range(iterations)]
        )

        return {
            "implementation": "python-reference-library",
            "iterations": iterations,
            "timing_unit": "microseconds",
            "metrics": metrics,
            "methodology": (
                "Python reference operations timed inside one process; filesystem operations use "
                "synthetic temporary data. memory_scan_100 validates each JSONL line."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-bench", required=True, help="Path to release almi_bench executable")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    iterations = max(3, min(200, args.iterations))

    rust = subprocess.run(
        [args.rust_bench, str(iterations)],
        check=True,
        capture_output=True,
        text=True,
    )
    rust_result = json.loads(rust.stdout)
    result = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "iterations": iterations,
        "python": python_metrics(iterations),
        "rust": rust_result,
        "interpretation": (
            "Measurements are descriptive for this CI machine/run only. No performance threshold "
            "or general Rust/Python superiority claim is inferred."
        ),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
