#!/usr/bin/env python3
"""Descriptive Python/Rust/C++ LightToken benchmark on synthetic oracle fixtures."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from a_lmi.core.light_token import LightToken, spectral_similarity

METHODS = ("power_correlation", "cosine", "euclidean")


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "samples": n,
        "median_us": statistics.median(ordered),
        "p95_us": ordered[min(n - 1, (95 * n + 99) // 100 - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-bench", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.iterations <= 200:
        parser.error("--iterations must be 3..200")

    left = LightToken.from_dict(json.loads((args.fixtures / "active_random.json").read_text()))
    right = LightToken.from_dict(json.loads((args.fixtures / "active_sinusoid.json").read_text()))
    native = json.loads(
        subprocess.run(
            [str(args.rust_bench), str(args.fixtures), str(args.iterations)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    python_methods: dict[str, object] = {}
    for method in METHODS:
        score = spectral_similarity(left, right, method)
        native_score = native["methods"][method]["score"]
        if abs(score - native_score) > 1e-4:
            raise ValueError(f"{method}: Python/Rust benchmark parity mismatch: {score} != {native_score}")
        spectral_similarity(left, right, method)
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            spectral_similarity(left, right, method)
            samples.append((time.perf_counter_ns() - start) / 1000.0)
        python_methods[method] = {"score": score, **summarize(samples)}

    report = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
        },
        "input": "synthetic_python_oracle_fixture",
        "iterations": args.iterations,
        "python_token_level": python_methods,
        "native": native,
        "methodology": (
            "One warmup, then individual microsecond samples. Python and Rust token-level "
            "comparisons both derive spectral power and score; Rust scalar and C++ dispatch "
            "are kernel-level paths with different overhead. C++ timing includes dynamic "
            "library loading and symbol validation. No universal performance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "methods": METHODS, "cpp_available": native["cpp_available"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
