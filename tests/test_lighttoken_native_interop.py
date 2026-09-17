from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_lighttoken_fixtures.py"


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    output = tmp_path / "lighttoken-fixtures"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output


@pytest.fixture()
def native_cli() -> Path:
    configured = os.environ.get("LIGHTTOKEN_NATIVE_CLI")
    assert configured, "LIGHTTOKEN_NATIVE_CLI must point to the native lighttoken executable"
    path = Path(configured)
    assert path.is_file(), f"native LightToken CLI does not exist: {path}"
    return path


def run_cli(native_cli: Path, *args: str) -> dict:
    result = subprocess.run(
        [str(native_cli), *args, "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_generator_is_deterministic_and_synthetic(fixture_dir: Path, tmp_path: Path):
    second = tmp_path / "second"
    subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(second)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    first_manifest = json.loads((fixture_dir / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["synthetic_only"] is True
    assert first_manifest["embedding_dimension"] == 1536
    assert first_manifest["spectral_dimension"] == 769
    for name, expected in first_manifest["files"].items():
        payload = (fixture_dir / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]


def test_rust_cli_validates_python_fixture(native_cli: Path, fixture_dir: Path):
    payload = run_cli(native_cli, "validate", str(fixture_dir / "active_random.json"))
    assert payload["valid"] is True
    assert payload["embedding_dimension"] == 1536
    assert payload["spectral_dimension"] == 769


def test_rust_cli_inspects_and_computes_python_spectrum(native_cli: Path, fixture_dir: Path):
    inspected = run_cli(native_cli, "inspect", str(fixture_dir / "active_sinusoid.json"))
    assert inspected["token_id"] == "00000000-0000-0000-0000-000000000004"
    assert inspected["spectral_transform"] == "embedding_rfft"
    spectrum = run_cli(native_cli, "spectrum", str(fixture_dir / "active_sinusoid.json"))
    assert spectrum["power_length"] == 769
    assert 0 <= spectrum["dominant_bin"] < 769
    assert spectrum["dominant_magnitude"] >= 0.0


def test_rust_cli_matches_python_similarity_scores(native_cli: Path, fixture_dir: Path):
    oracle = json.loads((fixture_dir / "similarity.json").read_text())
    selected = [
        pair
        for pair in oracle["pairs"]
        if pair["a"] == "active_sinusoid" and pair["b"] == "active_random"
    ]
    assert len(selected) == 3
    for pair in selected:
        payload = run_cli(
            native_cli,
            "compare",
            str(fixture_dir / f"{pair['a']}.json"),
            str(fixture_dir / f"{pair['b']}.json"),
            "--method",
            pair["method"],
        )
        assert payload["method"] == pair["method"]
        assert payload["score"] == pytest.approx(pair["score"], rel=2e-5, abs=2e-4)


def test_rust_cli_reads_historical_magnitude_phase_fixture(native_cli: Path, fixture_dir: Path):
    payload = run_cli(native_cli, "inspect", str(fixture_dir / "historical_magnitude_phase.json"))
    assert payload["token_id"] == "00000000-0000-0000-0000-000000000099"
    assert payload["spectral_dimension"] == 1536
    assert payload["spectral_transform"] == "legacy_embedding_fft"


def test_rust_canonical_round_trip_matches_python_bytes(native_cli: Path, fixture_dir: Path, tmp_path: Path):
    source = fixture_dir / "active_random.json"
    output = tmp_path / "rust-round-trip.json"
    payload = run_cli(native_cli, "inspect", str(source), "--write-canonical", str(output))
    assert payload["canonical_written"] is True
    assert output.read_bytes() == source.read_bytes()
