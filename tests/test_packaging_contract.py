import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_pyproject_keeps_heavy_stacks_behind_named_extras():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    default = " ".join(project["dependencies"]).lower()
    assert project["requires-python"] == ">=3.11"
    for heavy in ("torch", "transformers", "pyaudio", "pymilvus", "neo4j", "minio", "dash"):
        assert heavy not in default

    extras = project["optional-dependencies"]
    assert {"audio", "ml", "infra", "viz", "ipc", "dev"} <= set(extras)
    infra = {dependency.lower() for dependency in extras["infra"]}
    assert "pymilvus>=2.3,<2.4" in infra
    assert "marshmallow>=3,<4" in infra
    assert project["scripts"]["cosmic-synapse"] == "a_lmi.cli:main"


def test_dependency_light_cli_doctor_runs_from_checkout():
    result = subprocess.run(
        [sys.executable, "-m", "a_lmi.cli", "doctor", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"core_imports": "ok"' in result.stdout
    assert '"full_stack_ready"' in result.stdout
