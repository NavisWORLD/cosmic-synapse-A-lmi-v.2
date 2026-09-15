"""Dependency-light command line interface for the restored active package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Sequence


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def doctor_report() -> dict:
    """Return an inspectable capability report without opening services/hardware."""

    core_errors: list[str] = []
    try:
        import a_lmi  # noqa: F401
        from a_lmi.core.light_token import LightToken  # noqa: F401
        from cosmic_synapse.cst_state import CSTEngine  # noqa: F401
    except Exception as exc:  # doctor reports failures rather than hiding them
        core_errors.append(f"{type(exc).__name__}: {exc}")

    optional = {
        "audio_capture": _module_available("pyaudio"),
        "audio_analysis": _module_available("librosa") and _module_available("soundfile"),
        "speech_vosk": _module_available("vosk"),
        "ml_torch": _module_available("torch"),
        "ml_transformers": _module_available("transformers"),
        "kafka": _module_available("kafka"),
        "minio": _module_available("minio"),
        "milvus": _module_available("pymilvus"),
        "neo4j": _module_available("neo4j"),
        "websockets": _module_available("websockets"),
        "plotly": _module_available("plotly"),
        "dash": _module_available("dash"),
    }
    full_stack_keys = ("kafka", "minio", "milvus", "neo4j")
    full_stack_ready = not core_errors and all(optional[key] for key in full_stack_keys)
    return {
        "core_imports": "ok" if not core_errors else "error",
        "core_errors": core_errors,
        "full_stack_ready": full_stack_ready,
        "optional_dependencies": optional,
        "note": (
            "Availability means the Python dependency is importable; it does not prove "
            "that external services, model weights, audio devices, SDR hardware, or Unity are ready."
        ),
    }


def _doctor(args: argparse.Namespace) -> int:
    report = doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("COSMIC SYNAPSE // A-LMI doctor")
        print(f"core imports: {report['core_imports']}")
        print(f"full infrastructure dependency set: {report['full_stack_ready']}")
        for name, available in report["optional_dependencies"].items():
            print(f"  {name:20} {'available' if available else 'not installed'}")
        if report["core_errors"]:
            for error in report["core_errors"]:
                print(f"core error: {error}", file=sys.stderr)
    return 0 if report["core_imports"] == "ok" else 1


def _cst_demo(args: argparse.Namespace) -> int:
    from cosmic_synapse.cst_state import CSTEngine, CSTEvent

    engine = CSTEngine(seed=args.seed)
    events = [
        CSTEvent(
            dt=args.dt,
            omega=args.omega,
            audio_energy=args.audio_energy,
            neighbor_phases=(),
        )
        for _ in range(args.steps)
    ]
    snapshots = engine.replay(events)
    result = snapshots[-1] if snapshots else engine.state.snapshot()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmic-synapse",
        description="Dependency-light utilities for the COSMIC SYNAPSE / A-LMI restoration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Report core imports and optional subsystem dependencies."
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    doctor.set_defaults(handler=_doctor)

    cst = subparsers.add_parser(
        "cst-demo", help="Run the canonical deterministic CST software-state adapter."
    )
    cst.add_argument("--seed", type=int, default=2026)
    cst.add_argument("--steps", type=int, default=10)
    cst.add_argument("--dt", type=float, default=0.01)
    cst.add_argument("--omega", type=float, default=0.5)
    cst.add_argument("--audio-energy", type=float, default=0.0)
    cst.set_defaults(handler=_cst_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
