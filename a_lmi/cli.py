"""Dependency-light product CLI for COSMIC SYNAPSE / A-LMI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from typing import Any, Sequence

from .continuity import (
    ContinuityError,
    export_bundle,
    import_bundle,
    initialize_workspace,
    inspect_workspace,
    verify_bundle,
)
from .providers import DEFAULT_OLLAMA_ENDPOINT, OllamaProvider, ProviderError
from .runtime import PersistentRuntime


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _emit(payload: Any, *, json_mode: bool, title: str | None = None) -> None:
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if title:
        print(title)
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, sort_keys=True)
            else:
                rendered = value
            print(f"{key}: {rendered}")
    else:
        print(payload)


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


def _init_workspace(args: argparse.Namespace) -> int:
    result = initialize_workspace(args.path, name=args.name, seed=args.seed)
    _emit(result, json_mode=args.json, title="COSMIC SYNAPSE workspace initialized")
    return 0


def _inspect_workspace(args: argparse.Namespace) -> int:
    result = inspect_workspace(args.path)
    _emit(result, json_mode=args.json, title="COSMIC SYNAPSE continuity status")
    return 0


def _export_workspace(args: argparse.Namespace) -> int:
    result = export_bundle(args.path, args.bundle)
    _emit(result, json_mode=args.json, title="Portable continuity bundle exported")
    return 0


def _verify_bundle(args: argparse.Namespace) -> int:
    result = verify_bundle(args.bundle)
    _emit(result, json_mode=args.json, title="Portable continuity bundle verified")
    return 0


def _import_workspace(args: argparse.Namespace) -> int:
    result = import_bundle(args.bundle, args.path)
    _emit(result, json_mode=args.json, title="Portable continuity bundle imported")
    return 0


def _providers(args: argparse.Namespace) -> int:
    result = {
        "network_checked": False,
        "providers": [
            {
                "provider_id": "ollama",
                "status": "UNCONFIGURED",
                "default_endpoint": DEFAULT_OLLAMA_ENDPOINT,
                "capabilities": ["text"],
                "note": (
                    "The client is available in the core package. No service or model availability "
                    "is claimed until an explicit run/health request contacts Ollama."
                ),
            }
        ],
    }
    _emit(result, json_mode=args.json, title="Model providers")
    return 0


def _run(args: argparse.Namespace) -> int:
    try:
        if args.provider != "ollama":
            raise ProviderError(f"unsupported provider: {args.provider}")
        provider = OllamaProvider(
            model_id=args.model,
            endpoint=args.endpoint,
            revision=args.revision,
            timeout=args.timeout,
            retries=args.retries,
        )
        runtime = PersistentRuntime(args.path, provider)
        response = runtime.interact(args.prompt, system=args.system)
    except (ProviderError, ContinuityError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        else:
            print(f"run failed: {exc}", file=sys.stderr)
        return 2

    result = {
        "status": "VERIFIED",
        "text": response.text,
        "provider": response.provider.to_dict(),
        "provenance": dict(response.provenance),
    }
    _emit(result, json_mode=args.json, title="Model response")
    return 0


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmic-synapse",
        description=(
            "Persistent AI runtime utilities that keep user-owned memory, state, provenance, "
            "routing, and authority separate from replaceable model providers."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Report core imports and optional subsystem dependencies."
    )
    _add_json_flag(doctor)
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

    init = subparsers.add_parser("init", help="Create a new user-owned continuity workspace.")
    init.add_argument("path", help="Destination directory; it must be empty or absent.")
    init.add_argument("--name", required=True, help="Human-readable workspace name.")
    init.add_argument("--seed", type=int, default=0, help="Deterministic CST seed.")
    _add_json_flag(init)
    init.set_defaults(handler=_init_workspace)

    inspect = subparsers.add_parser("inspect", help="Inspect local continuity state without networking.")
    inspect.add_argument("path", help="Workspace directory.")
    _add_json_flag(inspect)
    inspect.set_defaults(handler=_inspect_workspace)

    export = subparsers.add_parser("export", help="Export an integrity-addressed portable bundle.")
    export.add_argument("path", help="Workspace directory.")
    export.add_argument("bundle", help="Destination .cosmos bundle path.")
    _add_json_flag(export)
    export.set_defaults(handler=_export_workspace)

    verify = subparsers.add_parser("verify", help="Verify a portable bundle without extracting it.")
    verify.add_argument("bundle", help="Portable .cosmos bundle path.")
    _add_json_flag(verify)
    verify.set_defaults(handler=_verify_bundle)

    import_cmd = subparsers.add_parser("import", help="Verify and import a portable bundle.")
    import_cmd.add_argument("bundle", help="Portable .cosmos bundle path.")
    import_cmd.add_argument("path", help="Destination directory; it must be empty or absent.")
    _add_json_flag(import_cmd)
    import_cmd.set_defaults(handler=_import_workspace)

    providers = subparsers.add_parser(
        "providers", help="List supported model-provider clients without contacting them."
    )
    _add_json_flag(providers)
    providers.set_defaults(handler=_providers)

    run = subparsers.add_parser(
        "run", help="Send one prompt through an explicit model provider and persist the turn."
    )
    run.add_argument("path", help="Continuity workspace directory.")
    run.add_argument("--provider", choices=("ollama",), required=True)
    run.add_argument("--model", required=True, help="Provider model identifier.")
    run.add_argument("--prompt", required=True, help="User prompt to persist and send.")
    run.add_argument("--system", default=None, help="Optional provider system instruction.")
    run.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    run.add_argument("--revision", default=None, help="Optional known model revision/checkpoint ID.")
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--retries", type=int, default=0)
    _add_json_flag(run)
    run.set_defaults(handler=_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ContinuityError as exc:
        print(f"continuity error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
