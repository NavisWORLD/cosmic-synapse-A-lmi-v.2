#!/usr/bin/env python3
"""Generate a synthetic A-LMI workspace + valid/corrupt .cosmos fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from a_lmi.continuity import export_bundle, initialize_workspace

FIXED_TIMESTAMP = "2000-01-01T00:00:00+00:00"
TOKEN_RELATIVE = "artifacts/lighttokens/active_sinusoid.json"
RAW_RELATIVE = "artifacts/raw/active_sinusoid.txt"


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def generate(lighttoken_dir: Path, workspace: Path, bundle: Path, corrupt_bundle: Path) -> None:
    for path in (workspace,):
        if path.exists():
            shutil.rmtree(path)
    for path in (bundle, corrupt_bundle):
        if path.exists():
            path.unlink()

    initialize_workspace(workspace, "Synthetic LightToken Workspace", seed=424242)

    system_path = workspace / "system.json"
    system = json.loads(system_path.read_text(encoding="utf-8"))
    system["created_at"] = FIXED_TIMESTAMP
    write_json(system_path, system)

    source = json.loads((lighttoken_dir / "active_sinusoid.json").read_text(encoding="utf-8"))
    source["raw_data_ref"] = f"workspace://{RAW_RELATIVE}"
    token_bytes = canonical_bytes(source)
    token_path = workspace / TOKEN_RELATIVE
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_bytes(token_bytes)

    raw_path = workspace / RAW_RELATIVE
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"synthetic-lighttoken-raw\n")

    token_sha = hashlib.sha256(token_bytes).hexdigest()
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    write_json(
        workspace / "artifacts/manifest.json",
        {
            "version": 1,
            "artifacts": [
                {
                    "kind": "lighttoken",
                    "path": TOKEN_RELATIVE,
                    "token_id": source["token_id"],
                    "sha256": token_sha,
                    "raw_sha256": raw_sha,
                }
            ],
        },
    )

    bundle.parent.mkdir(parents=True, exist_ok=True)
    export_bundle(workspace, bundle)

    with zipfile.ZipFile(bundle, "r") as source_zip, zipfile.ZipFile(
        corrupt_bundle, "w", compression=zipfile.ZIP_DEFLATED
    ) as target_zip:
        for info in source_zip.infolist():
            payload = source_zip.read(info.filename)
            if info.filename == TOKEN_RELATIVE:
                payload = payload.replace(b'"modality":"synthetic"', b'"modality":"corrupted"', 1)
            target_zip.writestr(info, payload)

    print(
        json.dumps(
            {
                "workspace": str(workspace),
                "bundle": str(bundle),
                "corrupt_bundle": str(corrupt_bundle),
                "token_id": source["token_id"],
                "token_sha256": token_sha,
                "raw_sha256": raw_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lighttoken-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--corrupt-bundle", type=Path, required=True)
    args = parser.parse_args()
    generate(args.lighttoken_dir, args.workspace, args.bundle, args.corrupt_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
