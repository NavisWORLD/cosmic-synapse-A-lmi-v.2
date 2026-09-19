#!/usr/bin/env python3
"""Package verified LightToken build outputs with file hashes and an SPDX dependency inventory.

Only synthetic/build-derived files are packaged.  User workspaces and .cosmos files
are never inputs.  This tool does not substitute for successful CI on its exact SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "native/lighttoken-rs"
JAVA = ROOT / "apps/lighttoken-workstation-java"
CPP = ROOT / "native/lighttoken-cpp"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked build output: {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def dependency_inventory() -> list[dict[str, object]]:
    cargo = tomllib.loads(require_file(RUST / "Cargo.lock").read_text(encoding="utf-8"))
    packages: list[dict[str, object]] = []
    for item in cargo.get("package", []):
        packages.append({"ecosystem": "cargo", "name": item["name"], "version": item["version"]})
    for line in require_file(JAVA / "gradle.lockfile").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        coordinate = line.split("=", 1)[0]
        parts = coordinate.split(":")
        if len(parts) == 3:
            packages.append({
                "ecosystem": "gradle",
                "name": f"{parts[0]}:{parts[1]}",
                "version": parts[2],
            })
    return packages


def produce_sbom(packages: list[dict[str, object]], commit: str, target: str, run_id: str) -> dict:
    items = []
    relationships = []
    for index, package in enumerate(packages, start=1):
        identifier = f"SPDXRef-Dependency-{index}"
        items.append({
            "SPDXID": identifier, "name": f"{package['ecosystem']}:{package['name']}",
            "versionInfo": package["version"], "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False, "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION", "copyrightText": "NOASSERTION",
        })
        relationships.append({
            "spdxElementId": "SPDXRef-LightToken", "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": identifier,
        })
    items.insert(0, {
        "SPDXID": "SPDXRef-LightToken", "name": "A-LMI LightToken Workstation",
        "versionInfo": "0.1.0", "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False, "licenseConcluded": "GPL-3.0-only",
        "licenseDeclared": "GPL-3.0-only", "copyrightText": "NOASSERTION",
    })
    relationships.insert(0, {
        "spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
        "relatedSpdxElement": "SPDXRef-LightToken",
    })
    return {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"lighttoken-{target}-{commit[:12]}",
        "documentNamespace": f"https://github.com/NavisWORLD/cosmic-synapse-A-lmi-v.2/lighttoken/{commit}/{target}/{run_id}",
        "creationInfo": {
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: package_lighttoken_release.py"],
            "comment": "Dependency inventory derived from committed Cargo and Gradle locks; licenses not audited individually.",
        },
        "packages": items, "relationships": relationships,
    }


def choose_cpp(target: str) -> Path | None:
    build = CPP / ("build-windows" if target == "windows" else "build")
    suffix = {"windows": ".dll", "macos": ".dylib", "linux": ".so"}[target]
    matches = sorted(path for path in build.rglob(f"*lighttoken_accel{suffix}") if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected one self-tested C++ library in {build}, got {matches}")
    return require_file(matches[0])


def ensure_package_native(stage: Path, filename: str, expected_hash: str) -> Path:
    jars = list(stage.rglob("lighttoken-workstation-java-0.1.0.jar"))
    if len(jars) != 1:
        raise ValueError(f"expected one Java application JAR, got {len(jars)}")
    native = require_file(jars[0].parent / "native" / filename)
    if sha256(native) != expected_hash:
        raise ValueError("packaged JNI library does not match the release build")
    return native.parent


def build_stage(target: str, stage: Path) -> tuple[Path, Path]:
    suffix = {"windows": ".dll", "macos": ".dylib", "linux": ".so"}[target]
    cli = RUST / "target/release" / ("lighttoken.exe" if target == "windows" else "lighttoken")
    ffi = RUST / "target/release" / (f"lighttoken_ffi{suffix}" if target == "windows" else f"liblighttoken_ffi{suffix}")
    cpp = choose_cpp(target)
    if target == "windows":
        source = ROOT / "dist/lighttoken/windows-x64"
        if not (source / "app/LightTokenWorkstation.exe").is_file():
            raise ValueError("missing user-local Windows application image")
        shutil.copytree(source, stage, dirs_exist_ok=True)
        native_dir = ensure_package_native(stage, ffi.name, sha256(require_file(ffi)))
        if not (stage / "bin/lighttoken.exe").is_file():
            raise ValueError("missing portable CLI")
    else:
        image = JAVA / "build/jpackage" / ("LightTokenWorkstation.app" if target == "macos" else "LightTokenWorkstation")
        if not image.is_dir():
            raise ValueError(f"jpackage app image missing: {image}")
        shutil.copytree(image, stage / "app" / "LightTokenWorkstation.app" if target == "macos" else stage / "app")
        copy_file(cli, stage / "bin/lighttoken")
        jars = list((stage / "app").rglob("lighttoken-workstation-java-0.1.0.jar"))
        if len(jars) != 1:
            raise ValueError("cannot locate packaged Java application JAR")
        native_dir = jars[0].parent / "native"
        copy_file(ffi, native_dir / ffi.name)
        copy_file(cpp, native_dir / cpp.name)
        copy_file(cpp, stage / "bin" / cpp.name)
        ensure_package_native(stage, ffi.name, sha256(ffi))
    if not (native_dir / cpp.name).is_file():
        raise ValueError("self-tested C++ accelerator is absent from release package")
    return native_dir, cli


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("linux", "macos", "windows"), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        parser.error("--commit must be the 40-hex Git SHA")
    if not re.fullmatch("[0-9]+", args.run_id):
        parser.error("--run-id must be an Actions run number")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    label = f"lighttoken-{args.target}-{platform.machine().lower()}-{args.commit[:12]}"
    stage = output / label
    if stage.exists():
        raise ValueError(f"release stage already exists: {stage}")
    stage.mkdir()
    native_dir, cli = build_stage(args.target, stage)
    copy_file(ROOT / "native/lighttoken-rs/include/lighttoken.h", stage / "include/lighttoken.h")
    copy_file(RUST / "Cargo.lock", stage / "provenance/Cargo.lock")
    copy_file(JAVA / "gradle.lockfile", stage / "provenance/gradle.lockfile")
    copy_file(ROOT / "LICENSE", stage / "LICENSE")
    packages = dependency_inventory()
    write_json(stage / "SBOM.spdx.json", produce_sbom(packages, args.commit, args.target, args.run_id))
    artifact_files = sorted(path for path in stage.rglob("*") if path.is_file())
    inventory = {
        path.relative_to(stage).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifact_files
    }
    evidence = {
        "schema_version": 1, "repo": "NavisWORLD/cosmic-synapse-A-lmi-v.2",
        "commit": args.commit, "actions_run_id": args.run_id, "target": args.target,
        "runner_platform": platform.platform(), "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "package_kind": "portable application image; no installer or device certification",
        "source_kind": "build products and dependency locks only",
        "native_directory": native_dir.relative_to(stage).as_posix(),
        "cli_sha256": sha256(require_file(cli)),
        "dependencies_in_sbom": len(packages),
        "files": inventory,
        "non_claims": ["physical frequency", "consciousness", "AGI", "general performance superiority"],
    }
    write_json(stage / "EVIDENCE.json", evidence)
    digest_files = sorted(path for path in stage.rglob("*") if path.is_file())
    (stage / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in digest_files),
        encoding="utf-8",
    )
    archive = output / f"{label}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError(f"refusing symlink in portable archive: {path}")
                bundle.write(path, (Path(label) / path.relative_to(stage)).as_posix())
    print(json.dumps({"archive": str(archive), "sha256": sha256(archive),
                      "files": len(digest_files), "dependencies": len(packages),
                      "commit": args.commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
