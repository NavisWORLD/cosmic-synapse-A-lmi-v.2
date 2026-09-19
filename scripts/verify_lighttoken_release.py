#!/usr/bin/env python3
"""Verify portable LightToken release ZIPs without extracting untrusted paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath


def verify(path: Path, expected_commit: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        for name in names:
            p = PurePosixPath(name)
            if (not name or name.startswith("/") or "\\" in name
                    or any(part in ("", ".", "..") for part in p.parts)):
                raise ValueError(f"unsafe archive entry: {name!r}")
            if archive.getinfo(name).file_size > 1024 * 1024 * 1024:
                raise ValueError("unexpected oversized archive member")
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise ValueError("release must have one root")
        root = roots.pop()
        def read(member: str) -> bytes:
            return archive.read(f"{root}/{member}")

        evidence = json.loads(read("EVIDENCE.json"))
        if evidence["commit"] != expected_commit:
            raise ValueError("release commit differs from exact CI head")
        sbom = json.loads(read("SBOM.spdx.json"))
        if sbom["spdxVersion"] != "SPDX-2.3":
            raise ValueError("missing SPDX 2.3 inventory")
        checksums = read("SHA256SUMS.txt").decode("utf-8").splitlines()
        seen = set()
        for entry in checksums:
            checksum, sep, relative = entry.partition("  ")
            if not sep or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError("malformed checksum line")
            if relative in seen or relative == "SHA256SUMS.txt":
                raise ValueError("duplicate/self-referential checksum")
            seen.add(relative)
            if hashlib.sha256(read(relative)).hexdigest() != checksum:
                raise ValueError(f"hash mismatch: {relative}")
        files = {name[len(root) + 1:] for name in names if name.startswith(root + "/")}
        if seen != files - {"SHA256SUMS.txt"}:
            raise ValueError(f"manifest misses files: {sorted((files - {'SHA256SUMS.txt'}) ^ seen)}")
        if not evidence["files"] or not set(evidence["files"]).issubset(seen):
            raise ValueError("missing file evidence")
        if not any(name.startswith("include/") and name.endswith(".h") for name in seen):
            raise ValueError("native C header absent")
        if not any(name.startswith("provenance/") and name.endswith("Cargo.lock") for name in seen):
            raise ValueError("Rust lock absent")
        if not any(name.startswith("provenance/") and name.endswith("gradle.lockfile") for name in seen):
            raise ValueError("Gradle lock absent")
        if not any("lighttoken_ffi" in name and name.endswith((".dll", ".so", ".dylib")) for name in seen):
            raise ValueError("native JNI library absent")
        if not any("lighttoken_accel" in name and name.endswith((".dll", ".so", ".dylib")) for name in seen):
            raise ValueError("accelerator library absent")
        return {"name": path.name, "target": evidence["target"],
                "commit": evidence["commit"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "entries": len(seen)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    archives = sorted(args.root.rglob("lighttoken-*.zip"))
    if len(archives) != 3:
        raise ValueError(f"expected exactly three OS archives; found {len(archives)}")
    reports = [verify(path, args.commit) for path in archives]
    if {report["target"] for report in reports} != {"linux", "macos", "windows"}:
        raise ValueError("missing release platform")
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
