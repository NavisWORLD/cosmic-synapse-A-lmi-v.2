"""Portable user-owned continuity workspaces and integrity-checked bundles.

The continuity layer deliberately owns software state and provenance, not model
parameters or model authority. A workspace is a normal directory of JSON/JSONL
files. Export creates a deterministic ZIP container (``.cosmos`` by convention)
whose manifest hashes every payload. Import verifies before writing and rejects
archive/path tricks rather than trusting ``ZipFile.extractall``.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from cosmic_synapse.cst_state import CSTEngine

WORKSPACE_FORMAT_VERSION = 1
BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST = "bundle-manifest.json"
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_FILES = 10_000

_REQUIRED_WORKSPACE_FILES = (
    "system.json",
    "memory/ledger.jsonl",
    "state/cst.json",
    "knowledge/graph.json",
    "artifacts/manifest.json",
    "provenance/provider.json",
    "routing/state.json",
    "policy/authority.json",
)

_SECRET_FILENAMES = {
    ".env",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


class ContinuityError(RuntimeError):
    """Base error for portable continuity operations."""


class ContinuityIntegrityError(ContinuityError):
    """Raised when a workspace or bundle fails integrity checks."""


class ContinuitySecurityError(ContinuityError):
    """Raised when an operation would cross the portable-data safety boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityIntegrityError(f"invalid JSON file: {path}") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_secret_path(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    for part in parts:
        lower = part.lower()
        if lower in _SECRET_FILENAMES:
            return True
        if lower.startswith(".env"):
            return True
        if any(lower.endswith(suffix) for suffix in _SECRET_SUFFIXES):
            return True
    return False


def _validate_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or name.startswith("/")
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ContinuitySecurityError(f"unsafe archive path: {name!r}")
    return path


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    if info.create_system != 3:
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def initialize_workspace(path: str | Path, name: str, seed: int = 0) -> dict[str, Any]:
    """Create a new portable continuity workspace with no tool authority."""

    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise ContinuitySecurityError(f"workspace destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    created_at = _utc_now()
    system = {
        "format_version": WORKSPACE_FORMAT_VERSION,
        "name": str(name),
        "created_at": created_at,
        "seed": int(seed),
    }
    _write_json(root / "system.json", system)

    ledger = root / "memory" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")

    cst = CSTEngine(seed=int(seed)).state.to_dict()
    _write_json(root / "state" / "cst.json", cst)
    _write_json(root / "knowledge" / "graph.json", {"version": 1, "nodes": [], "edges": []})
    _write_json(root / "artifacts" / "manifest.json", {"version": 1, "artifacts": []})
    _write_json(
        root / "provenance" / "provider.json",
        {
            "version": 1,
            "provider_id": None,
            "model_id": None,
            "revision": None,
            "endpoint": None,
            "capabilities": [],
            "updated_at": None,
        },
    )
    _write_json(root / "routing" / "state.json", {"version": 1, "routes": {}})
    _write_json(
        root / "policy" / "authority.json",
        {
            "version": 1,
            "tool_authority": [],
            "network_authority": [],
            "filesystem_authority": [],
        },
    )
    return dict(system)


def _require_workspace(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ContinuityIntegrityError(f"workspace does not exist: {root}")
    missing = [relative for relative in _REQUIRED_WORKSPACE_FILES if not (root / relative).is_file()]
    if missing:
        raise ContinuityIntegrityError(f"workspace is missing required files: {', '.join(missing)}")
    system = _read_json(root / "system.json")
    version = int(system.get("format_version", 0))
    if version != WORKSPACE_FORMAT_VERSION:
        raise ContinuityIntegrityError(f"unsupported workspace format version: {version}")
    return system


def append_memory_record(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append one canonical JSONL record and return its stable line digest."""

    root = Path(path)
    _require_workspace(root)
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    try:
        line = _canonical_json_bytes(dict(record))
    except (TypeError, ValueError) as exc:
        raise ContinuityIntegrityError("memory record must be JSON-serializable") from exc

    ledger = root / "memory" / "ledger.jsonl"
    with ledger.open("ab") as handle:
        handle.write(line)
    count = sum(1 for item in ledger.read_text(encoding="utf-8").splitlines() if item.strip())
    return {"index": count - 1, "sha256": _sha256(line)}


def inspect_workspace(path: str | Path) -> dict[str, Any]:
    """Return the user-visible continuity status without opening external services."""

    root = Path(path)
    system = _require_workspace(root)
    ledger_lines = [
        line for line in (root / "memory" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for line in ledger_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContinuityIntegrityError("memory ledger contains invalid JSONL") from exc

    return {
        "valid": True,
        "format_version": int(system["format_version"]),
        "name": system["name"],
        "created_at": system["created_at"],
        "memory_records": len(ledger_lines),
        "provider": _read_json(root / "provenance" / "provider.json"),
        "authority": _read_json(root / "policy" / "authority.json"),
        "state": _read_json(root / "state" / "cst.json"),
    }


def _workspace_payloads(root: Path) -> list[tuple[str, bytes]]:
    _require_workspace(root)
    payloads: list[tuple[str, bytes]] = []
    total = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            relative = candidate.relative_to(root).as_posix()
            raise ContinuitySecurityError(f"workspace symlink is not portable: {relative}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        _validate_archive_name(relative)
        if _is_secret_path(relative):
            raise ContinuitySecurityError(f"secret-bearing file is excluded from bundles: {relative}")
        data = candidate.read_bytes()
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise ContinuitySecurityError("workspace exceeds portable bundle size limit")
        payloads.append((relative, data))
        if len(payloads) > MAX_BUNDLE_FILES:
            raise ContinuitySecurityError("workspace exceeds portable bundle file-count limit")
    return payloads


def export_bundle(path: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    """Export a deterministic, integrity-addressed portable bundle."""

    root = Path(path)
    system = _require_workspace(root)
    payloads = _workspace_payloads(root)
    entries = [
        {"path": relative, "sha256": _sha256(data), "size": len(data)}
        for relative, data in payloads
    ]
    manifest = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "workspace_format_version": WORKSPACE_FORMAT_VERSION,
        "workspace_name": system["name"],
        "files": entries,
    }
    destination = Path(bundle_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        _write_zip_member(archive, BUNDLE_MANIFEST, _canonical_json_bytes(manifest))
        for relative, data in payloads:
            _write_zip_member(archive, relative, data)
    return {
        "valid": True,
        "path": str(destination),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "sha256": _sha256(destination.read_bytes()),
    }


def _scan_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_BUNDLE_FILES + 1:
        raise ContinuitySecurityError("bundle exceeds portable file-count limit")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ContinuitySecurityError("bundle contains duplicate archive member names")

    total = 0
    by_name: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        _validate_archive_name(info.filename)
        if info.is_dir():
            raise ContinuitySecurityError(f"directory archive member is not allowed: {info.filename}")
        if _zipinfo_is_symlink(info):
            raise ContinuitySecurityError(f"symlink archive member is not allowed: {info.filename}")
        if info.filename != BUNDLE_MANIFEST and _is_secret_path(info.filename):
            raise ContinuitySecurityError(
                f"secret-bearing archive member is not allowed: {info.filename}"
            )
        total += int(info.file_size)
        if total > MAX_BUNDLE_BYTES:
            raise ContinuitySecurityError("bundle exceeds portable uncompressed-size limit")
        by_name[info.filename] = info
    return by_name


def _load_bundle_manifest(archive: zipfile.ZipFile, by_name: Mapping[str, zipfile.ZipInfo]) -> dict:
    if BUNDLE_MANIFEST not in by_name:
        raise ContinuityIntegrityError("bundle manifest is missing")
    try:
        manifest = json.loads(archive.read(BUNDLE_MANIFEST).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise ContinuityIntegrityError("bundle manifest is invalid") from exc
    version = int(manifest.get("bundle_format_version", 0))
    if version != BUNDLE_FORMAT_VERSION:
        raise ContinuityIntegrityError(f"unsupported bundle format version: {version}")
    workspace_version = int(manifest.get("workspace_format_version", 0))
    if workspace_version != WORKSPACE_FORMAT_VERSION:
        raise ContinuityIntegrityError(
            f"unsupported workspace format version in bundle: {workspace_version}"
        )
    if not isinstance(manifest.get("files"), list):
        raise ContinuityIntegrityError("bundle manifest files must be a list")
    return manifest


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Verify archive safety, declarations, sizes, and SHA-256 digests."""

    source = Path(bundle_path)
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ContinuityIntegrityError("bundle is not a readable ZIP container") from exc

    with archive:
        by_name = _scan_archive(archive)
        manifest = _load_bundle_manifest(archive, by_name)
        declared_entries: dict[str, dict[str, Any]] = {}
        for raw_entry in manifest["files"]:
            if not isinstance(raw_entry, dict):
                raise ContinuityIntegrityError("bundle manifest file entry is invalid")
            name = str(raw_entry.get("path", ""))
            _validate_archive_name(name)
            if name == BUNDLE_MANIFEST:
                raise ContinuityIntegrityError("bundle manifest cannot declare itself as payload")
            if name in declared_entries:
                raise ContinuityIntegrityError(f"duplicate manifest declaration: {name}")
            declared_entries[name] = raw_entry

        actual_payload_names = set(by_name) - {BUNDLE_MANIFEST}
        declared_names = set(declared_entries)
        extras = sorted(actual_payload_names - declared_names)
        if extras:
            raise ContinuityIntegrityError(f"undeclared archive member: {extras[0]}")
        missing = sorted(declared_names - actual_payload_names)
        if missing:
            raise ContinuityIntegrityError(f"declared payload is missing: {missing[0]}")

        total = 0
        for name in sorted(declared_names):
            entry = declared_entries[name]
            data = archive.read(name)
            size = int(entry.get("size", -1))
            if len(data) != size:
                raise ContinuityIntegrityError(f"size mismatch for {name}")
            expected_hash = str(entry.get("sha256", ""))
            if _sha256(data) != expected_hash:
                raise ContinuityIntegrityError(f"hash mismatch for {name}")
            total += len(data)

        if "system.json" not in declared_names:
            raise ContinuityIntegrityError("bundle does not contain system.json")
        try:
            system = json.loads(archive.read("system.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContinuityIntegrityError("bundle system.json is invalid") from exc
        if int(system.get("format_version", 0)) != WORKSPACE_FORMAT_VERSION:
            raise ContinuityIntegrityError("bundle system.json has unsupported format version")

    return {
        "valid": True,
        "path": str(source),
        "name": system.get("name"),
        "file_count": len(declared_names),
        "total_bytes": total,
        "sha256": _sha256(source.read_bytes()),
    }


def import_bundle(bundle_path: str | Path, destination: str | Path) -> dict[str, Any]:
    """Verify a bundle completely, then materialize it without unsafe extraction."""

    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        raise ContinuitySecurityError(f"destination is not empty: {target}")

    verified = verify_bundle(bundle_path)
    source = Path(bundle_path)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as archive:
        by_name = _scan_archive(archive)
        manifest = _load_bundle_manifest(archive, by_name)
        for entry in sorted(manifest["files"], key=lambda item: str(item["path"])):
            name = str(entry["path"])
            relative = _validate_archive_name(name)
            output = target.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(name))

    summary = inspect_workspace(target)
    return {**verified, "destination": str(target), "memory_records": summary["memory_records"]}
