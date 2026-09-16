# Native Security Model

## Authority

A newly initialized native workspace is deny-by-default across six independent authority domains:

- tool
- network
- filesystem
- cloud
- deployment
- actuator

Configuring or replacing a model/provider does not populate any authority list. Provider output is data, not authority.

## `.cosmos` input handling

Native bundle verification rejects unsafe or inconsistent archives, including path traversal, absolute/prefixed paths, backslash path tricks, `.`/`..` components, symlink members, directory members, duplicate names, secret-bearing member names, excessive file counts, excessive sizes, unsupported versions, undeclared files, missing files, size mismatches, and SHA-256 mismatches.

Import verifies the bundle before extraction, refuses unsafe destination types, and refuses non-empty destinations where overwrite would be unsafe. It writes declared verified payloads rather than calling an unchecked archive `extractall` equivalent.

## Secrets

Secret-bearing paths such as `.env`, private-key-like suffixes, common credential files, and key filenames are excluded from portable bundles. Provider endpoints containing embedded credentials are rejected before provenance is persisted. Provider secrets must not be copied into logs, bundle metadata, or provenance.

## Provider networking

The native Ollama-compatible provider defaults to a loopback endpoint. Merely constructing a provider does not grant network authority to the model or mutate workspace policy.

## FFI

The C ABI uses opaque handles, explicit ownership functions, stable error codes, and panic containment. Rust panics are converted to an ABI error rather than unwinding across the foreign-function boundary.

## Tooling

Native CI runs formatting, Clippy, workspace tests, cross-language tests, an external C ABI link smoke, multi-OS release builds, Windows installer verification, and `cargo audit`. Findings must be fixed or explicitly recorded; passing compilation alone is not a security certification.

## Non-claim

These controls are engineering safeguards for the tested software surface. They do not constitute production security certification, formal verification, or a guarantee about hardware/services that were not exercised.
