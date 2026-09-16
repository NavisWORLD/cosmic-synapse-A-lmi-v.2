# Stable C ABI

A-LMI Native Core exposes a deliberately small C ABI in `native/almi-core-rs/include/almi.h`.

## ABI v1

The ABI exposes:

- `almi_abi_version`
- `almi_version`
- opaque `AlmiContext` allocation/free
- last-error access
- workspace validation
- `.cosmos` verification to JSON
- `.cosmos` metadata inspection to JSON
- explicit native string free

Rust-native struct layout is not exported.

## Ownership

`AlmiContext *` returned by `almi_context_new()` must be released with `almi_context_free()`.

JSON strings returned through `char **out_json` are allocated by Rust and must be released with `almi_string_free()`.

Pointers passed to the ABI must be non-null where documented. File paths are UTF-8 strings.

## Errors

ABI error constants are stable within ABI v1. Operations return an integer error code; diagnostic text may be read through `almi_last_error()` on the context.

## Panic boundary

Exported operations contain Rust panics at the FFI boundary. A Rust panic must not unwind into C/C++/Swift/Unity/Kotlin callers.

## Consumer smoke test

`native/almi-core-rs/tests/c_abi_smoke.c` is compiled and linked against the produced Rust library in CI. It validates ABI version, native version visibility, opaque context allocation, and ownership cleanup from an external C translation unit.

The first ABI intentionally remains small. Provider/runtime internals are not frozen into ABI v1.
