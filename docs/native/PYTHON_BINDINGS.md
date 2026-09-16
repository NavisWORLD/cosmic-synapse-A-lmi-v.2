# Python Bindings

The `almi-python` crate builds the `almi_native` CPython extension with PyO3 and maturin. The binding layer is additive: the existing Python A-LMI implementation remains the reference/oracle during native parity work.

## Build

```text
maturin build --release --manifest-path native/almi-core-rs/crates/almi-python/Cargo.toml
```

The current binding is configured for Python 3.11+ via PyO3's stable ABI support.

## Exposed operations

The native module exposes selected continuity operations including native `.cosmos` verification, workspace inspection, and bundle export. The API is intentionally narrower than the Rust internal crate graph.

Example:

```python
import almi_native

metadata = almi_native.verify_cosmos("story.cosmos")
workspace = almi_native.inspect_workspace("story")
almi_native.export_cosmos("story", "story.cosmos")
```

## Windows installer behavior

`INSTALL_WINDOWS.bat -WithPython` builds a wheel and installs it into the invoking user's Python environment. No system-wide administrator installation is required. `TEST_WINDOWS.bat` uses an isolated temporary virtual environment for interoperability tests so test dependencies do not become permanent product dependencies.

## Replacement boundary

The presence of `almi_native` does not mean the Python reference implementation has been removed. Future replacement decisions require explicit parity evidence beyond simply importing the extension.
