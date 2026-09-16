# Native Verification

Verification is evidence-based. Do not mark a gate passed unless the corresponding command or CI job executed successfully on the stated commit.

## Local Rust gates

```text
cd native/almi-core-rs
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
cargo build --release -p almi-cli -p almi-ffi
cargo audit
```

## Python interoperability

Build the native CLI and PyO3 wheel, install the wheel, then set `ALMI_NATIVE_CLI` to the built executable and execute:

```text
python -m pytest -q tests/test_native_interop.py tests/test_native_cst_interop.py native/almi-core-rs/crates/almi-python/python-tests/test_bindings.py
```

The suite covers Python-export/Rust-import, Rust-export/Python-import, deterministic repeated exports, Python/Rust binding loading, and CST replay from persisted Python state.

## C ABI

On a supported Unix CI runner:

```text
cargo build -p almi-ffi
cc tests/c_abi_smoke.c -I include -L target/debug -Wl,-rpath,$PWD/target/debug -lalmi_ffi -o target/c-abi-smoke
./target/c-abi-smoke
```

## Windows

```text
BUILD_WINDOWS.bat -WithPython
TEST_WINDOWS.bat
INSTALL_WINDOWS.bat -WithPython
VERIFY_WINDOWS.bat
RUN_WINDOWS.bat version
UNINSTALL_WINDOWS.bat
```

The Windows CI job executes those user-facing entry points rather than bypassing them with an unrelated internal script.

## Release-candidate artifacts

The Windows workflow packages `A-LMI-native-0.1.0-windows-x86_64.zip`, Python wheel output, `almi.h`, and `checksums.txt` as workflow artifacts. Artifact hashes used in a final report must be taken from the exact successful branch/merge run.

## Post-merge rule

A successful branch run is not sufficient for closure. After merge, the Native Core CI workflow must run again on the exact `main` merge SHA. The final report records both the final PR head and post-merge `main` evidence.
