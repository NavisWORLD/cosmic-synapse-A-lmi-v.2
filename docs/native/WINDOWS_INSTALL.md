# Windows Installation and Use

## Supported target

The closure workflow targets Windows 10/11 user-local installation, with x86_64 as the packaged release target. The Rust core remains portable and is not coupled to Windows APIs.

## Prerequisites

- Windows PowerShell
- Rust/Cargo for source installation
- Python 3.11+ only when Python bindings are requested

Missing Rust/Python prerequisites are reported explicitly. The installer does not silently install system-wide software or require administrator privileges.

## User-facing entry points

From the repository root:

```text
INSTALL_WINDOWS.bat
BUILD_WINDOWS.bat
RUN_WINDOWS.bat
TEST_WINDOWS.bat
VERIFY_WINDOWS.bat
UPDATE_WINDOWS.bat
UNINSTALL_WINDOWS.bat
```

The BAT files delegate to `scripts/windows/*.ps1` and return the underlying failure code.

## Install

```text
INSTALL_WINDOWS.bat
INSTALL_WINDOWS.bat -WithPython
INSTALL_WINDOWS.bat -SkipPython
```

Default native install location:

```text
%LOCALAPPDATA%\A-LMI\bin\almi.exe
```

The installer builds a release CLI and C ABI library, copies the native executable into the user-local install directory, optionally builds/installs the PyO3 wheel, runs `almi doctor`, and executes the synthetic continuity verification.

## Run

```text
RUN_WINDOWS.bat doctor
RUN_WINDOWS.bat version
RUN_WINDOWS.bat init MyStory --name "My Story" --seed 1
RUN_WINDOWS.bat inspect MyStory
RUN_WINDOWS.bat export MyStory story.cosmos
RUN_WINDOWS.bat verify story.cosmos
```

No developer path is hardcoded. The runner prefers the installed executable and falls back to a repository release build.

## Test and verify

`TEST_WINDOWS.bat` executes Rust workspace tests and, when Python is available, builds a temporary isolated Python environment for cross-language interoperability tests.

`VERIFY_WINDOWS.bat` uses only temporary synthetic data. It checks executable/version health, workspace creation, all six deny-by-default authority domains, deterministic double export, SHA-256 equality, bundle verification, import, restored identity, and restored authority. Temporary verification data is deleted afterward.

## Update

`UPDATE_WINDOWS.bat` refuses to update a dirty source checkout. It fetches the current branch, performs only a fast-forward merge, then reinstalls program files. It does not delete user workspaces or bundles.

## Uninstall

`UNINSTALL_WINDOWS.bat` removes installer-owned program files. It never scans for user workspaces, `.cosmos` bundles, memory ledgers, or backups. Optional installer-owned application-data removal requires `-RemoveData`; interactive deletion requires typing `DELETE` unless `-Force` is also supplied.

## Portable package

CI creates a versioned Windows ZIP containing the standalone `almi.exe`, runner/verification BAT files with their PowerShell support files, the C ABI header, README, and SHA-256 checksums. End users of that ZIP do not need Rust installed to execute `almi.exe`.
