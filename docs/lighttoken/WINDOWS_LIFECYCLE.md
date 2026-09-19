# A-LMI LightToken workstation — Windows lifecycle

The LightToken workstation is a separate Java 21/JavaFX application backed by the
Rust ABI v1 and optional C++ accelerator. This document covers the workstation,
not the existing A-LMI Native Core lifecycle.

## Supported and tested hosts

- Windows 10+; Windows x64 is exercised by the `windows-latest` GitHub Actions job.
- ARM64 is detected and passed to CMake as `ARM64`, but an ARM64 runner/device
  release test has not yet been recorded. Do not describe ARM64 as verified.
- Build/test/install workflows require a Java 21 JDK, Rust, CMake, and Python 3.11+.
  The installed workstation contains its own jlink runtime image and does not
  require a separate system JDK to launch.

## Developer lifecycle

From a clean source checkout:

```bat
BUILD_LIGHTTOKEN_WINDOWS.bat
TEST_LIGHTTOKEN_WINDOWS.bat
INSTALL_LIGHTTOKEN_WINDOWS.bat -SkipBuild
VERIFY_LIGHTTOKEN_WINDOWS.bat
RUN_LIGHTTOKEN_WINDOWS.bat
UPDATE_LIGHTTOKEN_WINDOWS.bat -SkipBuild
UNINSTALL_LIGHTTOKEN_WINDOWS.bat
```

`BUILD` produces `dist/lighttoken/windows-x64` (or `windows-arm64`) with a
jpackage application image, the Rust native CLI and JNI DLL, and the optional
self-tested C++ accelerator DLL. Missing optional acceleration uses the Rust
fallback. `TEST` runs Rust and Java tests against temporary synthetic inputs;
`VERIFY` checks the installed JNI launcher, native search, a valid A-LMI
workspace, a verified .cosmos and a deliberately corrupted .cosmos.

`RUN` launches the installed application rather than a source-tree classpath.
`UPDATE` requires a named branch and clean source checkout; it fetches only
that branch and uses `git merge --ff-only`. It must refuse dirty or divergent
source. `UNINSTALL` preserves user data by default.

## Owned locations and data

- Default program root: `%LOCALAPPDATA%\Programs\A-LMI\LightToken`.
- Default application/SQLite root: `%LOCALAPPDATA%\A-LMI\LightToken`.
- `LIGHTTOKEN_INSTALL_ROOT` and `LIGHTTOKEN_DATA_ROOT` can override these
  explicitly for isolated test or user-local installations.
- Removing data is a separate opt-in: `UNINSTALL_LIGHTTOKEN_WINDOWS.bat
  -RemoveData` requires the exact phrase `DELETE LIGHTTOKEN DATA`.
- User workspaces, .cosmos bundles, provenance, memory, and evidence outside
  the application-owned data directory are not uninstall targets.

## Interpretation and limits

LightToken spectral bins represent an embedding-domain spectrum, not physical
frequency or Hz. Backend scores and experimental provenance do not convey tool
authority. A successful Windows CI run is evidence for its specific hosted x64
runner, not device-level certification of all Windows 10 builds or ARM64 hardware.
