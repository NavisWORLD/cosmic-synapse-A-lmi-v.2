use serde_json::Value;
use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_almi"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn version_and_doctor_are_machine_readable() {
    let version = run(&["--json", "version"]);
    assert!(
        version.status.success(),
        "{}",
        String::from_utf8_lossy(&version.stderr)
    );
    let parsed: Value = serde_json::from_slice(&version.stdout).unwrap();
    assert_eq!(parsed["abi_version"], 1);
    assert!(parsed["version"].as_str().is_some());

    let doctor = run(&["--json", "doctor"]);
    assert!(
        doctor.status.success(),
        "{}",
        String::from_utf8_lossy(&doctor.stderr)
    );
    let parsed: Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(parsed["authority_default"], "deny");
}

#[test]
fn continuity_cli_round_trip_and_memory_verify_work() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("story");
    let bundle = dir.path().join("story.cosmos");
    let restored = dir.path().join("restored");
    let ws = workspace.to_str().unwrap();
    let b = bundle.to_str().unwrap();
    let restored_s = restored.to_str().unwrap();

    for args in [
        vec!["--json", "init", ws, "--name", "CLI Story", "--seed", "17"],
        vec!["--json", "inspect", ws],
        vec!["--json", "memory", "verify", ws],
        vec!["--json", "export", ws, b],
        vec!["--json", "verify", b],
        vec!["--json", "import", b, restored_s],
    ] {
        let output = run(&args);
        assert!(
            output.status.success(),
            "args={args:?}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(serde_json::from_slice::<Value>(&output.stdout).is_ok());
    }
}

#[test]
fn invalid_bundle_returns_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.cosmos");
    std::fs::write(&bad, b"not a zip").unwrap();
    let output = run(&["--json", "verify", bad.to_str().unwrap()]);
    assert!(!output.status.success());
}
