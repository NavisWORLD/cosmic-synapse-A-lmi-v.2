#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(temp) = tempfile::tempdir() else {
        return;
    };
    let bundle = temp.path().join("input.cosmos");
    if std::fs::write(&bundle, data).is_err() {
        return;
    }

    // Arbitrary hostile bytes must be rejected with a normal error or accepted
    // as a valid bundle; they must never panic the archive verifier.
    let _ = almi_cosmos::verify_bundle(&bundle);
});
