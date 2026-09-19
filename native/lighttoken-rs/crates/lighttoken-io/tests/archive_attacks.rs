use std::fs::File;
use std::io::Write;

use lighttoken_io::read_verified_cosmos;
use tempfile::tempdir;
use zip::write::SimpleFileOptions;

#[test]
fn duplicate_zip_members_are_rejected_by_the_shared_almi_verifier() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("duplicate.cosmos");
    let mut writer = zip::ZipWriter::new(File::create(&path).unwrap());
    // zip::ZipWriter itself rejects duplicate names. Create a structurally
    // valid two-member ZIP, then rename the second equal-length filename
    // in both local and central headers to exercise the *reader* boundary.
    let options = SimpleFileOptions::default();
    writer.start_file("dupA.json", options).unwrap();
    writer.write_all(b"first").unwrap();
    writer.start_file("dupB.json", options).unwrap();
    writer.write_all(b"second").unwrap();
    writer.finish().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let from = b"dupB.json";
    let to = b"dupA.json";
    let mut replaced = 0;
    for index in 0..=bytes.len() - from.len() {
        if &bytes[index..index + from.len()] == from {
            bytes[index..index + from.len()].copy_from_slice(to);
            replaced += 1;
        }
    }
    assert_eq!(replaced, 2, "local and central filenames must both change");
    std::fs::write(&path, bytes).unwrap();
    // The ZIP reader or the shared A-LMI verifier must reject this archive.
    // The rejection must happen before any import or token discovery.
    assert!(read_verified_cosmos(&path).is_err());
}

#[test]
fn traversal_zip_members_are_rejected_before_import() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("traversal.cosmos");
    let mut writer = zip::ZipWriter::new(File::create(&path).unwrap());
    writer
        .start_file("../outside", SimpleFileOptions::default())
        .unwrap();
    writer.write_all(b"should never be extracted").unwrap();
    writer.finish().unwrap();

    assert!(read_verified_cosmos(&path).is_err());
    assert!(!temp.path().join("outside").exists());
}
