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
    let options = SimpleFileOptions::default();
    writer.start_file("duplicate.json", options).unwrap();
    writer.write_all(b"first").unwrap();
    writer.start_file("duplicate.json", options).unwrap();
    writer.write_all(b"second").unwrap();
    writer.finish().unwrap();

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
