use almi_core::MemoryRecord;
use almi_memory::{append_record, verify_ledger};
use std::collections::BTreeMap;

#[test]
fn append_is_canonical_and_verify_preserves_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.jsonl");
    let record = MemoryRecord {
        version: Some(1),
        role: "user".into(),
        content: "sunflower".into(),
        provider: None,
        provenance: None,
        timestamp: None,
        extra: BTreeMap::new(),
    };
    let first = append_record(&path, &record).unwrap();
    let second = append_record(
        &path,
        &MemoryRecord {
            content: "second".into(),
            ..record.clone()
        },
    )
    .unwrap();
    assert_eq!(first.index, 0);
    assert_eq!(second.index, 1);
    let verified = verify_ledger(&path).unwrap();
    assert_eq!(verified.records, 2);
    assert_eq!(verified.digests.len(), 2);
}

#[test]
fn malformed_or_unknown_version_record_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.jsonl");
    std::fs::write(&path, "{bad json}\n").unwrap();
    assert!(verify_ledger(&path).is_err());
    std::fs::write(
        &path,
        "{\"version\":99,\"role\":\"user\",\"content\":\"x\"}\n",
    )
    .unwrap();
    assert!(verify_ledger(&path).is_err());
}
