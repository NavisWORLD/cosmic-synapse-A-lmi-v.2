use almi_core::MemoryRecord;
use almi_memory::{append_record, read_records, verify_ledger};
use proptest::prelude::*;
use std::collections::BTreeMap;
use tempfile::tempdir;

proptest! {
    #[test]
    fn appended_memory_round_trips_without_mutation(
        role in "[a-z]{1,12}",
        content in "[a-zA-Z0-9 _.,!?-]{0,120}",
    ) {
        let temp = tempdir().unwrap();
        let ledger = temp.path().join("ledger.jsonl");
        let record = MemoryRecord {
            version: Some(1),
            role,
            content,
            provider: None,
            provenance: None,
            timestamp: None,
            extra: BTreeMap::new(),
        };
        let appended = append_record(&ledger, &record).unwrap();
        let records = read_records(&ledger).unwrap();
        let verified = verify_ledger(&ledger).unwrap();
        prop_assert_eq!(appended.index, 0);
        prop_assert_eq!(records, vec![record]);
        prop_assert_eq!(verified.records, 1);
        prop_assert_eq!(verified.digests, vec![appended.sha256]);
    }
}

#[test]
fn malformed_jsonl_is_rejected_instead_of_skipped() {
    let temp = tempdir().unwrap();
    let ledger = temp.path().join("ledger.jsonl");
    std::fs::write(&ledger, b"{not-json}\n").unwrap();
    assert!(verify_ledger(&ledger).is_err());
}
