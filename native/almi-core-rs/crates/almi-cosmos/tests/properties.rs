use almi_cosmos::validate_archive_name;
use proptest::prelude::*;

proptest! {
    #[test]
    fn parent_traversal_is_always_rejected(
        left in "[a-zA-Z0-9_-]{1,20}",
        right in "[a-zA-Z0-9_.-]{1,20}",
    ) {
        let candidate = format!("{left}/../{right}");
        prop_assert!(validate_archive_name(&candidate).is_err());
    }

    #[test]
    fn backslash_paths_are_always_rejected(
        left in "[a-zA-Z0-9_-]{1,20}",
        right in "[a-zA-Z0-9_.-]{1,20}",
    ) {
        let candidate = format!("{left}\\{right}");
        prop_assert!(validate_archive_name(&candidate).is_err());
    }

    #[test]
    fn simple_relative_paths_are_accepted(
        parts in proptest::collection::vec("[a-zA-Z0-9_-]{1,20}", 1..6),
    ) {
        let candidate = parts.join("/");
        prop_assert!(validate_archive_name(&candidate).is_ok());
    }
}
