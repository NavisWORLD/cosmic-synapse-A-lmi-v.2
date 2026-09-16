use almi_core::{AuthorityPolicy, SystemIdentity, WORKSPACE_FORMAT_VERSION};

#[test]
fn native_authority_defaults_are_deny_by_default() {
    let policy = AuthorityPolicy::default();
    assert_eq!(policy.version, 1);
    assert!(policy.tool_authority.is_empty());
    assert!(policy.network_authority.is_empty());
    assert!(policy.filesystem_authority.is_empty());
    assert!(policy.cloud_authority.is_empty());
    assert!(policy.deployment_authority.is_empty());
    assert!(policy.actuator_authority.is_empty());
}

#[test]
fn system_identity_rejects_unknown_workspace_version() {
    let identity = SystemIdentity {
        format_version: WORKSPACE_FORMAT_VERSION + 1,
        name: "future".to_owned(),
        created_at: "2026-09-15T00:00:00+00:00".to_owned(),
        seed: 0,
    };
    assert!(identity.validate().is_err());
}
