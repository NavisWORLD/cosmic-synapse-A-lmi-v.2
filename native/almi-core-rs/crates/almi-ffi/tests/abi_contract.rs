use std::ffi::{CStr, CString};

#[test]
fn abi_exposes_version_and_workspace_validation_without_unwinding() {
    assert_eq!(almi_ffi::almi_abi_version(), 1);
    let version = almi_ffi::almi_version();
    assert!(!version.is_null());
    let text = unsafe { CStr::from_ptr(version) }.to_str().unwrap();
    assert!(!text.is_empty());

    let dir = tempfile::tempdir().unwrap();
    almi_continuity::initialize_workspace(dir.path().join("story"), "FFI", 1).unwrap();
    let ctx = almi_ffi::almi_context_new();
    assert!(!ctx.is_null());
    let path = CString::new(dir.path().join("story").to_string_lossy().as_bytes()).unwrap();
    unsafe {
        assert_eq!(almi_ffi::almi_workspace_validate(ctx, path.as_ptr()), 0);
        almi_ffi::almi_context_free(ctx);
    }
}

#[test]
fn cosmos_metadata_uses_owned_string_with_explicit_free() {
    let dir = tempfile::tempdir().unwrap();
    let ws = dir.path().join("story");
    almi_continuity::initialize_workspace(&ws, "FFI", 2).unwrap();
    let bundle = dir.path().join("story.cosmos");
    almi_cosmos::export_bundle(&ws, &bundle).unwrap();
    let ctx = almi_ffi::almi_context_new();
    let path = CString::new(bundle.to_string_lossy().as_bytes()).unwrap();
    let mut out = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            almi_ffi::almi_cosmos_verify_json(ctx, path.as_ptr(), &mut out),
            0
        );
        assert!(!out.is_null());
        let json = CStr::from_ptr(out).to_str().unwrap();
        assert!(json.contains("\"valid\":true"));
        almi_ffi::almi_string_free(out);
        almi_ffi::almi_context_free(ctx);
    }
}
