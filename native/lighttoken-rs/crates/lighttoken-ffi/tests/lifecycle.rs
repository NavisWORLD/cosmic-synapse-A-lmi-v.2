use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use lighttoken_ffi::c_api::{
    lighttoken_abi_version, lighttoken_backend_json, lighttoken_compare_json,
    lighttoken_context_free, lighttoken_context_new, lighttoken_search_json,
    lighttoken_string_free, lighttoken_validate_json, LIGHTTOKEN_INVALID_ARGUMENT, LIGHTTOKEN_OK,
};
use lighttoken_ffi::LIGHTTOKEN_ABI_VERSION;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn fixture(name: &str) -> CString {
    let bytes = std::fs::read(repo_root().join("tests/fixtures/lighttoken").join(name)).unwrap();
    CString::new(bytes).unwrap()
}

unsafe fn take_json(pointer: *mut std::ffi::c_char) -> serde_json::Value {
    assert!(!pointer.is_null());
    let text = CStr::from_ptr(pointer).to_str().unwrap().to_owned();
    lighttoken_string_free(pointer);
    serde_json::from_str(&text).unwrap()
}

#[test]
fn abi_is_frozen_at_one() {
    assert_eq!(LIGHTTOKEN_ABI_VERSION, 1);
    assert_eq!(lighttoken_abi_version(), 1);
}

#[test]
fn context_lifecycle_and_json_calls_are_owned_and_deterministic() {
    unsafe {
        let context = lighttoken_context_new();
        assert!(!context.is_null());

        let random = fixture("active_random.json");
        let sinusoid = fixture("active_sinusoid.json");
        let mut output = ptr::null_mut();

        assert_eq!(
            lighttoken_validate_json(context, random.as_ptr(), &mut output),
            LIGHTTOKEN_OK
        );
        let validation = take_json(output);
        assert_eq!(validation["valid"], true);
        assert_eq!(validation["embedding_dimension"], 1536);
        assert_eq!(validation["spectral_dimension"], 769);

        output = ptr::null_mut();
        let method = CString::new("cosine").unwrap();
        assert_eq!(
            lighttoken_compare_json(
                context,
                sinusoid.as_ptr(),
                random.as_ptr(),
                method.as_ptr(),
                &mut output,
            ),
            LIGHTTOKEN_OK
        );
        let comparison = take_json(output);
        assert_eq!(comparison["method"], "cosine");
        assert_eq!(comparison["backend"], "rust");

        let collection = CString::new(format!(
            "[{},{}]",
            random.to_str().unwrap().trim(),
            sinusoid.to_str().unwrap().trim()
        ))
        .unwrap();
        let request = CString::new(
            r#"{"method":"cosine","top_k":2,"threshold":null,"modality":null,"source_prefix":null}"#,
        )
        .unwrap();
        output = ptr::null_mut();
        assert_eq!(
            lighttoken_search_json(
                context,
                random.as_ptr(),
                collection.as_ptr(),
                request.as_ptr(),
                &mut output,
            ),
            LIGHTTOKEN_OK
        );
        let search = take_json(output);
        assert_eq!(search["backend"], "rust");
        assert_eq!(search["hits"].as_array().unwrap().len(), 2);

        output = ptr::null_mut();
        assert_eq!(lighttoken_backend_json(context, &mut output), LIGHTTOKEN_OK);
        let backend = take_json(output);
        assert_eq!(backend["active"], "rust");

        lighttoken_context_free(context);
    }
}

#[test]
fn null_inputs_fail_closed_without_panicking() {
    unsafe {
        let context = lighttoken_context_new();
        assert!(!context.is_null());
        assert_eq!(
            lighttoken_validate_json(context, ptr::null(), ptr::null_mut()),
            LIGHTTOKEN_INVALID_ARGUMENT
        );
        lighttoken_context_free(context);
        lighttoken_context_free(ptr::null_mut());
        lighttoken_string_free(ptr::null_mut());
    }
}

#[test]
fn repeated_context_allocation_validation_and_release_stays_bounded() {
    let valid = fixture("active_zero.json");
    let malformed = CString::new("{malformed").unwrap();
    unsafe {
        for _ in 0..256 {
            let context = lighttoken_context_new();
            assert!(!context.is_null());
            let mut output = ptr::null_mut();
            assert_eq!(
                lighttoken_validate_json(context, valid.as_ptr(), &mut output),
                LIGHTTOKEN_OK
            );
            assert_eq!(take_json(output)["valid"], true);
            output = ptr::null_mut();
            assert_ne!(
                lighttoken_validate_json(context, malformed.as_ptr(), &mut output),
                LIGHTTOKEN_OK
            );
            assert!(output.is_null(), "failed C ABI call must not leak an output string");
            lighttoken_context_free(context);
        }
    }
}
