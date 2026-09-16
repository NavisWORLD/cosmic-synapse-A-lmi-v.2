//! Minimal stable C ABI for A-LMI native validation and `.cosmos` inspection.
//! Rust layout is never exposed. Callers own opaque handles and explicitly free JSON strings.

use almi_core::{AlmiError, ABI_VERSION};
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

pub const ALMI_OK: i32 = 0;
pub const ALMI_ERROR_NULL: i32 = 1;
pub const ALMI_ERROR_INVALID: i32 = 3;
pub const ALMI_ERROR_INTEGRITY: i32 = 4;
pub const ALMI_ERROR_SECURITY: i32 = 5;
pub const ALMI_ERROR_UNSUPPORTED_VERSION: i32 = 6;
pub const ALMI_ERROR_IO: i32 = 7;
pub const ALMI_ERROR_PANIC: i32 = 8;
pub const ALMI_ERROR_INTERNAL: i32 = 9;

#[repr(C)]
pub struct AlmiContext {
    last_error: Mutex<Option<CString>>,
}

#[no_mangle]
pub extern "C" fn almi_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn almi_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[no_mangle]
pub extern "C" fn almi_context_new() -> *mut AlmiContext {
    catch_unwind(|| {
        Box::into_raw(Box::new(AlmiContext {
            last_error: Mutex::new(None),
        }))
    })
    .unwrap_or(ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn almi_context_free(context: *mut AlmiContext) {
    if context.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(context));
    }));
}

#[no_mangle]
pub extern "C" fn almi_last_error(context: *const AlmiContext) -> *const c_char {
    if context.is_null() {
        return ptr::null();
    }
    catch_unwind(AssertUnwindSafe(|| unsafe {
        let guard = (*context).last_error.lock().ok()?;
        guard.as_ref().map(|value| value.as_ptr())
    }))
    .ok()
    .flatten()
    .unwrap_or(ptr::null())
}

#[no_mangle]
pub extern "C" fn almi_workspace_validate(context: *mut AlmiContext, path: *const c_char) -> i32 {
    ffi_call(context, || {
        almi_continuity::validate_workspace(c_path(path)?)?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn almi_cosmos_verify_json(
    context: *mut AlmiContext,
    path: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    json_call(context, path, out_json, almi_cosmos::verify_bundle)
}

#[no_mangle]
pub extern "C" fn almi_cosmos_inspect_json(
    context: *mut AlmiContext,
    path: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    json_call(context, path, out_json, almi_cosmos::inspect_bundle)
}

#[no_mangle]
pub extern "C" fn almi_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(CString::from_raw(value));
    }));
}

fn json_call<F>(
    context: *mut AlmiContext,
    path: *const c_char,
    out_json: *mut *mut c_char,
    operation: F,
) -> i32
where
    F: FnOnce(PathBuf) -> Result<almi_core::CosmosBundleMetadata, AlmiError>,
{
    if out_json.is_null() {
        return set_error(context, ALMI_ERROR_NULL, "out_json must not be null");
    }
    unsafe {
        *out_json = ptr::null_mut();
    }
    ffi_call(context, || {
        let metadata = operation(c_path(path)?)?;
        let json = serde_json::to_string(&metadata)?;
        let string = CString::new(json)
            .map_err(|_| AlmiError::Integrity("metadata contained an interior NUL".into()))?;
        unsafe {
            *out_json = string.into_raw();
        }
        Ok(())
    })
}

fn ffi_call(context: *mut AlmiContext, operation: impl FnOnce() -> Result<(), AlmiError>) -> i32 {
    if context.is_null() {
        return ALMI_ERROR_NULL;
    }
    clear_error(context);
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => ALMI_OK,
        Ok(Err(error)) => map_error(context, &error),
        Err(_) => set_error(
            context,
            ALMI_ERROR_PANIC,
            "panic contained at C ABI boundary",
        ),
    }
}

fn c_path(path: *const c_char) -> Result<PathBuf, AlmiError> {
    if path.is_null() {
        return Err(AlmiError::InvalidInput("path must not be null".into()));
    }
    let text = unsafe { CStr::from_ptr(path) }
        .to_str()
        .map_err(|_| AlmiError::InvalidInput("path must be valid UTF-8".into()))?;
    if text.is_empty() {
        return Err(AlmiError::InvalidInput("path must not be empty".into()));
    }
    Ok(PathBuf::from(text))
}

fn map_error(context: *mut AlmiContext, error: &AlmiError) -> i32 {
    let code = match error {
        AlmiError::InvalidInput(_) => ALMI_ERROR_INVALID,
        AlmiError::Integrity(_) => ALMI_ERROR_INTEGRITY,
        AlmiError::Security(_) => ALMI_ERROR_SECURITY,
        AlmiError::UnsupportedVersion(_) => ALMI_ERROR_UNSUPPORTED_VERSION,
        AlmiError::Io(_) => ALMI_ERROR_IO,
        AlmiError::Json(_) | AlmiError::Provider(_) => ALMI_ERROR_INTERNAL,
    };
    set_error(context, code, &error.to_string())
}

fn set_error(context: *mut AlmiContext, code: i32, message: &str) -> i32 {
    if !context.is_null() {
        let safe = message.replace('\0', "�");
        if let Ok(value) = CString::new(safe) {
            if let Ok(mut slot) = unsafe { (*context).last_error.lock() } {
                *slot = Some(value);
            }
        }
    }
    code
}

fn clear_error(context: *mut AlmiContext) {
    if !context.is_null() {
        if let Ok(mut slot) = unsafe { (*context).last_error.lock() } {
            *slot = None;
        }
    }
}
