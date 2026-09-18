use crate::{
    backend_json_text, compare_json_text, search_json_text, validate_json_text, EngineError,
    LightTokenContext, LIGHTTOKEN_ABI_VERSION,
};
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub const LIGHTTOKEN_OK: i32 = 0;
pub const LIGHTTOKEN_INVALID_ARGUMENT: i32 = 1;
pub const LIGHTTOKEN_INVALID_TOKEN: i32 = 2;
pub const LIGHTTOKEN_IO: i32 = 3;
pub const LIGHTTOKEN_UNSUPPORTED_VERSION: i32 = 4;
pub const LIGHTTOKEN_BACKEND: i32 = 5;
pub const LIGHTTOKEN_PANIC_CONTAINED: i32 = 255;

fn status(error: &EngineError) -> i32 {
    match error {
        EngineError::InvalidArgument(_) | EngineError::Json(_) => LIGHTTOKEN_INVALID_ARGUMENT,
        EngineError::InvalidToken(_) => LIGHTTOKEN_INVALID_TOKEN,
        EngineError::UnsupportedVersion(_) => LIGHTTOKEN_UNSUPPORTED_VERSION,
        EngineError::Backend(_) => LIGHTTOKEN_BACKEND,
    }
}

unsafe fn input<'a>(pointer: *const c_char, label: &str) -> Result<&'a str, EngineError> {
    if pointer.is_null() {
        return Err(EngineError::InvalidArgument(format!("{label} is null")));
    }
    CStr::from_ptr(pointer)
        .to_str()
        .map_err(|_| EngineError::InvalidArgument(format!("{label} is not valid UTF-8")))
}

unsafe fn write_result<F>(
    context: *mut LightTokenContext,
    out_json: *mut *mut c_char,
    operation: F,
) -> i32
where
    F: FnOnce() -> Result<String, EngineError>,
{
    if context.is_null() || out_json.is_null() {
        return LIGHTTOKEN_INVALID_ARGUMENT;
    }
    *out_json = ptr::null_mut();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(json)) => match CString::new(json) {
            Ok(value) => {
                *out_json = value.into_raw();
                LIGHTTOKEN_OK
            }
            Err(_) => LIGHTTOKEN_INVALID_ARGUMENT,
        },
        Ok(Err(error)) => status(&error),
        Err(_) => LIGHTTOKEN_PANIC_CONTAINED,
    }
}

#[no_mangle]
pub extern "C" fn lighttoken_abi_version() -> u32 {
    LIGHTTOKEN_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn lighttoken_context_new() -> *mut LightTokenContext {
    match catch_unwind(|| Box::into_raw(Box::new(LightTokenContext::default()))) {
        Ok(pointer) => pointer,
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `context` must be null or a pointer returned by `lighttoken_context_new`
/// that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_context_free(context: *mut LightTokenContext) {
    if context.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(context))));
}

/// # Safety
/// All non-null pointers must be valid for the duration of this call. The
/// output string, when returned, must be released with `lighttoken_string_free`.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_validate_json(
    context: *mut LightTokenContext,
    json_utf8: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if context.is_null() || json_utf8.is_null() || out_json.is_null() {
        return LIGHTTOKEN_INVALID_ARGUMENT;
    }
    write_result(context, out_json, || {
        let json = input(json_utf8, "json_utf8")?;
        validate_json_text(json)
    })
}

/// # Safety
/// All non-null pointers must be valid for the duration of this call. The
/// output string, when returned, must be released with `lighttoken_string_free`.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_compare_json(
    context: *mut LightTokenContext,
    left_json_utf8: *const c_char,
    right_json_utf8: *const c_char,
    method_utf8: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if context.is_null()
        || left_json_utf8.is_null()
        || right_json_utf8.is_null()
        || method_utf8.is_null()
        || out_json.is_null()
    {
        return LIGHTTOKEN_INVALID_ARGUMENT;
    }
    write_result(context, out_json, || {
        let left = input(left_json_utf8, "left_json_utf8")?;
        let right = input(right_json_utf8, "right_json_utf8")?;
        let method = input(method_utf8, "method_utf8")?;
        compare_json_text(left, right, method)
    })
}

/// # Safety
/// All non-null pointers must be valid for the duration of this call. The
/// output string, when returned, must be released with `lighttoken_string_free`.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_search_json(
    context: *mut LightTokenContext,
    query_json_utf8: *const c_char,
    collection_json_utf8: *const c_char,
    request_json_utf8: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if context.is_null()
        || query_json_utf8.is_null()
        || collection_json_utf8.is_null()
        || request_json_utf8.is_null()
        || out_json.is_null()
    {
        return LIGHTTOKEN_INVALID_ARGUMENT;
    }
    write_result(context, out_json, || {
        let query = input(query_json_utf8, "query_json_utf8")?;
        let collection = input(collection_json_utf8, "collection_json_utf8")?;
        let request = input(request_json_utf8, "request_json_utf8")?;
        search_json_text(query, collection, request)
    })
}

/// # Safety
/// `context` and `out_json` must be valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_backend_json(
    context: *mut LightTokenContext,
    out_json: *mut *mut c_char,
) -> i32 {
    write_result(context, out_json, backend_json_text)
}

/// # Safety
/// `value` must be null or a string returned by this library that has not
/// already been freed.
#[no_mangle]
pub unsafe extern "C" fn lighttoken_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(CString::from_raw(value))));
}
