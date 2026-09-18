use crate::{
    backend_json_text, compare_json_text, search_json_text, validate_json_text, LightTokenContext,
    LIGHTTOKEN_ABI_VERSION,
};
use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use std::ptr;

fn java_string(env: &mut JNIEnv<'_>, value: JString<'_>) -> Result<String, String> {
    env.get_string(&value)
        .map(|text| text.into())
        .map_err(|error| error.to_string())
}

fn return_json(env: &mut JNIEnv<'_>, result: Result<String, impl std::fmt::Display>) -> jstring {
    match result {
        Ok(value) => match env.new_string(value) {
            Ok(value) => value.into_raw(),
            Err(error) => {
                let _ = env.throw_new("java/lang/RuntimeException", error.to_string());
                ptr::null_mut()
            }
        },
        Err(error) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", error.to_string());
            ptr::null_mut()
        }
    }
}

fn valid_context(handle: jlong) -> bool {
    handle != 0
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeAbiVersion(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jint {
    LIGHTTOKEN_ABI_VERSION as jint
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeCreateContext(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jlong {
    Box::into_raw(Box::new(LightTokenContext::default())) as jlong
}

/// # Safety
/// `handle` must be zero or a context handle returned by
/// `nativeCreateContext` that has not already been freed.
#[no_mangle]
pub unsafe extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeFreeContext(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    if valid_context(handle) {
        drop(Box::from_raw(handle as *mut LightTokenContext));
    }
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeValidateJson(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    json: JString<'_>,
) -> jstring {
    if !valid_context(handle) {
        let _ = env.throw_new("java/lang/IllegalStateException", "native context is closed");
        return ptr::null_mut();
    }
    match java_string(&mut env, json) {
        Ok(json) => return_json(&mut env, validate_json_text(&json)),
        Err(error) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", error);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeCompareJson(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    left: JString<'_>,
    right: JString<'_>,
    method: JString<'_>,
) -> jstring {
    if !valid_context(handle) {
        let _ = env.throw_new("java/lang/IllegalStateException", "native context is closed");
        return ptr::null_mut();
    }
    let result = (|| {
        let left = java_string(&mut env, left)?;
        let right = java_string(&mut env, right)?;
        let method = java_string(&mut env, method)?;
        compare_json_text(&left, &right, &method).map_err(|error| error.to_string())
    })();
    return_json(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeSearchJson(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    query: JString<'_>,
    collection: JString<'_>,
    request: JString<'_>,
) -> jstring {
    if !valid_context(handle) {
        let _ = env.throw_new("java/lang/IllegalStateException", "native context is closed");
        return ptr::null_mut();
    }
    let result = (|| {
        let query = java_string(&mut env, query)?;
        let collection = java_string(&mut env, collection)?;
        let request = java_string(&mut env, request)?;
        search_json_text(&query, &collection, &request).map_err(|error| error.to_string())
    })();
    return_json(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_world_navis_lighttoken_nativebridge_JniNativeEngine_nativeBackendJson(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jstring {
    if !valid_context(handle) {
        let _ = env.throw_new("java/lang/IllegalStateException", "native context is closed");
        return ptr::null_mut();
    }
    return_json(&mut env, backend_json_text())
}
