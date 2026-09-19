#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The parser must return a typed error or a valid token; malformed bytes
    // must never panic or produce a partially trusted LightToken.
    let _ = lighttoken_core::from_json_bytes(data);
});
