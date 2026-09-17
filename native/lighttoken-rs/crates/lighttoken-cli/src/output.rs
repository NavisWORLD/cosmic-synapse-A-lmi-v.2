use serde_json::Value;

pub fn print_success(json: bool, value: &Value, text: &str) -> Result<(), serde_json::Error> {
    if json {
        println!("{}", serde_json::to_string(value)?);
    } else {
        println!("{text}");
    }
    Ok(())
}

pub fn print_error(json: bool, category: &str, message: &str) {
    if json {
        let payload = serde_json::json!({
            "error": true,
            "category": category,
            "message": message,
        });
        eprintln!("{payload}");
    } else {
        eprintln!("{category}: {message}");
    }
}
