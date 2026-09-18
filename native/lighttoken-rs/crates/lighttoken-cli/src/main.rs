mod output;

use clap::{Parser, Subcommand, ValueEnum};
use lighttoken_core::{
    LightTokenError, SimilarityMethod, EMBEDDING_DIMENSION, LIGHTTOKEN_SCHEMA_VERSION,
    SPECTRAL_DIMENSION,
};
use lighttoken_index::{IndexError, SearchRequest, TokenCollection};
use lighttoken_io::{load_collection, load_token_file, IoError};
use lighttoken_spectrum::{
    backend_diagnostics, dominant_bin, spectral_power, token_similarity, BackendKind, SpectrumError,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const BACKEND: &str = "rust";
const INDEX_FILENAME: &str = "tokens.jsonl";

fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Rust => "rust",
        BackendKind::Cpp => "cpp",
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "lighttoken",
    about = "Native LightToken diagnostics and search"
)]
struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Doctor,
    Version,
    Inspect {
        path: PathBuf,
        #[arg(long)]
        write_canonical: Option<PathBuf>,
    },
    Validate {
        path: PathBuf,
    },
    Spectrum {
        path: PathBuf,
    },
    Compare {
        a: PathBuf,
        b: PathBuf,
        #[arg(long, value_enum)]
        method: MethodArg,
    },
    Index {
        #[command(subcommand)]
        command: IndexCommands,
    },
    Search {
        index_dir: PathBuf,
        query: PathBuf,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long, value_enum, default_value_t = MethodArg::PowerCorrelation)]
        method: MethodArg,
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long)]
        modality: Option<String>,
        #[arg(long)]
        source_prefix: Option<String>,
    },
    Backend,
}

#[derive(Debug, Subcommand)]
enum IndexCommands {
    Build { input: PathBuf, index_dir: PathBuf },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum MethodArg {
    #[value(name = "power_correlation", alias = "correlation")]
    PowerCorrelation,
    Cosine,
    Euclidean,
}

impl MethodArg {
    fn core(self) -> SimilarityMethod {
        match self {
            Self::PowerCorrelation => SimilarityMethod::PowerCorrelation,
            Self::Cosine => SimilarityMethod::Cosine,
            Self::Euclidean => SimilarityMethod::Euclidean,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::PowerCorrelation => "power_correlation",
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
        }
    }
}

#[derive(Debug)]
struct AppError {
    category: &'static str,
    message: String,
}

impl AppError {
    fn new(category: &'static str, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
        }
    }
}

impl From<IoError> for AppError {
    fn from(error: IoError) -> Self {
        let category = match &error {
            IoError::Io(_) => "io",
            IoError::Token(LightTokenError::Unsupported(_)) => "unsupported_version",
            IoError::Token(_) | IoError::Json(_) => "invalid_token",
            IoError::Almi(_) => "almi",
            IoError::Security(_) => "security",
            IoError::Limit(_) => "limit",
        };
        Self::new(category, error.to_string())
    }
}

impl From<SpectrumError> for AppError {
    fn from(error: SpectrumError) -> Self {
        Self::new("invalid_token", error.to_string())
    }
}

impl From<IndexError> for AppError {
    fn from(error: IndexError) -> Self {
        Self::new("invalid_request", error.to_string())
    }
}

impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        Self::new("io", error.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(error: serde_json::Error) -> Self {
        Self::new("internal", error.to_string())
    }
}

fn token_summary(token: &lighttoken_core::LightTokenRecord) -> Value {
    json!({
        "token_id": token.token_id,
        "timestamp": token.timestamp,
        "source_uri": token.source_uri,
        "modality": token.modality,
        "raw_data_ref": token.raw_data_ref,
        "embedding_dimension": token.joint_embedding.as_ref().map(Vec::len).unwrap_or(0),
        "spectral_dimension": token.spectral_dimension().unwrap_or(0),
        "spectral_transform": token.spectral_transform(),
    })
}

fn ensure_empty_or_new_directory(path: &Path) -> Result<(), AppError> {
    if path.exists() {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(AppError::new(
                "security",
                format!(
                    "index destination is not a regular directory: {}",
                    path.display()
                ),
            ));
        }
        if fs::read_dir(path)?.next().transpose()?.is_some() {
            return Err(AppError::new(
                "security",
                format!("index destination is not empty: {}", path.display()),
            ));
        }
    } else {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn write_index(input: &Path, index_dir: &Path) -> Result<Value, AppError> {
    let tokens = load_collection(input)?;
    TokenCollection::from_tokens(tokens.clone())?;
    ensure_empty_or_new_directory(index_dir)?;
    let index_path = index_dir.join(INDEX_FILENAME);
    let mut bytes = Vec::new();
    for token in &tokens {
        let encoded = token
            .canonical_json_bytes()
            .map_err(|error| AppError::new("invalid_token", error.to_string()))?;
        bytes.extend_from_slice(&encoded);
    }
    fs::write(&index_path, bytes)?;
    Ok(json!({
        "built": true,
        "token_count": tokens.len(),
        "index_dir": index_dir.display().to_string(),
        "index_file": index_path.display().to_string(),
        "backend": BACKEND,
    }))
}

fn run(cli: &Cli) -> Result<(Value, String), AppError> {
    match &cli.command {
        Commands::Doctor => {
            let diagnostics = backend_diagnostics();
            let payload = json!({
                "ok": true,
                "schema_version": LIGHTTOKEN_SCHEMA_VERSION,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "spectral_dimension": SPECTRAL_DIMENSION,
                "rust_backend_available": true,
                "cpp_backend_available": diagnostics.cpp_available,
                "active_backend": backend_name(diagnostics.active),
                "cpp_backend_detail": diagnostics.detail,
                "almi_integration_available": true,
            });
            Ok((payload, "LightToken native diagnostics: OK".into()))
        }
        Commands::Version => {
            let version = env!("CARGO_PKG_VERSION");
            Ok((json!({"version": version}), version.into()))
        }
        Commands::Inspect {
            path,
            write_canonical,
        } => {
            let token = load_token_file(path)?;
            let mut payload = token_summary(&token);
            let canonical_written = if let Some(destination) = write_canonical {
                if let Some(parent) = destination.parent() {
                    if !parent.as_os_str().is_empty() {
                        fs::create_dir_all(parent)?;
                    }
                }
                let bytes = token
                    .canonical_json_bytes()
                    .map_err(|error| AppError::new("unsupported_version", error.to_string()))?;
                fs::write(destination, bytes)?;
                true
            } else {
                false
            };
            if let Value::Object(object) = &mut payload {
                object.insert("canonical_written".into(), Value::Bool(canonical_written));
            }
            Ok((payload, format!("LightToken {}", token.token_id)))
        }
        Commands::Validate { path } => {
            let token = load_token_file(path)?;
            let payload = json!({
                "valid": true,
                "token_id": token.token_id,
                "embedding_dimension": token.joint_embedding.as_ref().map(Vec::len).unwrap_or(0),
                "spectral_dimension": token.spectral_dimension().unwrap_or(0),
            });
            Ok((payload, "valid".into()))
        }
        Commands::Spectrum { path } => {
            let token = load_token_file(path)?;
            let bins = token.spectral_signature.as_deref().ok_or_else(|| {
                AppError::new("invalid_token", "LightToken has no spectral signature")
            })?;
            let power = spectral_power(bins)?;
            let (bin, magnitude) = dominant_bin(&power)?;
            let payload = json!({
                "token_id": token.token_id,
                "power_length": power.len(),
                "dominant_bin": bin,
                "dominant_magnitude": magnitude,
                "backend": BACKEND,
            });
            Ok((payload, format!("dominant bin {bin}: {magnitude}")))
        }
        Commands::Compare { a, b, method } => {
            let left = load_token_file(a)?;
            let right = load_token_file(b)?;
            let score = token_similarity(&left, &right, method.core())?;
            let payload = json!({
                "a_token_id": left.token_id,
                "b_token_id": right.token_id,
                "method": method.as_str(),
                "score": score,
                "backend": BACKEND,
            });
            Ok((payload, format!("{}: {score}", method.as_str())))
        }
        Commands::Index { command } => match command {
            IndexCommands::Build { input, index_dir } => {
                let payload = write_index(input, index_dir)?;
                Ok((payload, format!("index built at {}", index_dir.display())))
            }
        },
        Commands::Search {
            index_dir,
            query,
            top_k,
            method,
            threshold,
            modality,
            source_prefix,
        } => {
            let index_path = index_dir.join(INDEX_FILENAME);
            let tokens = load_collection(&index_path)?;
            let collection = TokenCollection::from_tokens(tokens)?;
            let query_token = load_token_file(query)?;
            let request = SearchRequest {
                method: method.core(),
                top_k: Some(*top_k),
                threshold: *threshold,
                modality: modality.clone(),
                source_prefix: source_prefix.clone(),
            };
            let hits = collection.search(&query_token, &request)?;
            let payload = json!({
                "query_token_id": query_token.token_id,
                "method": method.as_str(),
                "backend": BACKEND,
                "hits": hits,
            });
            Ok((payload, format!("{} hits", hits.len())))
        }
        Commands::Backend => {
            let diagnostics = backend_diagnostics();
            let active = backend_name(diagnostics.active);
            let payload = json!({
                "active": active,
                "rust": {"available": true},
                "cpp": {"available": diagnostics.cpp_available, "detail": diagnostics.detail},
            });
            Ok((payload, active.into()))
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match run(&cli) {
        Ok((payload, text)) => {
            if let Err(error) = output::print_success(cli.json, &payload, &text) {
                output::print_error(cli.json, "internal", &error.to_string());
                std::process::exit(1);
            }
        }
        Err(error) => {
            output::print_error(cli.json, error.category, &error.message);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands, IndexCommands};
    use clap::Parser;

    #[test]
    fn parses_doctor() {
        let cli = Cli::try_parse_from(["lighttoken", "doctor", "--json"]).unwrap();
        assert!(matches!(cli.command, Commands::Doctor));
        assert!(cli.json);
    }

    #[test]
    fn parses_version() {
        let cli = Cli::try_parse_from(["lighttoken", "version", "--json"]).unwrap();
        assert!(matches!(cli.command, Commands::Version));
    }

    #[test]
    fn parses_inspect() {
        let cli = Cli::try_parse_from([
            "lighttoken",
            "inspect",
            "token.json",
            "--write-canonical",
            "canonical.json",
            "--json",
        ])
        .unwrap();
        assert!(matches!(cli.command, Commands::Inspect { .. }));
    }

    #[test]
    fn parses_validate() {
        let cli = Cli::try_parse_from(["lighttoken", "validate", "token.json", "--json"]).unwrap();
        assert!(matches!(cli.command, Commands::Validate { .. }));
    }

    #[test]
    fn parses_spectrum() {
        let cli = Cli::try_parse_from(["lighttoken", "spectrum", "token.json", "--json"]).unwrap();
        assert!(matches!(cli.command, Commands::Spectrum { .. }));
    }

    #[test]
    fn parses_compare_all_methods() {
        for method in ["power_correlation", "cosine", "euclidean"] {
            let cli = Cli::try_parse_from([
                "lighttoken",
                "compare",
                "a.json",
                "b.json",
                "--method",
                method,
                "--json",
            ])
            .unwrap();
            assert!(matches!(cli.command, Commands::Compare { .. }));
        }
    }

    #[test]
    fn parses_correlation_alias() {
        let cli = Cli::try_parse_from([
            "lighttoken",
            "compare",
            "a.json",
            "b.json",
            "--method",
            "correlation",
            "--json",
        ])
        .unwrap();
        assert!(matches!(cli.command, Commands::Compare { .. }));
    }

    #[test]
    fn parses_index_build() {
        let cli = Cli::try_parse_from([
            "lighttoken",
            "index",
            "build",
            "tokens.jsonl",
            "index-dir",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Commands::Index {
                command: IndexCommands::Build { .. }
            }
        ));
    }

    #[test]
    fn parses_search() {
        let cli = Cli::try_parse_from([
            "lighttoken",
            "search",
            "index-dir",
            "query.json",
            "--top-k",
            "10",
            "--method",
            "cosine",
            "--json",
        ])
        .unwrap();
        assert!(matches!(cli.command, Commands::Search { .. }));
    }

    #[test]
    fn parses_backend() {
        let cli = Cli::try_parse_from(["lighttoken", "backend", "--json"]).unwrap();
        assert!(matches!(cli.command, Commands::Backend));
    }
}
