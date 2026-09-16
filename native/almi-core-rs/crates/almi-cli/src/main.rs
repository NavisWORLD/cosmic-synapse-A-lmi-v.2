use almi_continuity::{initialize_workspace, inspect_workspace};
use almi_core::{AlmiError, Result, ABI_VERSION};
use almi_cosmos::{export_bundle, import_bundle, verify_bundle};
use almi_memory::verify_ledger;
use almi_provider::{ModelProvider, OllamaProvider, DEFAULT_OLLAMA_ENDPOINT};
use almi_runtime::PersistentRuntime;
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(name = "almi", version, about = "A-LMI Native Core continuity runtime")]
struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Doctor,
    Version,
    Init { workspace: PathBuf, #[arg(long)] name: String, #[arg(long, default_value_t = 0)] seed: i64 },
    Inspect { workspace: PathBuf },
    Export { workspace: PathBuf, bundle: PathBuf },
    Verify { bundle: PathBuf },
    Import { bundle: PathBuf, workspace: PathBuf },
    Memory { #[command(subcommand)] command: MemoryCommand },
    Provider { #[command(subcommand)] command: ProviderCommand },
    Runtime { #[command(subcommand)] command: RuntimeCommand },
}

#[derive(Debug, Subcommand)]
enum MemoryCommand { Verify { workspace: PathBuf } }

#[derive(Debug, Subcommand)]
enum ProviderCommand {
    List,
    Health { #[arg(long)] model: String, #[arg(long, default_value = DEFAULT_OLLAMA_ENDPOINT)] endpoint: String, #[arg(long, default_value_t = 30.0)] timeout: f64 },
    Run { #[arg(long)] model: String, #[arg(long)] prompt: String, #[arg(long)] system: Option<String>, #[arg(long, default_value = DEFAULT_OLLAMA_ENDPOINT)] endpoint: String, #[arg(long, default_value_t = 30.0)] timeout: f64, #[arg(long, default_value_t = 0)] retries: u32 },
}

#[derive(Debug, Subcommand)]
enum RuntimeCommand {
    Run { workspace: PathBuf, #[arg(long)] model: String, #[arg(long)] prompt: String, #[arg(long)] system: Option<String>, #[arg(long, default_value = DEFAULT_OLLAMA_ENDPOINT)] endpoint: String, #[arg(long)] revision: Option<String>, #[arg(long, default_value_t = 30.0)] timeout: f64, #[arg(long, default_value_t = 0)] retries: u32 },
}

fn main() {
    let cli = Cli::parse();
    match execute(&cli) {
        Ok(value) => emit(&value, cli.json),
        Err(error) => {
            if cli.json { eprintln!("{}", json!({"status":"ERROR","error":error.to_string()})); }
            else { eprintln!("almi: {error}"); }
            std::process::exit(2);
        }
    }
}

fn execute(cli: &Cli) -> Result<Value> {
    match &cli.command {
        Command::Doctor => Ok(json!({"status":"ok","version":env!("CARGO_PKG_VERSION"),"abi_version":ABI_VERSION,"authority_default":"deny","network_default":"loopback provider default; no network on construction","note":"software health only"})),
        Command::Version => Ok(json!({"version":env!("CARGO_PKG_VERSION"),"abi_version":ABI_VERSION})),
        Command::Init { workspace, name, seed } => Ok(serde_json::to_value(initialize_workspace(workspace, name, *seed)?)?),
        Command::Inspect { workspace } => Ok(serde_json::to_value(inspect_workspace(workspace)?)?),
        Command::Export { workspace, bundle } => Ok(serde_json::to_value(export_bundle(workspace, bundle)?)?),
        Command::Verify { bundle } => Ok(serde_json::to_value(verify_bundle(bundle)?)?),
        Command::Import { bundle, workspace } => Ok(serde_json::to_value(import_bundle(bundle, workspace)?)?),
        Command::Memory { command: MemoryCommand::Verify { workspace } } => {
            let report = verify_ledger(workspace.join("memory/ledger.jsonl"))?;
            Ok(json!({"valid":true,"records":report.records,"digests":report.digests}))
        }
        Command::Provider { command } => provider_command(command),
        Command::Runtime { command } => runtime_command(command),
    }
}

fn provider_command(command: &ProviderCommand) -> Result<Value> {
    match command {
        ProviderCommand::List => Ok(json!({"network_checked":false,"providers":[{"provider_id":"ollama","default_endpoint":DEFAULT_OLLAMA_ENDPOINT,"capabilities":["text"]}]})),
        ProviderCommand::Health { model, endpoint, timeout } => {
            let provider = build_provider(model, endpoint, None, *timeout, 0)?;
            Ok(serde_json::to_value(provider.health()?)?)
        }
        ProviderCommand::Run { model, prompt, system, endpoint, timeout, retries } => {
            let provider = build_provider(model, endpoint, None, *timeout, *retries)?;
            Ok(serde_json::to_value(provider.generate(&almi_core::ModelRequest { prompt: prompt.clone(), system: system.clone(), options: Default::default() })?)?)
        }
    }
}

fn runtime_command(command: &RuntimeCommand) -> Result<Value> {
    match command {
        RuntimeCommand::Run { workspace, model, prompt, system, endpoint, revision, timeout, retries } => {
            let provider = build_provider(model, endpoint, revision.clone(), *timeout, *retries)?;
            let mut runtime = PersistentRuntime::new(workspace, Box::new(provider))?;
            Ok(serde_json::to_value(runtime.interact(prompt, system.as_deref())?)?)
        }
    }
}

fn build_provider(model: &str, endpoint: &str, revision: Option<String>, timeout: f64, retries: u32) -> Result<OllamaProvider> {
    if !timeout.is_finite() || timeout <= 0.0 { return Err(AlmiError::InvalidInput("timeout must be positive and finite".into())); }
    OllamaProvider::with_policy(model, endpoint, revision, None, Duration::from_secs_f64(timeout), retries)
}

fn emit(value: &Value, json_mode: bool) {
    if json_mode { println!("{}", serde_json::to_string(value).expect("JSON values serialize")); }
    else if let Some(object) = value.as_object() { for (key, value) in object { println!("{key}: {value}"); } }
    else { println!("{value}"); }
}
