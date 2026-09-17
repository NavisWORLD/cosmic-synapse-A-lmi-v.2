//! Native LightToken CLI parser contract. Production command types are added after RED verification.

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
