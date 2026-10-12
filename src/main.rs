use anyhow::Result;

use kortex::chat::{list_available_models, print_cache_table, run_chat_repl, BANNER};

fn print_cli_help() {
    print!("{}", BANNER);
    println!("Usage: \x1b[1;36mkortex\x1b[0m [COMMAND | MODEL_NAME_OR_PATH]\n");
    println!("Commands:");
    println!("  \x1b[1;32mchat\x1b[0m [MODEL]       Launch interactive typewriter chat REPL (Default)");
    println!("  \x1b[1;32mmodels\x1b[0m, \x1b[1;32mlist\x1b[0m       List all locally available GGUF models");
    println!("  \x1b[1;32mcache\x1b[0m              Inspect SHA-256 model profile database (~/.kortex/)");
    println!("  \x1b[1;32mautotune\x1b[0m [MODEL]   Run auto-tuner to benchmark and cache champion runner");
    println!("  \x1b[1;32mhelp\x1b[0m, \x1b[1;32m--help\x1b[0m, \x1b[1;32m-h\x1b[0m   Display this CLI help message\n");
    println!("Quick Examples:");
    println!("  \x1b[2mkortex\x1b[0m                    # Launch chat with default model");
    println!("  \x1b[2mkortex 0.5b\x1b[0m               # Launch chat with Qwen 2.5 0.5B");
    println!("  \x1b[2mkortex llama\x1b[0m              # Launch chat with Llama 3.2 1B");
    println!("  \x1b[2mkortex /path/to/model.gguf\x1b[0m # Launch chat with custom model");
    println!("  \x1b[2mkortex models\x1b[0m             # Show available models in ~/.kortex/models/\n");
}

fn print_cache_status() {
    print_cache_table();
}

fn print_models_list() {
    let models = list_available_models();
    println!("\nAvailable GGUF models on disk (models/ or ~/.kortex/models/):");
    if models.is_empty() {
        println!("  (No models found. Place GGUF models in ~/.kortex/models/)");
    } else {
        for (i, m) in models.iter().enumerate() {
            println!("  {}. \x1b[1;36m{}\x1b[0m ({})", i + 1, m.file_name().unwrap().to_string_lossy(), m.display());
        }
    }
    println!("\nLaunch with `kortex <name>`\n");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() <= 1 {
        // Default action: launch chat REPL
        return run_chat_repl(None);
    }

    match args[1].as_str() {
        "help" | "--help" | "-h" => {
            print_cli_help();
            Ok(())
        }
        "models" | "list" => {
            print_models_list();
            Ok(())
        }
        "cache" => {
            print_cache_status();
            Ok(())
        }
        "chat" => {
            let model_arg = args.get(2).map(|s| s.as_str());
            run_chat_repl(model_arg)
        }
        "autotune" => {
            kortex::engine::silence_llama_logs();
            let models = list_available_models();
            let target_path = if let Some(m_arg) = args.get(2) {
                kortex::chat::resolve_model_path(Some(m_arg))?
            } else {
                if models.is_empty() {
                    anyhow::bail!("No models found to autotune.");
                }
                models[0].clone()
            };

            println!("Running AutoTuner on model: {}", target_path.display());
            let tuner = kortex::autotune::AutoTuner::new(kortex::autotune::BenchmarkProfile::Balanced);
            let report = tuner.tune_model(&target_path)?;
            let winner = report.winner();
            println!("\nTuning complete. Winner: {} with precision {}", winner.method_name, winner.kv_precision);
            Ok(())
        }
        model_or_cmd => {
            // Assume argument is model name or path
            run_chat_repl(Some(model_or_cmd))
        }
    }
}
