//! Interactive Terminal REPL & Multi-Turn Session for Kortex.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use anyhow::{bail, Result};

use crate::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use crate::engines::hybrid_sink::HybridSinkEngine;
use crate::model_cache::ModelCache;

pub const BANNER: &str = r#"
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                     KORTEX INTERACTIVE CHAT REPL                          ║
 ║          High-Throughput Edge Inference Engine with Hash Optimization     ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
"#;

pub struct ChatSession {
    pub engine: HybridSinkEngine,
    pub model_name: String,
    pub model_hash: String,
    pub model_path: PathBuf,
    pub history: Vec<(String, String)>, // (user, assistant)
    pub show_stats: bool,
    pub is_llama: bool,
}

impl ChatSession {
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let mut cache = ModelCache::load();
        let (hash, cache_entry) = cache.get_or_register(&model_path)?;
        let file_name = model_path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let is_llama = file_name.to_lowercase().contains("llama");

        let runner_prof = &cache_entry.optimal_runner;

        println!("╭─────────────────────────────────────────────────────────────────────────────╮");
        println!("│ Model:      \x1b[1;36m{:<62}\x1b[0m│", file_name);
        println!("│ Hash:       \x1b[1;33m{:<62}\x1b[0m│", &hash[..32]);
        println!("│ Runner:     \x1b[1;32m{:<62}\x1b[0m│", runner_prof.runner_type.label());
        println!("│ KV Config:  \x1b[1;35mPrecision: {}, Sinks: {}, Window: {}, Chunk: {:<17}\x1b[0m│", 
            runner_prof.kv_precision, runner_prof.sink_tokens, runner_prof.window_size, runner_prof.evict_chunk);
        println!("│ Status:     \x1b[1;32m[Cached Champion Runner Profile Loaded from Hash Database]     \x1b[0m│");
        println!("╰─────────────────────────────────────────────────────────────────────────────╯");

        let precision = match runner_prof.kv_precision.as_str() {
            "Q8_0" => KvPrecision::Q8_0,
            "Q4_0" => KvPrecision::Q4_0,
            _ => KvPrecision::F16,
        };

        let config = EngineConfig::new(model_path.clone())
            .with_gpu_layers(999)
            .with_ctx_size(2048)
            .with_device(Some(1));

        let engine = HybridSinkEngine::new(
            config,
            precision,
            runner_prof.sink_tokens,
            runner_prof.window_size,
            runner_prof.evict_chunk,
        )?;

        Ok(Self {
            engine,
            model_name: file_name,
            model_hash: hash,
            model_path,
            history: Vec::new(),
            show_stats: false,
            is_llama,
        })
    }

    pub fn build_prompt(&self, latest_user_msg: &str) -> String {
        let mut prompt = String::new();

        if self.is_llama {
            // Llama 3 / 3.2 formatting
            prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, concise AI assistant.<|eot_id|>");
            for (u, a) in &self.history {
                prompt.push_str(&format!("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>", u));
                prompt.push_str(&format!("<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>", a));
            }
            prompt.push_str(&format!("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", latest_user_msg));
        } else {
            // ChatML formatting (Qwen, etc.)
            prompt.push_str("<|im_start|>system\nYou are a helpful, concise AI assistant.<|im_end|>\n");
            for (u, a) in &self.history {
                prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n", u));
                prompt.push_str(&format!("<|im_start|>assistant\n{}<|im_end|>\n", a));
            }
            prompt.push_str(&format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", latest_user_msg));
        }

        prompt
    }

    pub fn print_info(&self) {
        let mem = self.engine.memory_profile();
        println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║                             LIVE MODEL SYSTEM INFO                        ║");
        println!("╠══════════════════════════╦════════════════════════════════════════════════╣");
        println!("║ Model Name               ║ {:<46} ║", self.model_name);
        println!("║ SHA-256 Hash             ║ {:<46} ║", &self.model_hash[..32]);
        println!("║ Parameters               ║ {:>42.2} B ║", mem.total_params as f64 / 1e9);
        println!("║ Base Weights VRAM        ║ {:>40.2} MiB ║", mem.weights_mib());
        println!("║ Transformer Depth        ║ {:>40} layers ║", mem.n_layers);
        println!("║ Attention Heads (Q / KV) ║ {:>35} Q / {:>2} KV ║", mem.n_heads, mem.n_kv_heads);
        println!("║ Head Dimension           ║ {:>42} ║", mem.head_dim);
        println!("║ Active KV Footprint      ║ {:>38.2} KiB/tok ║", mem.kv_kib_per_token());
        println!("║ Bounded Working Set      ║ {:>40.2} MiB ║", mem.total_memory_mib());
        println!("║ Active Device            ║ {:<46} ║", self.engine.device_name());
        println!("╚══════════════════════════╩════════════════════════════════════════════════╝\n");
    }

    pub fn step(&mut self, user_input: &str) -> Result<()> {
        let full_prompt = self.build_prompt(user_input);

        let params = GenerationParams {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            seed: rand_seed(),
        };

        let mut on_token = |piece: &str| -> bool {
            print!("{}", piece);
            let _ = io::stdout().flush();
            true
        };

        print!("\x1b[1;32mAssistant:\x1b[0m ");
        io::stdout().flush()?;

        let output = self.engine.generate(&full_prompt, &params, Some(&mut on_token))?;
        println!();

        let assistant_reply = output.text;

        if self.show_stats {
            let r = &output.report;
            println!(
                "\x1b[2m[{:.1} tok/s | TTFT: {:.1} ms | P50: {:.2} ms | KV: {:.2} MiB | {} tokens]\x1b[0m\n",
                r.generation_tok_per_sec,
                r.time_to_first_token.as_secs_f64() * 1000.0,
                r.latency_p50_ms,
                output.memory.kv_cache_mib(),
                output.generated_tokens
            );
        } else {
            println!();
        }

        self.history.push((user_input.to_string(), assistant_reply));
        Ok(())
    }
}

pub fn rand_seed() -> u32 {
    let now = std::time::SystemTime::now();
    let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    (duration.as_nanos() % (u32::MAX as u128)) as u32
}

pub fn list_available_models() -> Vec<PathBuf> {
    let mut models = Vec::new();
    let mut search_dirs = vec![
        PathBuf::from("models"),
        PathBuf::from("../models"),
        PathBuf::from("../../models"),
    ];

    if let Ok(home) = std::env::var("HOME") {
        search_dirs.push(PathBuf::from(&home).join(".kortex").join("models"));
        search_dirs.push(PathBuf::from(&home).join("Projects/larvance/analysis/gpuresearch/models"));
    }

    for dir in search_dirs {
        if dir.exists() {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "gguf") {
                        if !models.iter().any(|m: &PathBuf| m.file_name() == path.file_name()) {
                            models.push(path);
                        }
                    }
                }
            }
        }
    }
    models.sort();
    models
}

pub fn print_help() {
    println!("\x1b[1mAvailable Interactive Commands:\x1b[0m");
    println!("  \x1b[1;36m/help\x1b[0m    - Show this command reference");
    println!("  \x1b[1;36m/info\x1b[0m    - Display active model architecture, KV cache footprint & VRAM status");
    println!("  \x1b[1;36m/stats\x1b[0m   - Toggle per-response speed and latency statistics");
    println!("  \x1b[1;36m/clear\x1b[0m   - Reset conversation context memory");
    println!("  \x1b[1;36m/cache\x1b[0m   - Inspect all cached model hashes and their champion runner profiles");
    println!("  \x1b[1;36m/models\x1b[0m  - List all available GGUF models on disk");
    println!("  \x1b[1;36m/exit\x1b[0m    - Exit chat (also: /quit or Ctrl+D)\n");
}

pub fn resolve_model_path(arg_opt: Option<&str>) -> Result<PathBuf> {
    let available_models = list_available_models();

    if let Some(arg) = arg_opt {
        let p = PathBuf::from(arg);
        if p.exists() {
            return Ok(p);
        }

        let mut candidates = vec![
            PathBuf::from("models").join(arg),
            PathBuf::from("models").join(format!("{}.gguf", arg)),
        ];

        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(&home).join(".kortex").join("models").join(arg));
            candidates.push(PathBuf::from(&home).join(".kortex").join("models").join(format!("{}.gguf", arg)));
        }

        for c in candidates {
            if c.exists() {
                return Ok(c);
            }
        }

        // Substring search across available models (e.g. "0.5b", "llama", "7b")
        let arg_lower = arg.to_lowercase();
        if let Some(matched) = available_models.iter().find(|m| m.to_string_lossy().to_lowercase().contains(&arg_lower)) {
            return Ok(matched.clone());
        }

        bail!("Model file not found matching {:?}. Run `kortex models` to see available models.", arg);
    } else {
        if available_models.is_empty() {
            bail!("No GGUF models found in models/ or ~/.kortex/models/. Please download one first.");
        }
        // Default to Qwen 0.5B for fast testing or first available
        let best = available_models
            .iter()
            .find(|p| p.to_string_lossy().contains("0.5b"))
            .cloned()
            .unwrap_or_else(|| available_models[0].clone());
        Ok(best)
    }
}

pub fn run_chat_repl(model_arg: Option<&str>) -> Result<()> {
    crate::engine::silence_llama_logs();
    print!("{}", BANNER);

    let model_path = resolve_model_path(model_arg)?;
    let mut session = ChatSession::new(model_path)?;

    println!("Type your message below. Type \x1b[1;36m/help\x1b[0m for commands or \x1b[1;36m/exit\x1b[0m to quit.\n");

    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("\x1b[1;36m>>>\x1b[0m ");
        io::stdout().flush()?;

        line.clear();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            println!("\nGoodbye!");
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "/exit" | "/quit" => {
                println!("Goodbye!");
                break;
            }
            "/help" => {
                print_help();
            }
            "/info" => {
                session.print_info();
            }
            "/stats" => {
                session.show_stats = !session.show_stats;
                println!(
                    "Per-response statistics: \x1b[1;33m{}\x1b[0m\n",
                    if session.show_stats { "ENABLED" } else { "DISABLED" }
                );
            }
            "/clear" => {
                let turns = session.history.len();
                session.history.clear();
                println!("Conversation context cleared! ({} previous turns erased)\n", turns);
            }
            "/cache" => {
                let cache = ModelCache::load();
                println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
                println!("║                     OPTIMIZER CACHE HASH DATABASE                         ║");
                println!("╠═══════════════════════════════════════════════════════════════════════════╣");
                for (h, entry) in cache.all_entries() {
                    println!("║ Hash:   \x1b[1;33m{:<65}\x1b[0m ║", &h[..32]);
                    println!("║ File:   {:<65} ║", Path::new(&entry.model_path).file_name().unwrap_or_default().to_string_lossy());
                    println!("║ Runner: {:<65} ║", entry.optimal_runner.runner_type.label());
                    println!("║ Score:  {:<65.1} ║", entry.optimal_runner.efficiency_score);
                    println!("╟───────────────────────────────────────────────────────────────────────────╢");
                }
                println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");
            }
            "/models" => {
                let models = list_available_models();
                println!("\nAvailable models in models/ or ~/.kortex/models/:");
                for (i, m) in models.iter().enumerate() {
                    let active = if m == &session.model_path { " \x1b[1;32m[ACTIVE]\x1b[0m" } else { "" };
                    println!("  {}. {}{}", i + 1, m.file_name().unwrap().to_string_lossy(), active);
                }
                println!("\nTip: Launch with `kortex <model>`\n");
            }
            other if other.starts_with('/') => {
                println!("\x1b[1;31mUnknown command:\x1b[0m {}. Type \x1b[1;36m/help\x1b[0m for available commands.\n", other);
            }
            user_msg => {
                if let Err(e) = session.step(user_msg) {
                    eprintln!("\x1b[1;31mInference Error:\x1b[0m {:?}\n", e);
                }
            }
        }
    }

    Ok(())
}
