//! Experiment Suite: Dual-Model Speculative Decoding Benchmark.
//!
//! Evaluates speedup, acceptance rates, and memory overhead of dual-model speculative decoding:
//! Qwen 2.5 0.5B (Draft) -> Qwen 2.5 1.5B (Target) co-hosted on RTX 3050 Ti VRAM.
//!
//! Sweeps draft lengths K = 2, K = 3, K = 4 vs pure 1.5B autoregressive baseline.

use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use kortex::engines::{RawFfiEngine, SpeculativeEngine};

struct SpeculativeResult {
    label: &'static str,
    gen_tok_per_sec: f64,
    ttft_ms: f64,
    p50_ms: f64,
    acceptance_rate: f64,
    weights_mib: f64,
    speedup: f64,
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let draft_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let target_path = PathBuf::from("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");

    if !draft_path.exists() || !target_path.exists() {
        eprintln!("Models missing. Please ensure both 0.5B and 1.5B exist.");
        return Ok(());
    }

    let prompt = "Explain in one sentence why memory bandwidth limits LLM decoding.";
    let params = GenerationParams {
        max_tokens: 36,
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
    };

    println!("=========================================================================================");
    println!("  EXPERIMENT: DUAL-MODEL SPECULATIVE DECODING ON RTX 3050 TI (4 GB VRAM)");
    println!("  Draft: Qwen 0.5B (463 MiB) | Target: Qwen 1.5B (1,060 MiB) | Combined: 1.52 GiB Weights");
    println!("=========================================================================================\n");

    // 1. Baseline: Pure 1.5B Autoregressive
    print!(">>> 1. Benchmarking Pure Qwen 1.5B Baseline (Autoregressive)... ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let base_cfg = EngineConfig::new(&target_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_kv_precision(KvPrecision::Q8_0);
    let mut base_engine = RawFfiEngine::new(base_cfg)?;
    let base_out = base_engine.generate(prompt, &params, None)?;
    let base_speed = base_out.report.generation_tok_per_sec;
    println!(
        "Done -> Speed: {:.2} tok/s | TTFT: {:.2} ms | Weights: {:.2} MiB",
        base_speed,
        base_out.report.time_to_first_token.as_secs_f64() * 1000.0,
        base_out.memory.weights_mib()
    );

    let mut results: Vec<SpeculativeResult> = Vec::new();
    results.push(SpeculativeResult {
        label: "Standard 1.5B (No Draft)",
        gen_tok_per_sec: base_speed,
        ttft_ms: base_out.report.time_to_first_token.as_secs_f64() * 1000.0,
        p50_ms: base_out.report.latency_p50_ms,
        acceptance_rate: 0.0,
        weights_mib: base_out.memory.weights_mib(),
        speedup: 1.0,
    });

    // 2. Speculative with K = 2, 3, 4
    for k in [2, 3, 4] {
        print!(">>> 2. Benchmarking Speculative Decoding (K = {} Draft Tokens)... ", k);
        std::io::Write::flush(&mut std::io::stdout())?;
        let spec_cfg = EngineConfig::new(&target_path)
            .with_device(Some(1))
            .with_gpu_layers(999)
            .with_ctx_size(1024);
        let mut spec_engine = SpeculativeEngine::new(&draft_path, &target_path, spec_cfg, k)?;
        let spec_out = spec_engine.generate(prompt, &params, None)?;
        let speed = spec_out.report.generation_tok_per_sec;
        let speedup = speed / base_speed;
        let stats = &spec_engine.last_stats;

        println!(
            "Done -> Speed: {:.2} tok/s ({:.2}x) | Accept: {:.1}% | TTFT: {:.2} ms",
            speed, speedup, stats.acceptance_rate, spec_out.report.time_to_first_token.as_secs_f64() * 1000.0
        );

        let label = match k {
            2 => "Speculative (K = 2)",
            3 => "Speculative (K = 3)",
            4 => "Speculative (K = 4)",
            _ => "Speculative",
        };

        results.push(SpeculativeResult {
            label,
            gen_tok_per_sec: speed,
            ttft_ms: spec_out.report.time_to_first_token.as_secs_f64() * 1000.0,
            p50_ms: spec_out.report.latency_p50_ms,
            acceptance_rate: stats.acceptance_rate,
            weights_mib: spec_out.memory.weights_mib(),
            speedup,
        });
    }

    println!("\n=======================================================================================================================");
    println!("                               DUAL-MODEL SPECULATIVE DECODING COMPARISON TABLE");
    println!("=======================================================================================================================");
    println!(
        "| {:<24} | {:>12} | {:>10} | {:>11} | {:>11} | {:>11} | {:>11} |",
        "Decoding Strategy", "Gen Speed", "Speedup", "Accept Rate", "TTFT", "P50 Lat", "VRAM Weights"
    );
    println!(
        "| {:<24} | {:>12} | {:>10} | {:>11} | {:>11} | {:>11} | {:>11} |",
        "", "(tok/s)", "(vs Base)", "(%)", "(ms)", "(ms)", "(MiB)"
    );
    println!("|--------------------------|--------------|------------|-------------|-------------|-------------|-------------|");

    for r in &results {
        println!(
            "| {:<24} | {:>12.2} | {:>9.2}x | {:>10.1}% | {:>11.2} | {:>11.2} | {:>11.2} |",
            r.label,
            r.gen_tok_per_sec,
            r.speedup,
            r.acceptance_rate,
            r.ttft_ms,
            r.p50_ms,
            r.weights_mib
        );
    }
    println!("=======================================================================================================================\n");

    Ok(())
}
