//! Experiment Suite: Eviction Granularity & Latency Jitter Sweep.
//!
//! Evaluates the sweet spot for eviction chunk size `k` in Sliding-Window Attention:
//! k = 1 (continuous single-token shift) vs k = 4 vs k = 8 vs k = 16 vs k = 32 (block eviction).
//!
//! Measures inter-token latency spikes, jitter standard deviation (σ), P95, and P99
//! tail latency to find the optimal balance between FFI call frequency and frame smoothness.

use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use kortex::engines::SlidingWindowEngine;

struct ChunkResult {
    chunk_k: i32,
    gen_tok_per_sec: f64,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stddev_ms: f64,
}

fn test_chunk_size(model_path: &PathBuf, k: i32) -> Result<ChunkResult> {
    let config = EngineConfig::new(model_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_ctx_size(128)
        .with_kv_precision(KvPrecision::F16);

    // Budget: 4 sink tokens + 64 window size = 68 tokens bound
    let mut engine = SlidingWindowEngine::new(config, 4, 64, k)?;

    let prompt = "Explain the fundamental trade-offs between continuous token eviction and amortized block eviction in streaming language models.";
    let params = GenerationParams {
        max_tokens: 80, // Generates 80 tokens, forcing multiple eviction cycles
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
    };

    let output = engine.generate(prompt, &params, None)?;
    let rep = output.report;

    Ok(ChunkResult {
        chunk_k: k,
        gen_tok_per_sec: rep.generation_tok_per_sec,
        mean_ms: rep.latency_mean_ms,
        p50_ms: rep.latency_p50_ms,
        p95_ms: rep.latency_p95_ms,
        min_ms: rep.latency_min_ms,
        max_ms: rep.latency_max_ms,
        stddev_ms: rep.latency_std_dev_ms,
    })
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let model_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}.", model_path);
        return Ok(());
    }

    let chunk_sizes = [1, 4, 8, 16, 32];

    println!("=========================================================================================");
    println!("  EXPERIMENT: EVICTION CHUNK GRANULARITY & LATENCY JITTER SWEEP");
    println!("  Model: Qwen 2.5 0.5B Instruct | 80 Generated Tokens | Sink=4, Win=64 (Budget=68)");
    println!("=========================================================================================\n");

    let mut results: Vec<ChunkResult> = Vec::new();

    for &k in &chunk_sizes {
        print!(">>> Testing Eviction Chunk k = {:>2}... ", k);
        std::io::Write::flush(&mut std::io::stdout())?;
        let res = test_chunk_size(&model_path, k)?;
        println!(
            "Done -> Speed: {:>5.1} tok/s | P50: {:>4.2} ms | P95: {:>4.2} ms | Max Spike: {:>5.2} ms | σ: {:>4.2} ms",
            res.gen_tok_per_sec, res.p50_ms, res.p95_ms, res.max_ms, res.stddev_ms
        );
        results.push(res);
    }

    println!("\n=======================================================================================================================");
    println!("                              EVICTION CHUNK GRANULARITY & JITTER COMPARISON TABLE");
    println!("=======================================================================================================================");
    println!(
        "| {:<10} | {:>12} | {:>11} | {:>10} | {:>10} | {:>10} | {:>11} | {:>11} |",
        "Chunk (k)", "Gen Speed", "Mean Lat", "P50 Lat", "P95 Lat", "Min Step", "Max Spike", "Jitter (σ)"
    );
    println!(
        "| {:<10} | {:>12} | {:>11} | {:>10} | {:>10} | {:>10} | {:>11} | {:>11} |",
        "", "(tok/s)", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)", "(ms)"
    );
    println!("|------------|--------------|-------------|------------|------------|------------|-------------|-------------|");

    for r in &results {
        println!(
            "| k = {:<6} | {:>12.2} | {:>11.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>11.2} | {:>11.2} |",
            r.chunk_k,
            r.gen_tok_per_sec,
            r.mean_ms,
            r.p50_ms,
            r.p95_ms,
            r.min_ms,
            r.max_ms,
            r.stddev_ms
        );
    }
    println!("=======================================================================================================================\n");

    Ok(())
}
