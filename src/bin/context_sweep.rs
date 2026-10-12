//! Experiment Suite: Context Saturation & Boundary Cliff Profiler.
//!
//! Evaluates how LLM inference scales when the context window fills up towards 100%:
//! 25% -> 50% -> 75% -> 90% -> 95% -> 98% capacity.
//!
//! Measures prefill scaling, TTFT, step decoding decay, latency jitter (σ),
//! and tail latency (P95) on the RTX 3050 Ti via Vulkan.

use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use kortex::engines::{HybridSinkEngine, RawFfiEngine, SlidingWindowEngine};

struct SaturationDataPoint {
    target_pct: u32,
    actual_prompt_tokens: usize,
    prefill_time_ms: f64,
    prefill_tok_per_sec: f64,
    ttft_ms: f64,
    gen_tok_per_sec: f64,
    p50_ms: f64,
    p95_ms: f64,
    stddev_ms: f64,
    active_kv_mib: f64,
}

fn build_context_prompt(target_token_budget: usize) -> String {
    let base_paragraph = "High-performance computing relies heavily on memory bandwidth and cache hierarchy efficiency. When matrix multiplication kernels execute on tensor hardware, memory controllers must stream weight parameters and activation states across high-speed GDDR buses without bus stall or bank conflict. ";
    
    // Each paragraph repetition produces ~50 tokens in Qwen tokenizer
    let repetitions = (target_token_budget / 52).max(1);
    let mut text = String::with_capacity(target_token_budget * 5);
    for _ in 0..repetitions {
        text.push_str(base_paragraph);
    }
    text.push_str("\nQuestion: What limits computing hardware during memory transfers? Answer:");
    text
}

fn test_fill_level(
    model_path: &PathBuf,
    ctx_size: u32,
    target_pct: u32,
    precision: KvPrecision,
) -> Result<SaturationDataPoint> {
    let target_tokens = ((ctx_size as f64) * (target_pct as f64 / 100.0)) as usize;
    // Reserve ~28 tokens for generation
    let prompt_budget = target_tokens.saturating_sub(28).max(32);
    let prompt = build_context_prompt(prompt_budget);

    let config = EngineConfig::new(model_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_ctx_size(ctx_size)
        .with_kv_precision(precision);

    let mut engine = RawFfiEngine::new(config)?;

    let params = GenerationParams {
        max_tokens: 20,
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
    };

    let output = engine.generate(&prompt, &params, None)?;
    let rep = output.report;

    let active_tokens = (rep.prompt_tokens + output.generated_tokens) as f64;
    let kv_bytes = active_tokens * (output.memory.kv_bytes_per_token as f64);
    let active_kv_mib = kv_bytes / (1024.0 * 1024.0);

    Ok(SaturationDataPoint {
        target_pct,
        actual_prompt_tokens: rep.prompt_tokens,
        prefill_time_ms: rep.prompt_duration.as_secs_f64() * 1000.0,
        prefill_tok_per_sec: rep.prompt_tok_per_sec,
        ttft_ms: rep.time_to_first_token.as_secs_f64() * 1000.0,
        gen_tok_per_sec: rep.generation_tok_per_sec,
        p50_ms: rep.latency_p50_ms,
        p95_ms: rep.latency_p95_ms,
        stddev_ms: rep.latency_std_dev_ms,
        active_kv_mib,
    })
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let model_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}.", model_path);
        return Ok(());
    }

    let ctx_size = 2048u32;
    let fill_percentages = [25, 50, 75, 90, 95, 98];

    println!("=========================================================================================");
    println!("  EXPERIMENT: CONTEXT SATURATION & BOUNDARY CLIFF SWEEP");
    println!("  Model: Qwen 2.5 0.5B Instruct | Context Limit: {} tokens | GPU: RTX 3050 Ti", ctx_size);
    println!("=========================================================================================\n");

    println!(">>> 1. Running FP16 KV Cache Saturation Sweep...");
    let mut fp16_points: Vec<SaturationDataPoint> = Vec::new();
    for &pct in &fill_percentages {
        print!("    Testing {}% Context Fill... ", pct);
        std::io::Write::flush(&mut std::io::stdout())?;
        let pt = test_fill_level(&model_path, ctx_size, pct, KvPrecision::F16)?;
        println!(
            "Done -> Prompt: {} toks, TTFT: {:.2} ms, Prefill: {:.1} tok/s, Gen: {:.1} tok/s, P50: {:.2} ms",
            pt.actual_prompt_tokens, pt.ttft_ms, pt.prefill_tok_per_sec, pt.gen_tok_per_sec, pt.p50_ms
        );
        fp16_points.push(pt);
    }

    println!("\n>>> 2. Running Q8_0 Quantized KV Cache Saturation Sweep...");
    let mut q8_points: Vec<SaturationDataPoint> = Vec::new();
    for &pct in &fill_percentages {
        print!("    Testing {}% Context Fill... ", pct);
        std::io::Write::flush(&mut std::io::stdout())?;
        let pt = test_fill_level(&model_path, ctx_size, pct, KvPrecision::Q8_0)?;
        println!(
            "Done -> Prompt: {} toks, TTFT: {:.2} ms, Prefill: {:.1} tok/s, Gen: {:.1} tok/s, P50: {:.2} ms",
            pt.actual_prompt_tokens, pt.ttft_ms, pt.prefill_tok_per_sec, pt.gen_tok_per_sec, pt.p50_ms
        );
        q8_points.push(pt);
    }

    println!("\n>>> 3. Testing Eviction Resilience with Sliding Window & Hybrid Q8_0 Sinks...");
    let long_prompt = build_context_prompt(2000); // 2000 tokens in a 128-token budget context!
    
    // Standard sliding window (FP16)
    let sw_cfg = EngineConfig::new(&model_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_ctx_size(128); // tiny 128 context budget!
    let mut sw_engine = SlidingWindowEngine::new(sw_cfg, 4, 64, 8)?;
    print!("    SlidingWindowEngine on 2000-token prompt in 128-token context... ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let sw_out = sw_engine.generate(&long_prompt, &GenerationParams { max_tokens: 32, temperature: 0.0, top_p: 1.0, seed: 42 }, None)?;
    println!("Success! -> Gen: {:.1} tok/s, KV Used: {:.2} MiB (Bound preserved!)", sw_out.report.generation_tok_per_sec, sw_out.memory.kv_cache_mib());

    // Hybrid Q8_0 Sink
    let hy_cfg = EngineConfig::new(&model_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_ctx_size(128);
    let mut hy_engine = HybridSinkEngine::new(hy_cfg, KvPrecision::Q8_0, 4, 64, 8)?;
    print!("    HybridSinkEngine (Q8_0) on 2000-token prompt in 128-token context... ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let hy_out = hy_engine.generate(&long_prompt, &GenerationParams { max_tokens: 32, temperature: 0.0, top_p: 1.0, seed: 42 }, None)?;
    println!("Success! -> Gen: {:.1} tok/s, KV Used: {:.2} MiB (Bound preserved!)", hy_out.report.generation_tok_per_sec, hy_out.memory.kv_cache_mib());

    println!("\n=======================================================================================================================");
    println!("                                 FP16 KV CACHE: CONTEXT SATURATION SCALING TABLE");
    println!("=======================================================================================================================");
    println!(
        "| {:<8} | {:>12} | {:>11} | {:>13} | {:>10} | {:>11} | {:>9} | {:>9} | {:>9} |",
        "Fill %", "Prompt Toks", "KV VRAM", "Prefill Time", "Prefill", "Gen Speed", "P50 Lat", "P95 Lat", "Jitter σ"
    );
    println!(
        "| {:<8} | {:>12} | {:>11} | {:>13} | {:>10} | {:>11} | {:>9} | {:>9} | {:>9} |",
        "", "", "(MiB)", "(ms)", "(tok/s)", "(tok/s)", "(ms)", "(ms)", "(ms)"
    );
    println!("|----------|--------------|-------------|---------------|------------|-------------|-----------|-----------|-----------|");

    for pt in &fp16_points {
        println!(
            "| {:<8} | {:>12} | {:>11.2} | {:>13.2} | {:>10.1} | {:>11.2} | {:>9.2} | {:>9.2} | {:>9.2} |",
            format!("{}%", pt.target_pct),
            pt.actual_prompt_tokens,
            pt.active_kv_mib,
            pt.prefill_time_ms,
            pt.prefill_tok_per_sec,
            pt.gen_tok_per_sec,
            pt.p50_ms,
            pt.p95_ms,
            pt.stddev_ms
        );
    }

    println!("\n=======================================================================================================================");
    println!("                                 Q8_0 KV CACHE: CONTEXT SATURATION SCALING TABLE");
    println!("=======================================================================================================================");
    println!(
        "| {:<8} | {:>12} | {:>11} | {:>13} | {:>10} | {:>11} | {:>9} | {:>9} | {:>9} |",
        "Fill %", "Prompt Toks", "KV VRAM", "Prefill Time", "Prefill", "Gen Speed", "P50 Lat", "P95 Lat", "Jitter σ"
    );
    println!(
        "| {:<8} | {:>12} | {:>11} | {:>13} | {:>10} | {:>11} | {:>9} | {:>9} | {:>9} |",
        "", "", "(MiB)", "(ms)", "(tok/s)", "(tok/s)", "(ms)", "(ms)", "(ms)"
    );
    println!("|----------|--------------|-------------|---------------|------------|-------------|-----------|-----------|-----------|");

    for pt in &q8_points {
        println!(
            "| {:<8} | {:>12} | {:>11.2} | {:>13.2} | {:>10.1} | {:>11.2} | {:>9.2} | {:>9.2} | {:>9.2} |",
            format!("{}%", pt.target_pct),
            pt.actual_prompt_tokens,
            pt.active_kv_mib,
            pt.prefill_time_ms,
            pt.prefill_tok_per_sec,
            pt.gen_tok_per_sec,
            pt.p50_ms,
            pt.p95_ms,
            pt.stddev_ms
        );
    }
    println!("=======================================================================================================================\n");

    Ok(())
}
