use std::io::Write;
use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{
    EngineConfig, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile,
};
use kortex::engines::{
    CpuEngine, QuantizedKvEngine, RawFfiEngine, SafeLlamaEngine, SlidingWindowEngine,
};

#[allow(dead_code)]
struct BenchmarkRecord {
    method_name: &'static str,
    device_name: String,
    memory: ModelMemoryProfile,
    ttft_ms: f64,
    prompt_tok_per_sec: f64,
    gen_tok_per_sec: f64,
    p50_ms: f64,
    stddev_ms: f64,
    sample_text_snippet: String,
}

fn run_engine_benchmark(
    name: &'static str,
    mut engine: Box<dyn ModelEngine>,
    prompt: &str,
    params: &GenerationParams,
) -> Result<BenchmarkRecord> {
    println!("\n================================================================================");
    println!(">>> Running [{}]: {}", name, engine.name());
    println!("    Target Hardware: {}", engine.device_name());
    let mem = engine.memory_profile();
    println!(
        "    Weights: {:.2} MiB | KV Precision: {} | KV Cache: {:.2} KiB/token | Ctx Total KV: {:.2} MiB",
        mem.weights_mib(),
        mem.kv_precision.label(),
        mem.kv_kib_per_token(),
        mem.kv_cache_mib()
    );
    println!("--------------------------------------------------------------------------------");
    print!("    Streaming output: ");
    std::io::stdout().flush()?;

    let output = engine.generate(
        prompt,
        params,
        Some(&mut |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            true
        }),
    )?;
    println!("\n");

    let rep = &output.report;
    println!(
        "    Result -> TTFT: {:.2} ms | Prompt: {:.2} tok/s | Gen: {:.2} tok/s | P50: {:.2} ms (σ={:.2} ms)",
        rep.time_to_first_token.as_secs_f64() * 1000.0,
        rep.prompt_tok_per_sec,
        rep.generation_tok_per_sec,
        rep.latency_p50_ms,
        rep.latency_std_dev_ms,
    );

    let snippet = output
        .text
        .lines()
        .next()
        .unwrap_or(&output.text)
        .chars()
        .take(60)
        .collect::<String>();

    Ok(BenchmarkRecord {
        method_name: name,
        device_name: engine.device_name(),
        memory: output.memory,
        ttft_ms: rep.time_to_first_token.as_secs_f64() * 1000.0,
        prompt_tok_per_sec: rep.prompt_tok_per_sec,
        gen_tok_per_sec: rep.generation_tok_per_sec,
        p50_ms: rep.latency_p50_ms,
        stddev_ms: rep.latency_std_dev_ms,
        sample_text_snippet: snippet,
    })
}

fn main() -> Result<()> {
    // Suppress verbose internal llama.cpp C logs
    kortex::engine::silence_llama_logs();

    let model_0_5b = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let model_1_5b = PathBuf::from("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");

    if !model_0_5b.exists() {
        eprintln!("Model not found at {:?}. Please download first.", model_0_5b);
        return Ok(());
    }

    let test_prompt = "Explain in one sentence why memory bandwidth limits LLM decoding.";
    let params = GenerationParams {
        max_tokens: 48,
        temperature: 0.0, // greedy argmax for zero-variance deterministic benchmarking
        top_p: 1.0,
        seed: 42,
    };

    println!("################################################################################");
    println!("#  EXTREMAL MULTI-PATH LLM INFERENCE RESEARCH & MEMORY BENCHMARK               #");
    println!("#  Host: AMD Ryzen 5 6600H (6C/12T, DDR5) | GPU: NVIDIA GeForce RTX 3050 Ti     #");
    println!("################################################################################");

    let mut records: Vec<BenchmarkRecord> = Vec::new();

    // -------------------------------------------------------------------------
    // PATH 1: Safe High-Level Safe Rust Engine (FP16 KV)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b)
            .with_device(Some(1))
            .with_gpu_layers(999);
        let engine = SafeLlamaEngine::new(config)?;
        records.push(run_engine_benchmark(
            "1. Safe Rust (FP16 KV)",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // -------------------------------------------------------------------------
    // PATH 2: Low-Level Raw C-ABI FFI Engine (FP16 KV)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b)
            .with_device(Some(1))
            .with_gpu_layers(999);
        let engine = RawFfiEngine::new(config)?;
        records.push(run_engine_benchmark(
            "2. Raw C-ABI FFI (FP16 KV)",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // -------------------------------------------------------------------------
    // PATH 3: Quantized KV Cache - Q8_0 (8-bit KV, 50% Memory Reduction)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b)
            .with_device(Some(1))
            .with_gpu_layers(999);
        let engine = QuantizedKvEngine::new(config, KvPrecision::Q8_0)?;
        records.push(run_engine_benchmark(
            "3. Quantized KV (Q8_0 KV)",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // -------------------------------------------------------------------------
    // PATH 4: Quantized KV Cache - Q4_0 (4-bit KV, 75% Memory Reduction)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b)
            .with_device(Some(1))
            .with_gpu_layers(999);
        let engine = QuantizedKvEngine::new(config, KvPrecision::Q4_0)?;
        records.push(run_engine_benchmark(
            "4. Quantized KV (Q4_0 KV)",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // -------------------------------------------------------------------------
    // PATH 5: Algorithmic Sliding-Window Attention & StreamingLLM (O(1) Bounded)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b)
            .with_device(Some(1))
            .with_gpu_layers(999);
        let engine = SlidingWindowEngine::new(config, 4, 64, 8)?;
        records.push(run_engine_benchmark(
            "5. Sliding-Window O(1) Cache",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // -------------------------------------------------------------------------
    // PATH 6: Pure CPU SIMD Multithreaded Execution (Host DDR5 RAM)
    // -------------------------------------------------------------------------
    {
        let config = EngineConfig::new(&model_0_5b);
        let engine = CpuEngine::new(config)?;
        records.push(run_engine_benchmark(
            "6. Pure CPU (Host DDR5)",
            Box::new(engine),
            test_prompt,
            &params,
        )?);
    }

    // =========================================================================
    // GRAND COMPARISON TABLE
    // =========================================================================
    println!("\n\n========================================================================================================================");
    println!("                                   COMPREHENSIVE MULTI-PATH BENCHMARK COMPARISON TABLE");
    println!("========================================================================================================================");
    println!(
        "| {:<28} | {:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9} | {:>9} |",
        "Method / Engine",
        "KV Precision",
        "Weights",
        "KV Cache",
        "Total RAM",
        "TTFT",
        "Prompt",
        "Gen Speed",
        "P50 Lat"
    );
    println!(
        "| {:<28} | {:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>9} | {:>9} | {:>9} |",
        "",
        "",
        "(MiB)",
        "(KiB/tok)",
        "(MiB)",
        "(ms)",
        "(tok/s)",
        "(tok/s)",
        "(ms)"
    );
    println!("|------------------------------|--------------|------------|------------|------------|------------|-----------|-----------|-----------|");

    for r in &records {
        println!(
            "| {:<28} | {:<12} | {:>10.2} | {:>10.2} | {:>10.2} | {:>10.2} | {:>9.2} | {:>9.2} | {:>9.2} |",
            r.method_name,
            r.memory.kv_precision.label().split(' ').next().unwrap_or(""),
            r.memory.weights_mib(),
            r.memory.kv_kib_per_token(),
            r.memory.total_memory_mib(),
            r.ttft_ms,
            r.prompt_tok_per_sec,
            r.gen_tok_per_sec,
            r.p50_ms
        );
    }
    println!("========================================================================================================================\n");

    // Also check Qwen 1.5B scaling profile
    if model_1_5b.exists() {
        println!(">>> Architectural Scaling Comparison with Qwen 2.5 1.5B:");
        let cfg_fp16 = EngineConfig::new(&model_1_5b);
        let prof_fp16 = RawFfiEngine::new(cfg_fp16)?.memory_profile();

        let cfg_q8 = EngineConfig::new(&model_1_5b).with_kv_precision(KvPrecision::Q8_0);
        let prof_q8 = RawFfiEngine::new(cfg_q8)?.memory_profile();

        let cfg_q4 = EngineConfig::new(&model_1_5b).with_kv_precision(KvPrecision::Q4_0);
        let prof_q4 = RawFfiEngine::new(cfg_q4)?.memory_profile();

        println!(
            "    Qwen 1.5B (FP16 KV): Weights = {:.2} MiB | KV = {:.2} KiB/token | Ctx 2048 Total = {:.2} MiB",
            prof_fp16.weights_mib(),
            prof_fp16.kv_kib_per_token(),
            prof_fp16.total_memory_mib()
        );
        println!(
            "    Qwen 1.5B (Q8_0 KV): Weights = {:.2} MiB | KV = {:.2} KiB/token | Ctx 2048 Total = {:.2} MiB",
            prof_q8.weights_mib(),
            prof_q8.kv_kib_per_token(),
            prof_q8.total_memory_mib()
        );
        println!(
            "    Qwen 1.5B (Q4_0 KV): Weights = {:.2} MiB | KV = {:.2} KiB/token | Ctx 2048 Total = {:.2} MiB",
            prof_q4.weights_mib(),
            prof_q4.kv_kib_per_token(),
            prof_q4.total_memory_mib()
        );
        println!();
    }

    Ok(())
}
