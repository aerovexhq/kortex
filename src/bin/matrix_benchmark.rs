//! Multi-Variate Pareto Matrix Benchmark Suite.
//!
//! Evaluates all architectures, engines, quantization precisions, and sliding-window
//! combinations across Qwen 0.5B and Qwen 1.5B models on the RTX 3050 Ti.
//!
//! Generates an end-to-end Pareto comparison table and exports telemetry to `analysis/matrix_results.json`.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{
    EngineConfig, GenerationParams, KvPrecision, ModelEngine,
};
use kortex::engines::{
    CpuEngine, HybridSinkEngine, QuantizedKvEngine, RawFfiEngine, SafeLlamaEngine,
    SlidingWindowEngine,
};

#[derive(serde::Serialize)]
struct MatrixEntry {
    model: String,
    method: String,
    hardware: String,
    kv_precision: String,
    weights_mib: f64,
    kv_kib_per_token: f64,
    kv_cache_mib: f64,
    total_ram_mib: f64,
    ttft_ms: f64,
    prefill_tok_per_sec: f64,
    gen_tok_per_sec: f64,
    p50_ms: f64,
    p95_ms: f64,
    jitter_sigma_ms: f64,
    efficiency_score: f64,
}

fn profile_candidate(
    model_name: &str,
    method_label: &str,
    mut engine: Box<dyn ModelEngine>,
    prompt: &str,
    params: &GenerationParams,
) -> Result<MatrixEntry> {
    print!("    Profiling {:<38} ... ", method_label);
    std::io::stdout().flush()?;

    let output = engine.generate(prompt, params, None)?;
    let rep = &output.report;
    let mem = &output.memory;

    let total_gib = mem.total_memory_mib() / 1024.0;
    let p50_ms = rep.latency_p50_ms.max(0.1);
    let efficiency = rep.generation_tok_per_sec / (total_gib * p50_ms);

    println!(
        "Done -> {:>5.1} tok/s | P50: {:>4.2} ms | RAM: {:>6.1} MiB | Score: {:>6.1}",
        rep.generation_tok_per_sec, p50_ms, mem.total_memory_mib(), efficiency
    );

    Ok(MatrixEntry {
        model: model_name.to_string(),
        method: method_label.to_string(),
        hardware: engine.device_name(),
        kv_precision: mem.kv_precision.label().split(' ').next().unwrap_or("").to_string(),
        weights_mib: mem.weights_mib(),
        kv_kib_per_token: mem.kv_kib_per_token(),
        kv_cache_mib: mem.kv_cache_mib(),
        total_ram_mib: mem.total_memory_mib(),
        ttft_ms: rep.time_to_first_token.as_secs_f64() * 1000.0,
        prefill_tok_per_sec: rep.prompt_tok_per_sec,
        gen_tok_per_sec: rep.generation_tok_per_sec,
        p50_ms,
        p95_ms: rep.latency_p95_ms,
        jitter_sigma_ms: rep.latency_std_dev_ms,
        efficiency_score: efficiency,
    })
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let model_0_5b = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let model_1_5b = PathBuf::from("models/qwen2.5-1.5b-instruct-q4_k_m.gguf");

    if !model_0_5b.exists() {
        eprintln!("Model not found at {:?}.", model_0_5b);
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
    println!("  MULTI-VARIATE PARETO MATRIX BENCHMARK SUITE");
    println!("  Hardware: NVIDIA GeForce RTX 3050 Ti (Vulkan) & AMD Ryzen 5 6600H (12T)");
    println!("=========================================================================================\n");

    let mut entries: Vec<MatrixEntry> = Vec::new();

    // -------------------------------------------------------------------------
    // 1. QWEN 2.5 0.5B MATRIX
    // -------------------------------------------------------------------------
    println!(">>> [Model Suite 1] Qwen 2.5 0.5B Instruct (0.49B Params):");

    // 1. Safe Rust (llama-cpp-2)
    {
        let cfg = EngineConfig::new(&model_0_5b).with_device(Some(1)).with_gpu_layers(999);
        let eng = SafeLlamaEngine::new(cfg)?;
        entries.push(profile_candidate("Qwen 0.5B", "Safe Rust Wrapper (FP16)", Box::new(eng), prompt, &params)?);
    }

    // 2. Raw C-ABI FFI (FP16)
    {
        let cfg = EngineConfig::new(&model_0_5b).with_device(Some(1)).with_gpu_layers(999);
        let eng = RawFfiEngine::new(cfg)?;
        entries.push(profile_candidate("Qwen 0.5B", "Raw C-ABI FFI (FP16)", Box::new(eng), prompt, &params)?);
    }

    // 3. Raw C-ABI Q8_0 KV
    {
        let cfg = EngineConfig::new(&model_0_5b).with_device(Some(1)).with_gpu_layers(999);
        let eng = QuantizedKvEngine::new(cfg, KvPrecision::Q8_0)?;
        entries.push(profile_candidate("Qwen 0.5B", "Quantized KV (Q8_0)", Box::new(eng), prompt, &params)?);
    }

    // 4. Sliding Window (FP16, S=4, Win=64, Chunk=16)
    {
        let cfg = EngineConfig::new(&model_0_5b).with_device(Some(1)).with_gpu_layers(999);
        let eng = SlidingWindowEngine::new(cfg, 4, 64, 16)?;
        entries.push(profile_candidate("Qwen 0.5B", "Sliding Window (FP16, k=16)", Box::new(eng), prompt, &params)?);
    }

    // 5. Hybrid Sink (Q8_0 + S=4 + Win=64 + Chunk=16) [The Sweet Spot Candidate]
    {
        let cfg = EngineConfig::new(&model_0_5b).with_device(Some(1)).with_gpu_layers(999);
        let eng = HybridSinkEngine::new(cfg, KvPrecision::Q8_0, 4, 64, 16)?;
        entries.push(profile_candidate("Qwen 0.5B", "Hybrid Sink (Q8_0, k=16)", Box::new(eng), prompt, &params)?);
    }

    // 6. Pure CPU Baseline
    {
        let cfg = EngineConfig::new(&model_0_5b);
        let eng = CpuEngine::new(cfg)?;
        entries.push(profile_candidate("Qwen 0.5B", "Pure CPU (12 Threads)", Box::new(eng), prompt, &params)?);
    }

    // -------------------------------------------------------------------------
    // 2. QWEN 2.5 1.5B MATRIX
    // -------------------------------------------------------------------------
    if model_1_5b.exists() {
        println!("\n>>> [Model Suite 2] Qwen 2.5 1.5B Instruct (1.54B Params):");

        // 1. Raw C-ABI 1.5B (FP16)
        {
            let cfg = EngineConfig::new(&model_1_5b).with_device(Some(1)).with_gpu_layers(999);
            let eng = RawFfiEngine::new(cfg)?;
            entries.push(profile_candidate("Qwen 1.5B", "Raw C-ABI FFI (FP16)", Box::new(eng), prompt, &params)?);
        }

        // 2. Raw C-ABI 1.5B (Q8_0 KV)
        {
            let cfg = EngineConfig::new(&model_1_5b).with_device(Some(1)).with_gpu_layers(999);
            let eng = QuantizedKvEngine::new(cfg, KvPrecision::Q8_0)?;
            entries.push(profile_candidate("Qwen 1.5B", "Quantized KV (Q8_0)", Box::new(eng), prompt, &params)?);
        }

        // 3. Hybrid Sink 1.5B (Q8_0 + S=4 + Win=64 + Chunk=16)
        {
            let cfg = EngineConfig::new(&model_1_5b).with_device(Some(1)).with_gpu_layers(999);
            let eng = HybridSinkEngine::new(cfg, KvPrecision::Q8_0, 4, 64, 16)?;
            entries.push(profile_candidate("Qwen 1.5B", "Hybrid Sink (Q8_0, k=16)", Box::new(eng), prompt, &params)?);
        }
    }

    // Sort candidates by Efficiency Score descending
    entries.sort_by(|a, b| b.efficiency_score.partial_cmp(&a.efficiency_score).unwrap());

    // -------------------------------------------------------------------------
    // PRINT GRAND PARETO MATRIX TABLE
    // -------------------------------------------------------------------------
    println!("\n\n=========================================================================================================================================================");
    println!("                                                    MULTI-VARIATE PARETO MATRIX RANKING TABLE");
    println!("=========================================================================================================================================================");
    println!(
        "| {:<4} | {:<10} | {:<28} | {:<7} | {:>9} | {:>8} | {:>8} | {:>8} | {:>9} | {:>7} | {:>7} | {:>10} |",
        "Rank", "Model", "Method / Engine", "KV Prec", "RAM (MiB)", "KV (KiB)", "TTFT(ms)", "Gen(t/s)", "P50 (ms)", "P95(ms)", "σ (ms)", "Efficiency"
    );
    println!("|------|------------|------------------------------|---------|-----------|----------|----------|----------|-----------|---------|---------|------------|");

    for (idx, e) in entries.iter().enumerate() {
        let medal = match idx {
            0 => "🥇 1",
            1 => "🥈 2",
            2 => "🥉 3",
            _ => "",
        };
        let rank_str = if medal.is_empty() { format!("{:>4}", idx + 1) } else { medal.to_string() };

        println!(
            "| {:<4} | {:<10} | {:<28} | {:<7} | {:>9.1} | {:>8.1} | {:>8.1} | {:>8.1} | {:>9.2} | {:>7.2} | {:>7.2} | {:>10.1} |",
            rank_str,
            e.model,
            e.method,
            e.kv_precision,
            e.total_ram_mib,
            e.kv_kib_per_token,
            e.ttft_ms,
            e.gen_tok_per_sec,
            e.p50_ms,
            e.p95_ms,
            e.jitter_sigma_ms,
            e.efficiency_score
        );
    }
    println!("=========================================================================================================================================================\n");

    // Export telemetry to JSON file
    let json_path = PathBuf::from("analysis/matrix_results.json");
    if let Ok(file) = File::create(&json_path) {
        let _ = serde_json::to_writer_pretty(file, &entries);
        println!(">>> Telemetry exported successfully to {:?}", json_path);
    }

    Ok(())
}
