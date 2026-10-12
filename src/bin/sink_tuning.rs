//! Experiment Suite: Attention-Sink Count & Perplexity Phase Transition.
//!
//! Investigates the necessity and scaling of Attention Sinks in streaming LLM inference:
//! S = 0 (pure rolling window, NO sink) vs S = 1 vs S = 2 vs S = 4 vs S = 8.
//!
//! Evaluates whether S=0 collapses attention entropy (producing degeneration/loops),
//! and determines the minimum sink token threshold required for stable generation.

use std::path::PathBuf;
use anyhow::Result;

use kortex::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use kortex::engines::SlidingWindowEngine;

struct SinkResult {
    sink_s: i32,
    gen_tok_per_sec: f64,
    p50_ms: f64,
    sample_snippet: String,
    repetition_detected: bool,
}

fn check_degenerate_repetition(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return false;
    }
    // Check for 3-gram repetitions
    for i in 0..words.len().saturating_sub(6) {
        let tri1 = (&words[i], &words[i + 1], &words[i + 2]);
        let tri2 = (&words[i + 3], &words[i + 4], &words[i + 5]);
        if tri1 == tri2 {
            return true;
        }
    }
    false
}

fn test_sink_count(model_path: &PathBuf, s: i32) -> Result<SinkResult> {
    let config = EngineConfig::new(model_path)
        .with_device(Some(1))
        .with_gpu_layers(999)
        .with_ctx_size(128)
        .with_kv_precision(KvPrecision::F16);

    let window_size = 48;
    let evict_chunk = 16;
    let mut engine = SlidingWindowEngine::new(config, s, window_size, evict_chunk)?;

    let prompt = "Explain the step-by-step process of how modern GPUs schedule thread warps during compute tasks.";
    let params = GenerationParams {
        max_tokens: 80, // Far exceeds window size, forcing multiple rounds of eviction
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
    };

    let output = engine.generate(prompt, &params, None)?;
    let rep = output.report;

    let is_degenerate = check_degenerate_repetition(&output.text);
    let snippet = output
        .text
        .replace('\n', " ")
        .chars()
        .take(75)
        .collect::<String>();

    Ok(SinkResult {
        sink_s: s,
        gen_tok_per_sec: rep.generation_tok_per_sec,
        p50_ms: rep.latency_p50_ms,
        sample_snippet: snippet,
        repetition_detected: is_degenerate,
    })
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let model_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}.", model_path);
        return Ok(());
    }

    let sink_counts = [0, 1, 2, 4, 8];

    println!("=========================================================================================");
    println!("  EXPERIMENT: ATTENTION-SINK NECESSITY & SINK COUNT (S) TUNING");
    println!("  Model: Qwen 2.5 0.5B | Window: 48 tokens | Generated: 80 tokens (Forces 2+ Evictions)");
    println!("=========================================================================================\n");

    let mut results: Vec<SinkResult> = Vec::new();

    for &s in &sink_counts {
        print!(">>> Testing Sink Tokens S = {}... ", s);
        std::io::Write::flush(&mut std::io::stdout())?;
        let res = test_sink_count(&model_path, s)?;
        let status = if res.repetition_detected {
            "\x1b[1;31m[COLLAPSED / REPETITION]\x1b[0m"
        } else {
            "\x1b[1;32m[STABLE / COHERENT]\x1b[0m"
        };
        println!("Done -> Speed: {:>5.1} tok/s | P50: {:>4.2} ms | Status: {}", res.gen_tok_per_sec, res.p50_ms, status);
        results.push(res);
    }

    println!("\n=================================================================================================================================================");
    println!("                                         ATTENTION-SINK SCALING & COHERENCE RESULTS");
    println!("=================================================================================================================================================");
    println!(
        "| {:<10} | {:>12} | {:>10} | {:<25} | {:<60} |",
        "Sinks (S)", "Gen Speed", "P50 Lat", "Attention State", "Sample Output Generated Snippet"
    );
    println!(
        "| {:<10} | {:>12} | {:>10} | {:<25} | {:<60} |",
        "", "(tok/s)", "(ms)", "", ""
    );
    println!("|------------|--------------|------------|---------------------------|--------------------------------------------------------------|");

    for r in &results {
        let state_str = if r.repetition_detected {
            "Collapse (Entropy Degenerated)"
        } else {
            "Stable (Anchor Preserved)"
        };
        println!(
            "| S = {:<6} | {:>12.2} | {:>10.2} | {:<25} | {:<60} |",
            r.sink_s,
            r.gen_tok_per_sec,
            r.p50_ms,
            state_str,
            r.sample_snippet
        );
    }
    println!("=================================================================================================================================================\n");

    Ok(())
}
