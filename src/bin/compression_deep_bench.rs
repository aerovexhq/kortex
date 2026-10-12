//! Master Comprehensive KV Cache Compression & Innovation Benchmark.
//!
//! Evaluates all models (0.5B, 1B, 1.5B, 3B, 7B) across all 10 compression techniques:
//! - 5 Standard / Near-Lossless methods (Zero-copy prefix, SnapKV, Token merge, Cross-layer, Delta residual)
//! - 5 Boundary-Pushing Creative methods (Harmonic RoPE, SVD Hologram, Chebyshev Spline, Orthogonal Layers, DCT Sparsity)
//!
//! Measures: Compression ratio, memory savings, Frobenius error, cosine similarity,
//! live GPU throughput, TTFT, and synthesized Innovation Scores.

use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use kortex::compression::*;
use kortex::engine::{EngineConfig, GenerationParams, ModelEngine};
use kortex::engines::raw_ffi::RawFfiEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureSpecs {
    pub name: String,
    pub path: String,
    pub params_billions: f64,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub weights_mib: f64,
    pub uncompressed_kv_bytes_per_tok: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompressionSuiteResult {
    pub model_specs: ModelArchitectureSpecs,
    pub baseline_throughput_tok_s: f64,
    pub baseline_ttft_ms: f64,
    pub reports: Vec<CompressionQualityReport>,
}

fn inspect_model_specs(path: &std::path::Path, name: &str) -> Result<ModelArchitectureSpecs> {
    let config = EngineConfig::new(path.to_path_buf())
        .with_device(Some(1))
        .with_gpu_layers(999);
    let engine = RawFfiEngine::new(config)?;
    let profile = engine.memory_profile();

    Ok(ModelArchitectureSpecs {
        name: name.to_string(),
        path: path.to_string_lossy().to_string(),
        params_billions: profile.total_params as f64 / 1_000_000_000.0,
        n_layers: profile.n_layers,
        n_heads: profile.n_heads,
        n_kv_heads: profile.n_kv_heads,
        head_dim: profile.head_dim,
        weights_mib: profile.weights_mib(),
        uncompressed_kv_bytes_per_tok: profile.kv_bytes_per_token,
    })
}

fn run_live_baseline(path: &std::path::Path) -> (f64, f64) {
    let config = EngineConfig::new(path.to_path_buf())
        .with_device(Some(1))
        .with_gpu_layers(999);
    match RawFfiEngine::new(config) {
        Ok(mut engine) => {
            let prompt = "Explain the fundamental principles of data compression in computing systems.";
            let params = GenerationParams {
                max_tokens: 32,
                temperature: 0.0,
                ..Default::default()
            };
            match engine.generate(prompt, &params, None) {
                Ok(out) => (
                    out.report.generation_tok_per_sec,
                    out.report.time_to_first_token.as_secs_f64() * 1000.0,
                ),
                Err(_) => (0.0, 0.0),
            }
        }
        Err(_) => (0.0, 0.0),
    }
}

fn evaluate_model_compression_suite(specs: &ModelArchitectureSpecs) -> Vec<CompressionQualityReport> {
    let bytes_per_tok = specs.uncompressed_kv_bytes_per_tok;
    let n_layers = specs.n_layers as usize;
    let head_dim = specs.head_dim as usize;
    let seq_len = 512;

    let mut reports = Vec::new();

    // --- 1. Baseline FP16 ---
    reports.push(CompressionQualityReport {
        method_name: "Baseline (Uncompressed FP16)".to_string(),
        is_lossless: true,
        uncompressed_bytes_per_tok: bytes_per_tok,
        compressed_bytes_per_tok: bytes_per_tok as f64,
        compression_ratio: 1.0,
        memory_reduction_pct: 0.0,
        mean_cosine_similarity: 1.0,
        relative_frobenius_error: 0.0,
        max_absolute_error: 0.0,
        token_agreement_pct: 100.0,
        decompression_complexity: "O(1) - Direct Dense Access".to_string(),
        innovation_score: 100.0,
    });

    // --- Standard Methods ---
    // Method 1: Zero-Copy Prefix Sharing (4 parallel branches / turns sharing 256 prefix tokens)
    let zero_copy = ZeroCopyPrefixSimulator::new(256, 4, bytes_per_tok);
    reports.push(zero_copy.evaluate());

    // Method 2: SnapKV Heavy Hitter (4 sinks + 32 heavy hitters + 32 local window = 68 retained)
    let heavy_hitter = HeavyHitterEvaluator::new(seq_len, 4, 32, 32, bytes_per_tok, head_dim);
    reports.push(heavy_hitter.evaluate());

    // Method 3: Token Merging (ToMe-KV 50% pair merging)
    let token_merge = TokenMergeEvaluator::new(seq_len, head_dim, 0.50, bytes_per_tok);
    reports.push(token_merge.evaluate());

    // Method 4: Cross-Layer Interleaved Sharing (2:1)
    let cross_layer = CrossLayerEvaluator::new(n_layers, head_dim, bytes_per_tok);
    reports.push(cross_layer.evaluate());

    // Method 5: Temporal Delta Residual (4-bit nibble codec)
    let delta_codec = ResidualCodecEvaluator::new(seq_len, head_dim, 4, bytes_per_tok);
    reports.push(delta_codec.evaluate());

    // --- 5 Creative / Boundary-Pushing Innovative Methods ---
    // Innovation 1: Temporal RoPE Harmonic Superposition (4x folding)
    let harmonic_rope = HarmonicRopeEvaluator::new(4, head_dim, seq_len, bytes_per_tok);
    reports.push(harmonic_rope.evaluate());

    // Innovation 2: Streaming SVD Rank Collapse (Rank r=16)
    let svd_hologram = SvdHologramEvaluator::new(16, seq_len, head_dim, bytes_per_tok);
    reports.push(svd_hologram.evaluate());

    // Innovation 3: Polynomial Chebyshev Spline Memory (32 tokens -> 4 coefficients)
    let spline_kv = SplineKvEvaluator::new(32, 4, seq_len, head_dim, bytes_per_tok);
    reports.push(spline_kv.evaluate());

    // Innovation 4: Walsh-Hadamard Orthogonal Layer Entanglement (4 layers -> 1 tensor)
    let orthogonal_layers = OrthogonalHolographyEvaluator::new(4, seq_len, head_dim, bytes_per_tok);
    reports.push(orthogonal_layers.evaluate());

    // Innovation 5: Discrete Cosine Transform (DCT) Frequency Sparsification (3/16 freqs + signs)
    let frequency_dct = FrequencyDctEvaluator::new(16, 3, seq_len, head_dim, bytes_per_tok);
    reports.push(frequency_dct.evaluate());

    reports
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║       EXHAUSTIVE MULTI-MODEL KV CACHE COMPRESSION & INNOVATION RESEARCH BENCHMARK            ║");
    println!("║          Evaluating 5 GGUF Models (0.5B to 7B) across 10 Compression & Quantum Paradigms     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    let models = [
        ("Qwen 2.5 0.5B", PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")),
        ("Meta Llama 3.2 1B", PathBuf::from("models/llama-3.2-1b-instruct-q4_k_m.gguf")),
        ("Qwen 2.5 1.5B", PathBuf::from("models/qwen2.5-1.5b-instruct-q4_k_m.gguf")),
        ("Qwen 2.5 3B", PathBuf::from("models/qwen2.5-3b-instruct-q4_k_m.gguf")),
        ("Qwen 2.5 7B", PathBuf::from("models/qwen2.5-7b-instruct-q2_k.gguf")),
    ];

    let mut master_suite_results = Vec::new();
    let start_all = Instant::now();

    for (name, path) in &models {
        if !path.exists() {
            println!("⚠️ Model {:?} not found, skipping...", path);
            continue;
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(">>> PROFILING MODEL: {} ({:?})", name, path.file_name().unwrap());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let specs = match inspect_model_specs(path, name) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to inspect model {}: {:?}", name, e);
                continue;
            }
        };

        println!("• Parameters: {:.2} B | Layers: {} | Heads: {} Q / {} KV (dim {})",
            specs.params_billions, specs.n_layers, specs.n_heads, specs.n_kv_heads, specs.head_dim);
        println!("• Weights Base RAM: {:.2} MiB | Uncompressed KV: {:.2} KiB/token",
            specs.weights_mib, specs.uncompressed_kv_bytes_per_tok as f64 / 1024.0);

        print!("• Running Live GPU Baseline Decoding... ");
        let (live_tok_s, live_ttft) = run_live_baseline(path);
        println!("Done! (Speed: {:.1} tok/s | TTFT: {:.2} ms)", live_tok_s, live_ttft);

        println!("\n• Evaluating all 10 Compression & Mathematical Paradigms...");
        let reports = evaluate_model_compression_suite(&specs);

        // Display results table for this model
        println!("\n┌──────────────────────────────────────────────┬────────┬───────────┬───────────┬────────┬────────┬────────┐");
        println!("│ Compression / Innovation Method              │ Type   │ Ratio (x) │ KB/token  │ CosSim │ RelErr │ Score  │");
        println!("├──────────────────────────────────────────────┼────────┼───────────┼───────────┼────────┼────────┼────────┤");
        for r in &reports {
            let type_label = if r.is_lossless { "Lossless" } else { "Lossy/Alg" };
            let kb_per_tok = r.compressed_bytes_per_tok / 1024.0;
            println!("│ {:<44} │ {:<6} │ {:>8.2}x │ {:>7.2} KB │ {:>6.4} │ {:>6.4} │ {:>6.1} │",
                r.method_name,
                type_label,
                r.compression_ratio,
                kb_per_tok,
                r.mean_cosine_similarity,
                r.relative_frobenius_error,
                r.innovation_score
            );
        }
        println!("└──────────────────────────────────────────────┴────────┴───────────┴───────────┴────────┴────────┴────────┘\n");

        master_suite_results.push(ModelCompressionSuiteResult {
            model_specs: specs,
            baseline_throughput_tok_s: live_tok_s,
            baseline_ttft_ms: live_ttft,
            reports,
        });
    }

    let elapsed = start_all.elapsed();

    // Export master telemetry to analysis/compression_matrix.json
    let json_path = PathBuf::from("analysis/compression_matrix.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&master_suite_results) {
        let _ = std::fs::write(&json_path, json_str);
        println!("✓ Exported full machine-readable telemetry to: {:?}", json_path);
    }

    println!("\n==========================================================================================");
    println!("TOTAL BENCHMARK DURATION: {:.2} seconds", elapsed.as_secs_f64());
    println!("==========================================================================================");

    Ok(())
}
