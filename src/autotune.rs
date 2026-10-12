//! Automated Tester, In-Depth Efficiency Heuristic & Model Optimization Engine.
//!
//! Evaluates candidate inference methods across multiple deployment profiles:
//! Balanced (Default), Interactive (Low TTFT & Jitter), Throughput (Raw tok/s),
//! and EdgeMemory (Constrained VRAM & O(1) Bounded Caches).
//!
//! Automatically discovers models, benchmarks candidate methods, ranks them via
//! multi-objective Pareto scoring, and generates architectural rationale reports.

use std::path::{Path, PathBuf};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkReport;
use crate::engine::{
    EngineConfig, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile,
};
use crate::engines::{
    CpuEngine, HybridSinkEngine, QuantizedKvEngine, RawFfiEngine, SafeLlamaEngine,
    SlidingWindowEngine,
};

/// Target deployment profile guiding the efficiency score heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkProfile {
    /// Balanced production score combining throughput, reactivity, and memory.
    Balanced,
    /// Heavily rewards sub-50ms TTFT, sub-1ms jitter (σ), and low P95 latency.
    Interactive,
    /// Maximizes raw generation speed (tokens/sec) for background batch generation.
    Throughput,
    /// Heavily rewards small memory footprints and gives a bonus to O(1) bounded caches.
    EdgeMemory,
}

impl BenchmarkProfile {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Balanced => "Balanced Production",
            Self::Interactive => "Interactive & Realtime (Low TTFT + Jitter)",
            Self::Throughput => "Maximum Throughput (Batch Processing)",
            Self::EdgeMemory => "Edge & Memory-Constrained (O(1) Bounded)",
        }
    }
}

/// Detailed breakdown of the multi-objective efficiency score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub composite_score: f64,
    pub throughput_factor: f64,
    pub reactivity_factor: f64,
    pub smoothness_factor: f64,
    pub memory_factor: f64,
    pub stability_factor: f64,
    pub profile: BenchmarkProfile,
}

impl ScoreBreakdown {
    /// Computes the in-depth multi-dimensional efficiency score.
    pub fn compute(
        profile: BenchmarkProfile,
        gen_tok_per_sec: f64,
        ttft_ms: f64,
        p50_ms: f64,
        sigma_ms: f64,
        total_ram_mib: f64,
        is_bounded_o1: bool,
        is_coherent: bool,
    ) -> Self {
        // 1. Throughput factor (normalized against 100 tok/s interactive baseline)
        let t_norm = (gen_tok_per_sec / 100.0).max(0.01);
        let throughput_factor = match profile {
            BenchmarkProfile::Throughput => t_norm.powf(1.3),
            _ => t_norm.powf(1.0),
        };

        // 2. Reactivity factor (Time To First Token)
        let reactivity_factor = match profile {
            BenchmarkProfile::Interactive => (100.0 / (ttft_ms + 25.0)).powf(1.2),
            BenchmarkProfile::Throughput => (100.0 / (ttft_ms + 100.0)).powf(0.5),
            _ => (100.0 / (ttft_ms + 50.0)).powf(0.85),
        };

        // 3. Smoothness factor (P50 latency + Jitter standard deviation)
        let effective_step_latency = p50_ms + 2.5 * sigma_ms;
        let smoothness_factor = match profile {
            BenchmarkProfile::Interactive => (10.0 / effective_step_latency.max(0.5)).powf(1.2),
            _ => (10.0 / effective_step_latency.max(0.5)).powf(0.8),
        };

        // 4. Memory factor (utilization against 4096 MiB VRAM ceiling + O(1) bonus)
        let vram_budget = 4096.0;
        let mem_ratio = vram_budget / (total_ram_mib + 256.0);
        let bounded_bonus = if is_bounded_o1 { 1.30 } else { 1.00 };
        let memory_factor = match profile {
            BenchmarkProfile::EdgeMemory => mem_ratio.powf(1.2) * bounded_bonus,
            BenchmarkProfile::Throughput => mem_ratio.powf(0.4),
            _ => mem_ratio.powf(0.75) * bounded_bonus,
        };

        // 5. Stability factor (severe penalty for repetition loops or collapse)
        let stability_factor = if is_coherent { 1.0 } else { 0.05 };

        // Composite formulation
        let raw_composite = match profile {
            BenchmarkProfile::Interactive => {
                throughput_factor * reactivity_factor * smoothness_factor * memory_factor.powf(0.5)
            }
            BenchmarkProfile::Throughput => {
                throughput_factor * reactivity_factor.powf(0.4) * smoothness_factor.powf(0.6) * memory_factor.powf(0.3)
            }
            BenchmarkProfile::EdgeMemory => {
                throughput_factor.powf(0.8) * reactivity_factor.powf(0.7) * smoothness_factor * memory_factor
            }
            BenchmarkProfile::Balanced => {
                throughput_factor * reactivity_factor.powf(0.8) * smoothness_factor.powf(0.8) * memory_factor.powf(0.8)
            }
        };

        let composite_score = (raw_composite * 100.0 * stability_factor).max(0.0);

        Self {
            composite_score,
            throughput_factor,
            reactivity_factor,
            smoothness_factor,
            memory_factor,
            stability_factor,
            profile,
        }
    }
}

/// Linguistic repetition checker detecting n-gram degeneration.
pub fn is_text_coherent(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 8 {
        return true;
    }
    for i in 0..words.len().saturating_sub(6) {
        let tri1 = (&words[i], &words[i + 1], &words[i + 2]);
        let tri2 = (&words[i + 3], &words[i + 4], &words[i + 5]);
        if tri1 == tri2 {
            return false;
        }
    }
    true
}

/// Evaluation record for a single engine candidate.
#[derive(Debug, Clone)]
pub struct EvaluatedCandidate {
    pub method_name: String,
    pub kv_precision: String,
    pub is_bounded_o1: bool,
    pub is_coherent: bool,
    pub memory: ModelMemoryProfile,
    pub report: BenchmarkReport,
    pub score: ScoreBreakdown,
}

/// AutoTuner outcome summary for a model.
#[derive(Debug, Clone)]
pub struct ModelTuningReport {
    pub model_path: PathBuf,
    pub model_name: String,
    pub profile: BenchmarkProfile,
    pub candidates: Vec<EvaluatedCandidate>,
    pub winning_index: usize,
    pub architectural_rationale: String,
}

impl ModelTuningReport {
    pub fn winner(&self) -> &EvaluatedCandidate {
        &self.candidates[self.winning_index]
    }
}

/// Automatic Model Tester and Method Optimizer.
pub struct AutoTuner {
    profile: BenchmarkProfile,
    test_prompt: String,
    params: GenerationParams,
}

impl AutoTuner {
    pub fn new(profile: BenchmarkProfile) -> Self {
        Self {
            profile,
            test_prompt: "Explain in one clear sentence why memory bandwidth limits LLM decoding speed.".to_string(),
            params: GenerationParams {
                max_tokens: 36,
                temperature: 0.0,
                top_p: 1.0,
                seed: 42,
            },
        }
    }

    /// Evaluates all viable engine candidates on the given GGUF model and selects the winner.
    pub fn tune_model(&self, model_path: &Path) -> Result<ModelTuningReport> {
        let model_filename = model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "Unknown Model".to_string());

        let mut evaluated: Vec<EvaluatedCandidate> = Vec::new();

        // Load model weights once for GPU-based candidates (prevents reloading gigabytes of weights 4-5 times)
        let shared_gpu_model = crate::engine::LlamaModelHandle::load_from_file(model_path, 999, Some(1)).ok();

        // 1. Raw C-ABI FFI (FP16 Baseline)
        {
            let cfg = EngineConfig::new(model_path).with_device(Some(1)).with_gpu_layers(999);
            let eng_res = match &shared_gpu_model {
                Some(shared) => RawFfiEngine::from_shared_model(std::sync::Arc::clone(shared), cfg),
                None => RawFfiEngine::new(cfg),
            };
            if let Ok(mut eng) = eng_res {
                if let Ok(cand) = self.evaluate_engine("Raw C-ABI FFI (FP16)", false, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // 2. Hardware Quantized KV Cache (Q8_0)
        {
            let cfg = EngineConfig::new(model_path).with_device(Some(1)).with_gpu_layers(999);
            let eng_res = match &shared_gpu_model {
                Some(shared) => QuantizedKvEngine::from_shared_model(std::sync::Arc::clone(shared), cfg, KvPrecision::Q8_0),
                None => QuantizedKvEngine::new(cfg, KvPrecision::Q8_0),
            };
            if let Ok(mut eng) = eng_res {
                if let Ok(cand) = self.evaluate_engine("Quantized KV (Q8_0)", false, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // 3. Sliding Window Attention (FP16, S=4, Win=64, Chunk=16)
        {
            let cfg = EngineConfig::new(model_path).with_device(Some(1)).with_gpu_layers(999);
            let eng_res = match &shared_gpu_model {
                Some(shared) => SlidingWindowEngine::from_shared_model(std::sync::Arc::clone(shared), cfg, 4, 64, 16),
                None => SlidingWindowEngine::new(cfg, 4, 64, 16),
            };
            if let Ok(mut eng) = eng_res {
                if let Ok(cand) = self.evaluate_engine("Sliding Window (FP16, k=16)", true, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // 4. Hybrid Quantized Sink Engine (Q8_0 + S=4 + Win=64 + Chunk=16)
        {
            let cfg = EngineConfig::new(model_path).with_device(Some(1)).with_gpu_layers(999);
            let eng_res = match &shared_gpu_model {
                Some(shared) => HybridSinkEngine::from_shared_model(std::sync::Arc::clone(shared), cfg, KvPrecision::Q8_0, 4, 64, 16),
                None => HybridSinkEngine::new(cfg, KvPrecision::Q8_0, 4, 64, 16),
            };
            if let Ok(mut eng) = eng_res {
                if let Ok(cand) = self.evaluate_engine("Hybrid Sink (Q8_0, k=16)", true, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // 5. Safe Rust Wrapper (llama-cpp-2)
        {
            let cfg = EngineConfig::new(model_path).with_device(Some(1)).with_gpu_layers(999);
            if let Ok(mut eng) = SafeLlamaEngine::new(cfg) {
                if let Ok(cand) = self.evaluate_engine("Safe Rust Wrapper (FP16)", false, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // 6. Pure Host CPU Baseline
        {
            let cfg = EngineConfig::new(model_path);
            if let Ok(mut eng) = CpuEngine::new(cfg) {
                if let Ok(cand) = self.evaluate_engine("Pure Host CPU (12 Threads)", false, &mut eng) {
                    evaluated.push(cand);
                }
            }
        }

        // Sort candidates descending by composite score
        evaluated.sort_by(|a, b| b.score.composite_score.partial_cmp(&a.score.composite_score).unwrap());

        // Derive architectural rationale
        let winner = &evaluated[0];
        let rationale = self.derive_rationale(&model_filename, winner);

        Ok(ModelTuningReport {
            model_path: model_path.to_path_buf(),
            model_name: model_filename,
            profile: self.profile,
            candidates: evaluated,
            winning_index: 0,
            architectural_rationale: rationale,
        })
    }

    fn evaluate_engine(
        &self,
        name: &str,
        is_bounded: bool,
        engine: &mut dyn ModelEngine,
    ) -> Result<EvaluatedCandidate> {
        let output = engine.generate(&self.test_prompt, &self.params, None)?;
        let coherent = is_text_coherent(&output.text);

        let score = ScoreBreakdown::compute(
            self.profile,
            output.report.generation_tok_per_sec,
            output.report.time_to_first_token.as_secs_f64() * 1000.0,
            output.report.latency_p50_ms,
            output.report.latency_std_dev_ms,
            output.memory.total_memory_mib(),
            is_bounded,
            coherent,
        );

        Ok(EvaluatedCandidate {
            method_name: name.to_string(),
            kv_precision: output.memory.kv_precision.label().split(' ').next().unwrap_or("").to_string(),
            is_bounded_o1: is_bounded,
            is_coherent: coherent,
            memory: output.memory,
            report: output.report,
            score,
        })
    }

    fn derive_rationale(&self, model_name: &str, winner: &EvaluatedCandidate) -> String {
        let mem = &winner.memory;
        let rep = &winner.report;
        format!(
            "Model '{}' ({} params, {} layers, {} KV heads) achieved optimal efficiency using '{}'. With a base weights footprint of {:.1} MiB and KV footprint of {:.2} KiB/token, this method delivered {:.1} tok/s with a TTFT of {:.1} ms and P50 latency of {:.2} ms (σ={:.2} ms), while holding memory to {:.1} MiB.",
            model_name,
            mem.total_params,
            mem.n_layers,
            mem.n_kv_heads,
            winner.method_name,
            mem.weights_mib(),
            mem.kv_kib_per_token(),
            rep.generation_tok_per_sec,
            rep.time_to_first_token.as_secs_f64() * 1000.0,
            rep.latency_p50_ms,
            rep.latency_std_dev_ms,
            mem.total_memory_mib()
        )
    }
}
