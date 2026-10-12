//! Mathematical metrics and evaluation harnesses for KV cache compression.
//! Computes reconstruction error, cosine similarity, entropy, memory savings, and quality preservation.

use serde::{Deserialize, Serialize};

/// Comprehensive statistical quality report for a compressed KV cache technique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionQualityReport {
    /// Name of the compression algorithm
    pub method_name: String,
    /// Whether this method is mathematically lossless (bit-for-bit)
    pub is_lossless: bool,
    /// Uncompressed KV cache bytes per token
    pub uncompressed_bytes_per_tok: usize,
    /// Effective compressed KV cache bytes per token
    pub compressed_bytes_per_tok: f64,
    /// Compression ratio: uncompressed / compressed (e.g. 4.0x)
    pub compression_ratio: f64,
    /// Percentage memory saved: (1 - compressed/uncompressed) * 100
    pub memory_reduction_pct: f64,
    /// Average cosine similarity against ground truth activations (1.0 = identical)
    pub mean_cosine_similarity: f64,
    /// Relative Frobenius error ||K - K_hat||_F / ||K||_F (0.0 = perfect reconstruction)
    pub relative_frobenius_error: f64,
    /// Maximum absolute deviation across all tensor elements
    pub max_absolute_error: f32,
    /// Estimated output token agreement rate (0.0% to 100.0%)
    pub token_agreement_pct: f64,
    /// Algorithmic complexity of decompressing during attention
    pub decompression_complexity: String,
    /// Synthesized Innovation Score (0 - 1000)
    pub innovation_score: f64,
}

/// Compute cosine similarity between two float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }

    if norm_a <= 1e-12 || norm_b <= 1e-12 {
        return 1.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
}

/// Compute relative Frobenius norm error: ||Original - Reconstructed||_F / ||Original||_F
pub fn relative_frobenius_error(orig: &[f32], recon: &[f32]) -> f64 {
    if orig.len() != recon.len() || orig.is_empty() {
        return 1.0;
    }
    let mut diff_sq_sum = 0.0f64;
    let mut orig_sq_sum = 0.0f64;

    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let diff = (o - r) as f64;
        diff_sq_sum += diff * diff;
        orig_sq_sum += (o as f64) * (o as f64);
    }

    if orig_sq_sum <= 1e-12 {
        return 0.0;
    }
    (diff_sq_sum.sqrt() / orig_sq_sum.sqrt()).clamp(0.0, 10.0)
}

/// Compute maximum absolute difference across all elements.
pub fn max_absolute_error(orig: &[f32], recon: &[f32]) -> f32 {
    let mut max_err = 0.0f32;
    for (&o, &r) in orig.iter().zip(recon.iter()) {
        let err = (o - r).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

/// Synthesize a deep innovation score balancing compression ratio, fidelity, and novelty.
/// Score = 100 * (CompressionRatio)^0.7 * (CosineSim)^3.0 * (1 / (1 + RelError)) * NoveltyMultiplier
pub fn calculate_innovation_score(
    compression_ratio: f64,
    cosine_sim: f64,
    relative_error: f64,
    novelty_multiplier: f64,
) -> f64 {
    let cr_term = compression_ratio.max(1.0).powf(0.75);
    let fidelity_term = cosine_sim.max(0.0).powi(4);
    let error_term = 1.0 / (1.0 + 2.0 * relative_error);
    
    100.0 * cr_term * fidelity_term * error_term * novelty_multiplier
}
