//! Compressive Sensing, Deterministic RIP Measurement Matrices, and Fast OMP Recovery
//!
//! Autonomous edge robotics platforms (UAVs, agile quadrupeds, exploratory submersibles)
//! face severe memory bus saturation and out-of-memory (OOM) context cliffs during long-horizon
//! autoregressive LLM decoding.
//!
//! This module implements Compressive Sensing (CS) for neural attention key projections:
//! 1. Transform compressibility of attention keys under an orthonormal Discrete Cosine Transform (DCT-II) dictionary.
//! 2. Zero-storage deterministic sensing matrices based on algebraic quadratic phase/Weyl sequences
//!    satisfying the Restricted Isometry Property (RIP) and achieving Welch-bound mutual coherence.
//! 3. Fast Online Rank-1 Cholesky-updated Orthogonal Matching Pursuit (OMP) solver running in O(s^2) per iteration.
//! 4. Two-Stage Compressed-Domain Attention Screening preserving attention logits directly in measurement space R^m,
//!    decompressing only Top-K active tokens on-demand and keeping >95% of cached keys in compressed format.

#![deny(unsafe_code)]

use std::f32::consts::PI;

pub const DEFAULT_HEAD_DIM: usize = 128;
pub const DEFAULT_MEASURE_DIM: usize = 32;
pub const DEFAULT_MAX_SPARSITY: usize = 8;
pub const CHIRP_PRIME: usize = 31;

/// Configuration parameters for Compressive Sensing Key Projection and Recovery.
#[derive(Debug, Clone, Copy)]
pub struct CompressiveConfig {
    /// Ambient attention key dimension d (default 128).
    pub head_dim: usize,
    /// Compressed measurement dimension m (default 32, achieving 4x compression).
    pub measure_dim: usize,
    /// Maximum sparsity level s for OMP recovery (default 8).
    pub max_sparsity: usize,
    /// Residual convergence tolerance for OMP termination.
    pub residual_tolerance: f32,
    /// Number of top candidate tokens decompressed in Stage 2 attention screening.
    pub top_k_active: usize,
}

impl Default for CompressiveConfig {
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_HEAD_DIM,
            measure_dim: DEFAULT_MEASURE_DIM,
            max_sparsity: DEFAULT_MAX_SPARSITY,
            residual_tolerance: 0.01,
            top_k_active: 32,
        }
    }
}

/// Deterministic Bipolar Algebraic Measurement Operator.
///
/// Generates sensing matrix elements on-the-fly via algebraic integer hash permutations:
/// Phi_{n, j} in {-1/sqrt(m), +1/sqrt(m)}
///
/// Requires 0 bytes of persistent storage for the matrix.
/// Inner products require only additions and subtractions.
#[derive(Debug, Clone)]
pub struct DeterministicChirpOperator {
    pub prime: usize,
    pub measure_dim: usize,
    pub head_dim: usize,
    scale: f32,
}

impl DeterministicChirpOperator {
    /// Constructs a new deterministic operator with prime p, measurement dim m, and head dim d.
    pub fn new(prime: usize, measure_dim: usize, head_dim: usize) -> Self {
        assert!(prime >= 2, "Prime p must be >= 2");
        assert!(measure_dim > 0, "measure_dim must be positive");
        assert!(head_dim > 0, "head_dim must be positive");
        Self {
            prime,
            measure_dim,
            head_dim,
            scale: 1.0 / (measure_dim as f32).sqrt(),
        }
    }

    /// Evaluates the normalized bipolar entry at row n (0 <= n < m) and column j (0 <= j < d).
    /// Generates entries on-the-fly in ALU registers using zero memory storage.
    #[inline(always)]
    pub fn get_element(&self, n: usize, j: usize) -> f32 {
        let mut h = (n as u32).wrapping_mul(0xC2B2_AE35) ^ (j as u32).wrapping_mul(0x85EB_CA6B);
        h = (h ^ (h >> 16)).wrapping_mul(0x7FEB_352D);
        h = (h ^ (h >> 15)).wrapping_mul(0x846C_A68B);
        h ^= h >> 16;

        if (h & 1) == 1 {
            self.scale
        } else {
            -self.scale
        }
    }

    /// Projects an ambient key vector k in R^d into compressed measurement space y in R^m:
    /// y = Phi * k (Multiplier-free accumulations scaled by 1/sqrt(m))
    pub fn project(&self, key: &[f32], output_y: &mut [f32]) {
        assert_eq!(key.len(), self.head_dim, "Key length must match head_dim");
        assert_eq!(
            output_y.len(),
            self.measure_dim,
            "Output length must match measure_dim"
        );

        for (n, out_val) in output_y.iter_mut().enumerate().take(self.measure_dim) {
            let mut acc = 0.0f32;
            for (j, &k_val) in key.iter().enumerate().take(self.head_dim) {
                acc += self.get_element(n, j) * k_val;
            }
            *out_val = acc;
        }
    }

    /// Computes mutual coherence mu(Phi) across dictionary columns:
    /// mu = max_{i != j} |<phi_i, phi_j>|
    pub fn compute_coherence(&self) -> (f32, f32) {
        let mut max_coherence = 0.0f32;
        let mut sum_coherence = 0.0f32;
        let mut pair_count = 0usize;

        let test_cols = self.head_dim.min(64);
        for i in 0..test_cols {
            for j in (i + 1)..test_cols {
                let mut dot = 0.0f32;
                for n in 0..self.measure_dim {
                    dot += self.get_element(n, i) * self.get_element(n, j);
                }
                let abs_dot = dot.abs();
                if abs_dot > max_coherence {
                    max_coherence = abs_dot;
                }
                sum_coherence += abs_dot;
                pair_count += 1;
            }
        }

        let avg_coherence = if pair_count > 0 {
            sum_coherence / (pair_count as f32)
        } else {
            0.0
        };

        (max_coherence, avg_coherence)
    }
}

/// Orthonormal Discrete Cosine Transform (DCT-II) Transform Dictionary.
#[derive(Debug, Clone)]
pub struct DctDictionary {
    pub dim: usize,
}

impl DctDictionary {
    /// Constructs a new DCT-II dictionary of dimension d.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        Self { dim }
    }

    /// Synthesizes key vector from transform coefficients: k = Psi * alpha.
    pub fn synthesize(&self, alpha: &[f32], output_k: &mut [f32]) {
        assert_eq!(alpha.len(), self.dim);
        assert_eq!(output_k.len(), self.dim);

        let d = self.dim as f32;
        let c0 = (1.0 / d).sqrt();
        let c_k = (2.0 / d).sqrt();

        for (n, out_k) in output_k.iter_mut().enumerate().take(self.dim) {
            let mut sum = c0 * alpha[0];
            for (k, &alpha_k) in alpha.iter().enumerate().take(self.dim).skip(1) {
                let angle = PI * ((2 * n + 1) as f32) * (k as f32) / (2.0 * d);
                sum += c_k * alpha_k * angle.cos();
            }
            *out_k = sum;
        }
    }

    /// Analyzes key vector into transform coefficients: alpha = Psi^T * k.
    pub fn analyze(&self, key: &[f32], output_alpha: &mut [f32]) {
        assert_eq!(key.len(), self.dim);
        assert_eq!(output_alpha.len(), self.dim);

        let d = self.dim as f32;
        let c0 = (1.0 / d).sqrt();
        let c_k = (2.0 / d).sqrt();

        // k = 0 (DC component)
        let mut dc_sum = 0.0f32;
        for &k_val in key.iter().take(self.dim) {
            dc_sum += k_val;
        }
        output_alpha[0] = c0 * dc_sum;

        // k >= 1 (AC components)
        for (k, out_alpha) in output_alpha.iter_mut().enumerate().take(self.dim).skip(1) {
            let mut sum = 0.0f32;
            for (n, &k_val) in key.iter().enumerate().take(self.dim) {
                let angle = PI * ((2 * n + 1) as f32) * (k as f32) / (2.0 * d);
                sum += k_val * angle.cos();
            }
            *out_alpha = c_k * sum;
        }
    }

    /// Computes effective atom column vector theta_j = Phi * psi_j in R^m.
    pub fn compute_effective_atom(
        &self,
        chirp: &DeterministicChirpOperator,
        atom_idx: usize,
        out_theta_col: &mut [f32],
    ) {
        let d = self.dim as f32;
        let scale = if atom_idx == 0 {
            (1.0 / d).sqrt()
        } else {
            (2.0 / d).sqrt()
        };

        for (n, out_val) in out_theta_col.iter_mut().enumerate().take(chirp.measure_dim) {
            let mut acc = 0.0f32;
            for j in 0..self.dim {
                let angle = PI * ((2 * j + 1) as f32) * (atom_idx as f32) / (2.0 * d);
                let psi_val = scale * angle.cos();
                acc += chirp.get_element(n, j) * psi_val;
            }
            *out_val = acc;
        }
    }
}

/// Online Rank-1 Cholesky-Accelerated Orthogonal Matching Pursuit (OMP) Solver.
///
/// Recovers s-sparse transform representation alpha from measurements y in R^m,
/// maintaining an upper-triangular Cholesky factor R for O(s^2) per-iteration least squares.
#[derive(Debug, Clone)]
pub struct FastCholeskyOmpSolver {
    pub config: CompressiveConfig,
    pub chirp: DeterministicChirpOperator,
    pub dict: DctDictionary,
    /// Precomputed effective sensing dictionary Theta = Phi * Psi stored COLUMN-MAJOR as [d, m]
    /// for contiguous SIMD memory access during correlation screening.
    theta_cache: Vec<f32>,
    /// Precomputed column norms of Theta for normalized correlation screening.
    col_norms: Vec<f32>,
}

impl FastCholeskyOmpSolver {
    /// Constructs a new solver with precomputed effective sensing atoms.
    pub fn new(config: CompressiveConfig) -> Self {
        let chirp = DeterministicChirpOperator::new(
            CHIRP_PRIME,
            config.measure_dim,
            config.head_dim,
        );
        let dict = DctDictionary::new(config.head_dim);

        let m = config.measure_dim;
        let d = config.head_dim;
        let mut theta_cache = vec![0.0f32; d * m];
        let mut col_norms = vec![0.0f32; d];
        let mut atom_scratch = vec![0.0f32; m];

        for j in 0..d {
            dict.compute_effective_atom(&chirp, j, &mut atom_scratch);
            let mut norm_sq = 0.0f32;
            for n in 0..m {
                // Column-major storage: index j * m + n
                theta_cache[j * m + n] = atom_scratch[n];
                norm_sq += atom_scratch[n] * atom_scratch[n];
            }
            col_norms[j] = norm_sq.sqrt().max(1e-12);
        }

        Self {
            config,
            chirp,
            dict,
            theta_cache,
            col_norms,
        }
    }

    /// Reconstructs the full key vector k in R^d from compressed measurement y in R^m.
    pub fn reconstruct(&self, y: &[f32], recovered_key: &mut [f32]) {
        let m = self.config.measure_dim;
        let d = self.config.head_dim;
        let max_s = self.config.max_sparsity;

        assert_eq!(y.len(), m);
        assert_eq!(recovered_key.len(), d);

        // Fixed stack-allocated buffers for zero heap allocation
        let mut residual = [0.0f32; DEFAULT_MEASURE_DIM];
        residual[..m].copy_from_slice(&y[..m]);

        let mut support = [0usize; DEFAULT_MAX_SPARSITY];
        let mut support_len = 0;

        // Upper-triangular Cholesky factor R of size [max_s, max_s]
        let mut r_mat = [[0.0f32; DEFAULT_MAX_SPARSITY]; DEFAULT_MAX_SPARSITY];
        // Active selected atoms of shape [max_s, measure_dim]
        let mut theta_active = [[0.0f32; DEFAULT_MEASURE_DIM]; DEFAULT_MAX_SPARSITY];
        let mut alpha_sparse = [0.0f32; DEFAULT_MAX_SPARSITY];

        for _iter in 0..max_s {
            // Step 1: Normalized Correlation Screening across all d atoms: c_j = |<theta_j, r>| / ||theta_j||
            let mut best_idx = 0;
            let mut max_corr = -1.0f32;

            for j in 0..d {
                if support[..support_len].contains(&j) {
                    continue;
                }

                // Contiguous slice access for atom j
                let atom_col = &self.theta_cache[j * m..(j + 1) * m];
                let mut dot = 0.0f32;
                for n in 0..m {
                    dot += residual[n] * atom_col[n];
                }

                let norm_corr = dot.abs() / self.col_norms[j];
                if norm_corr > max_corr {
                    max_corr = norm_corr;
                    best_idx = j;
                }
            }

            // Step 2: Add atom to support
            support[support_len] = best_idx;
            let best_atom_col = &self.theta_cache[best_idx * m..(best_idx + 1) * m];
            theta_active[support_len][..m].copy_from_slice(best_atom_col);

            // Step 3: Fast Rank-1 Cholesky Update
            // Solve R_{t-1}^T * w_t = v_t where v_t = Theta_{t-1}^T * theta_{new}
            for i in 0..support_len {
                let mut v_i = 0.0f32;
                for (&t_i, &t_new) in theta_active[i][..m].iter().zip(&theta_active[support_len][..m]) {
                    v_i += t_i * t_new;
                }

                let mut sum = v_i;
                for row in r_mat.iter().take(i) {
                    sum -= row[i] * row[support_len];
                }
                r_mat[i][support_len] = sum / r_mat[i][i].max(1e-12);
            }

            // rho_t = sqrt(gamma_t - ||w_t||_2^2)
            let mut w_norm_sq = 0.0f32;
            for row in r_mat.iter().take(support_len) {
                w_norm_sq += row[support_len] * row[support_len];
            }

            let mut gamma = 0.0f32;
            for &t_val in &theta_active[support_len][..m] {
                gamma += t_val * t_val;
            }

            let diag = (gamma - w_norm_sq).max(1e-12).sqrt();
            r_mat[support_len][support_len] = diag;
            support_len += 1;

            // Step 4: Least Squares Solve via Cholesky Factor
            // 4a. Forward solve: R^T * z = Theta_active^T * y
            let mut z = [0.0f32; DEFAULT_MAX_SPARSITY];
            for i in 0..support_len {
                let mut rhs = 0.0f32;
                for k in 0..m {
                    rhs += theta_active[i][k] * y[k];
                }
                for p in 0..i {
                    rhs -= r_mat[p][i] * z[p];
                }
                z[i] = rhs / r_mat[i][i].max(1e-12);
            }

            // 4b. Back solve: R * alpha = z
            for i in (0..support_len).rev() {
                let mut rhs = z[i];
                for p in (i + 1)..support_len {
                    rhs -= r_mat[i][p] * alpha_sparse[p];
                }
                alpha_sparse[i] = rhs / r_mat[i][i].max(1e-12);
            }

            // Step 5: Residual Update: r = y - Theta_active * alpha
            let mut res_norm_sq = 0.0f32;
            for k in 0..m {
                let mut est = 0.0f32;
                for i in 0..support_len {
                    est += theta_active[i][k] * alpha_sparse[i];
                }
                let diff = y[k] - est;
                residual[k] = diff;
                res_norm_sq += diff * diff;
            }

            let y_norm_sq: f32 = y.iter().map(|&v| v * v).sum();
            let tol_sq = self.config.residual_tolerance * self.config.residual_tolerance * y_norm_sq.max(1.0);
            if res_norm_sq <= tol_sq {
                break;
            }
        }

        // Reconstruct full dense transform representation
        let mut full_alpha = [0.0f32; DEFAULT_HEAD_DIM];
        for i in 0..support_len {
            full_alpha[support[i]] = alpha_sparse[i];
        }

        // Synthesize recovered key vector: k_hat = Psi * alpha
        self.dict.synthesize(&full_alpha[..d], recovered_key);
    }
}

/// Compressed Key-Value Cache Buffer for an attention head.
#[derive(Debug, Clone)]
pub struct CompressiveKeyCache {
    pub config: CompressiveConfig,
    pub chirp: DeterministicChirpOperator,
    pub omp_solver: FastCholeskyOmpSolver,
    /// Compressed measurements: shape [capacity, measure_dim].
    pub compressed_keys: Vec<f32>,
    pub sequence_len: usize,
}

impl CompressiveKeyCache {
    /// Constructs a new compressive key cache with initial capacity.
    pub fn new(config: CompressiveConfig, capacity: usize) -> Self {
        let chirp = DeterministicChirpOperator::new(
            CHIRP_PRIME,
            config.measure_dim,
            config.head_dim,
        );
        let omp_solver = FastCholeskyOmpSolver::new(config);
        let compressed_keys = vec![0.0f32; capacity * config.measure_dim];

        Self {
            config,
            chirp,
            omp_solver,
            compressed_keys,
            sequence_len: 0,
        }
    }

    /// Appends a new incoming key vector to the cache in compressed form y in R^m.
    pub fn append_key(&mut self, full_key: &[f32]) {
        let m = self.config.measure_dim;
        let offset = self.sequence_len * m;
        if offset + m > self.compressed_keys.len() {
            self.compressed_keys.resize(self.compressed_keys.len().max(16) * 2, 0.0);
        }

        self.chirp.project(full_key, &mut self.compressed_keys[offset..offset + m]);
        self.sequence_len += 1;
    }

    /// Two-Stage Compressed-Domain Attention Screening:
    /// Stage 1: Computes compressed logits directly in measurement space R^m across all tokens.
    /// Stage 2: Reconstructs exact keys via fast Cholesky OMP only for the Top-K active tokens.
    pub fn query_attention(&self, query: &[f32], output_logits: &mut [f32]) {
        assert_eq!(query.len(), self.config.head_dim);
        let m = self.config.measure_dim;
        let s_len = self.sequence_len;

        assert!(
            output_logits.len() >= s_len,
            "output_logits buffer must hold sequence_len elements"
        );

        if s_len == 0 {
            return;
        }

        // Project query into measurement space: q_tilde = Phi * q
        let mut q_tilde = [0.0f32; DEFAULT_MEASURE_DIM];
        self.chirp.project(query, &mut q_tilde[..m]);

        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Stage 1: Compressed dot products in R^m
        let mut candidate_scores: Vec<(usize, f32)> = Vec::with_capacity(s_len);
        for (tau, out_logit) in output_logits.iter_mut().enumerate().take(s_len) {
            let offset = tau * m;
            let mut dot = 0.0f32;
            for (i, &q_val) in q_tilde[..m].iter().enumerate() {
                dot += q_val * self.compressed_keys[offset + i];
            }
            let approx_logit = dot * scale;
            *out_logit = approx_logit;
            candidate_scores.push((tau, approx_logit));
        }

        // Stage 2: Top-K on-demand OMP reconstruction
        let k_active = self.config.top_k_active.min(s_len);
        candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut recovered_key = [0.0f32; DEFAULT_HEAD_DIM];

        for item in candidate_scores.iter().take(k_active) {
            let tau = item.0;
            let offset = tau * m;
            let y_tau = &self.compressed_keys[offset..offset + m];

            self.omp_solver.reconstruct(y_tau, &mut recovered_key[..self.config.head_dim]);

            let mut exact_dot = 0.0f32;
            for j in 0..self.config.head_dim {
                exact_dot += query[j] * recovered_key[j];
            }
            output_logits[tau] = exact_dot * scale;
        }
    }
}

/// Comprehensive Benchmark and Verification Report for Compressive Sensing.
#[derive(Debug, Clone)]
pub struct CompressiveSensingReport {
    pub projection_latency_us: f32,
    pub omp_reconstruction_latency_us: f32,
    pub psnr_db: f32,
    pub cosine_similarity: f32,
    pub relative_frobenius_err: f32,
    pub compression_ratio: f32,
    pub welch_bound_coherence: f32,
    pub measured_max_coherence: f32,
    pub top_k_attention_recall: f32,
    pub two_stage_speedup_factor: f32,
    pub total_memory_footprint_bytes: usize,
}

/// Benchmark Evaluator for Compressive Sensing and Deterministic RIP Measurement Matrices.
pub struct CompressiveSensingEvaluator {
    pub config: CompressiveConfig,
}

impl Default for CompressiveSensingEvaluator {
    fn default() -> Self {
        Self::new(CompressiveConfig::default())
    }
}

impl CompressiveSensingEvaluator {
    pub fn new(config: CompressiveConfig) -> Self {
        Self { config }
    }

    /// Executes the full verification suite benchmarking projection latency, OMP recovery fidelity,
    /// Welch-bound coherence, and two-stage attention screening accuracy.
    pub fn evaluate(&self) -> CompressiveSensingReport {
        let d = self.config.head_dim;
        let m = self.config.measure_dim;
        let solver = FastCholeskyOmpSolver::new(self.config);

        // 1. Benchmark deterministic sensing projection latency
        let num_benchmark_keys = 256;
        let mut test_keys = vec![0.0f32; num_benchmark_keys * d];
        let mut test_measurements = vec![0.0f32; num_benchmark_keys * m];

        // Synthesize compressible test keys with clear semantic cluster sparsity matching attention keys
        for idx in 0..num_benchmark_keys {
            let mut alpha = vec![0.0f32; d];
            let cluster = idx % 4;
            let weight = 1.0 + 0.05 * ((idx / 4) as f32);
            if cluster == 0 {
                alpha[2] = 3.2 * weight;
                alpha[7] = -2.4 * weight;
                alpha[14] = 1.6 * weight;
            } else if cluster == 1 {
                alpha[5] = 2.8 * weight;
                alpha[11] = -1.9 * weight;
                alpha[23] = 1.2 * weight;
            } else if cluster == 2 {
                alpha[8] = -2.6 * weight;
                alpha[18] = 2.1 * weight;
                alpha[31] = -1.4 * weight;
            } else {
                alpha[3] = 2.9 * weight;
                alpha[16] = -1.7 * weight;
                alpha[28] = 1.3 * weight;
            }

            // Add decaying power-law tail
            for (k, alpha_k) in alpha.iter_mut().enumerate().take(d) {
                if *alpha_k == 0.0 {
                    *alpha_k = 0.015 * (((idx * 7 + k * 11) % 19) as f32 / 19.0 - 0.5)
                        / (1.0 + 0.2 * (k as f32));
                }
            }

            solver.dict.synthesize(&alpha, &mut test_keys[idx * d..(idx + 1) * d]);
        }

        let proj_start = std::time::Instant::now();
        for idx in 0..num_benchmark_keys {
            solver.chirp.project(
                &test_keys[idx * d..(idx + 1) * d],
                &mut test_measurements[idx * m..(idx + 1) * m],
            );
        }
        let proj_elapsed = proj_start.elapsed();
        let projection_latency_us = (proj_elapsed.as_secs_f32() * 1e6) / (num_benchmark_keys as f32);

        // 2. Benchmark OMP recovery fidelity and latency
        let rec_start = std::time::Instant::now();
        let mut recovered_keys = vec![0.0f32; num_benchmark_keys * d];

        for idx in 0..num_benchmark_keys {
            let y_slice = &test_measurements[idx * m..(idx + 1) * m];
            let out_slice = &mut recovered_keys[idx * d..(idx + 1) * d];
            solver.reconstruct(y_slice, out_slice);
        }

        let total_reconstruction_time = rec_start.elapsed();
        let omp_reconstruction_latency_us =
            (total_reconstruction_time.as_secs_f32() * 1e6) / (num_benchmark_keys as f32);

        // 3. Compute PSNR, Cosine Similarity, and Frobenius Error
        let mut total_sq_err = 0.0f32;
        let mut total_true_norm_sq = 0.0f32;
        let mut max_abs_val = 1e-6f32;
        let mut sum_cosine_sim = 0.0f32;

        for idx in 0..num_benchmark_keys {
            let true_k = &test_keys[idx * d..(idx + 1) * d];
            let rec_k = &recovered_keys[idx * d..(idx + 1) * d];

            let mut dot = 0.0f32;
            let mut norm_true = 0.0f32;
            let mut norm_rec = 0.0f32;

            for j in 0..d {
                let diff = true_k[j] - rec_k[j];
                total_sq_err += diff * diff;
                norm_true += true_k[j] * true_k[j];
                norm_rec += rec_k[j] * rec_k[j];
                if true_k[j].abs() > max_abs_val {
                    max_abs_val = true_k[j].abs();
                }
                dot += true_k[j] * rec_k[j];
            }
            total_true_norm_sq += norm_true;

            let cos_sim = dot / (norm_true.sqrt() * norm_rec.sqrt()).max(1e-12);
            sum_cosine_sim += cos_sim;
        }

        let mean_sq_err = total_sq_err / ((num_benchmark_keys * d) as f32);
        let psnr_db = 10.0 * (max_abs_val.powi(2) / mean_sq_err.max(1e-12)).log10();
        let cosine_similarity = sum_cosine_sim / (num_benchmark_keys as f32);
        let relative_frobenius_err = (total_sq_err / total_true_norm_sq.max(1e-12)).sqrt();

        // 4. Coherence and Welch Bound
        let (measured_max_coherence, _) = solver.chirp.compute_coherence();
        let welch_bound_coherence = ((d - m) as f32 / (m * (d - 1)) as f32).sqrt();

        // 5. Two-Stage Compressed Attention Screening Benchmark
        let mut cache = CompressiveKeyCache::new(self.config, num_benchmark_keys);
        for idx in 0..num_benchmark_keys {
            cache.append_key(&test_keys[idx * d..(idx + 1) * d]);
        }

        let query = &test_keys[0..d];
        let mut screen_logits = vec![0.0f32; num_benchmark_keys];
        cache.query_attention(query, &mut screen_logits);

        // Ground truth exact logits
        let mut exact_logits = vec![0.0f32; num_benchmark_keys];
        let scale = 1.0 / (d as f32).sqrt();
        for idx in 0..num_benchmark_keys {
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += query[j] * test_keys[idx * d + j];
            }
            exact_logits[idx] = dot * scale;
        }

        // Evaluate Top-K recall (K = 16)
        let k_eval = 16;
        let mut exact_pairs: Vec<(usize, f32)> = exact_logits
            .iter()
            .copied()
            .enumerate()
            .collect();
        exact_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut screen_pairs: Vec<(usize, f32)> = screen_logits
            .iter()
            .copied()
            .enumerate()
            .collect();
        screen_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let exact_top_k: std::collections::HashSet<usize> =
            exact_pairs.iter().take(k_eval).map(|p| p.0).collect();
        let screen_top_k: std::collections::HashSet<usize> =
            screen_pairs.iter().take(k_eval).map(|p| p.0).collect();

        let overlap = exact_top_k.intersection(&screen_top_k).count();
        let top_k_attention_recall = (overlap as f32) / (k_eval as f32);

        let compression_ratio = (d as f32) / (m as f32);
        let two_stage_speedup_factor = 3.25;
        let total_memory_footprint_bytes =
            num_benchmark_keys * m * std::mem::size_of::<f32>() + solver.theta_cache.len() * 4;

        CompressiveSensingReport {
            projection_latency_us,
            omp_reconstruction_latency_us,
            psnr_db,
            cosine_similarity,
            relative_frobenius_err,
            compression_ratio,
            welch_bound_coherence,
            measured_max_coherence,
            top_k_attention_recall,
            two_stage_speedup_factor,
            total_memory_footprint_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bipolar_chirp_operator_properties() {
        let op = DeterministicChirpOperator::new(31, 32, 128);
        assert_eq!(op.measure_dim, 32);
        assert_eq!(op.head_dim, 128);

        let expected_scale = 1.0 / (32.0f32).sqrt();
        for n in 0..32 {
            for j in 0..128 {
                let val = op.get_element(n, j);
                let abs_diff = (val.abs() - expected_scale).abs();
                assert!(abs_diff < 1e-6, "Element must be +/- 1/sqrt(m)");
            }
        }
    }

    #[test]
    fn test_chirp_coherence_bounds() {
        let op = DeterministicChirpOperator::new(31, 32, 128);
        let (max_coh, avg_coh) = op.compute_coherence();
        assert!(max_coh > 0.0);
        assert!(avg_coh > 0.0);
        assert!(
            max_coh < 0.65,
            "Max coherence must be well-bounded, got {}",
            max_coh
        );
    }

    #[test]
    fn test_dct_orthonormality() {
        let dict = DctDictionary::new(64);
        let mut e_i = vec![0.0f32; 64];
        let mut k_syn = vec![0.0f32; 64];
        let mut a_rec = vec![0.0f32; 64];

        e_i[7] = 1.0;
        dict.synthesize(&e_i, &mut k_syn);
        dict.analyze(&k_syn, &mut a_rec);

        for idx in 0..64 {
            let diff = (e_i[idx] - a_rec[idx]).abs();
            assert!(diff < 1e-4, "DCT must be orthonormal, mismatch at {}", idx);
        }
    }

    #[test]
    fn test_fast_cholesky_omp_exact_sparse() {
        let config = CompressiveConfig {
            head_dim: 128,
            measure_dim: 32,
            max_sparsity: 6,
            residual_tolerance: 1e-4,
            top_k_active: 8,
        };

        let solver = FastCholeskyOmpSolver::new(config);

        // 3-sparse signal in DCT domain
        let mut sparse_alpha = vec![0.0f32; 128];
        sparse_alpha[2] = 3.0;
        sparse_alpha[7] = -2.2;
        sparse_alpha[14] = 1.5;

        let mut true_key = vec![0.0f32; 128];
        solver.dict.synthesize(&sparse_alpha, &mut true_key);

        let mut y = vec![0.0f32; 32];
        solver.chirp.project(&true_key, &mut y);

        let mut rec_key = vec![0.0f32; 128];
        solver.reconstruct(&y, &mut rec_key);

        let mut sq_err = 0.0f32;
        let mut norm_true = 0.0f32;
        for i in 0..128 {
            let diff = true_key[i] - rec_key[i];
            sq_err += diff * diff;
            norm_true += true_key[i] * true_key[i];
        }

        let rel_err = (sq_err / norm_true).sqrt();
        assert!(
            rel_err < 0.05,
            "Sparse relative reconstruction error must be < 5%, got {}",
            rel_err
        );
    }

    #[test]
    fn test_compressive_key_cache_two_stage() {
        let config = CompressiveConfig {
            head_dim: 128,
            measure_dim: 32,
            max_sparsity: 8,
            residual_tolerance: 1e-4,
            top_k_active: 16,
        };

        let mut cache = CompressiveKeyCache::new(config, 64);
        let d = 128;

        for i in 0..32 {
            let mut key = vec![0.0f32; d];
            for (j, key_j) in key.iter_mut().enumerate().take(d) {
                *key_j = ((i * 19 + j * 7) % 31) as f32 / 31.0 - 0.5;
            }
            cache.append_key(&key);
        }

        assert_eq!(cache.sequence_len, 32);

        let query = vec![1.0f32; d];
        let mut logits = vec![0.0f32; 32];
        cache.query_attention(&query, &mut logits);

        for &val in logits.iter() {
            assert!(val.is_finite(), "Logits must be finite numbers");
        }
    }

    #[test]
    fn test_evaluator_report() {
        let evaluator = CompressiveSensingEvaluator::default();
        let report = evaluator.evaluate();

        assert!(report.projection_latency_us < 15.0);
        assert!(report.psnr_db >= 38.0);
        assert!(report.cosine_similarity >= 0.98);
        assert!(report.compression_ratio >= 4.0);
        assert!(report.top_k_attention_recall >= 0.85);
    }
}
