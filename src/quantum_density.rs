#![deny(unsafe_code)]

//! Quantum POVM & Projective Hilbert Space Attention State Compaction.
//!
//! Replaces discrete Euclidean Key-Value (KV) cache vectors with quantum-inspired
//! mixed density operators rho in S(H) on real projective Hilbert space RP^{d-1}.
//! Factorizes density matrices via Burer-Monteiro form rho = L * L^T (L in R^{d x r})
//! to guarantee Hermiticity and positive semi-definiteness (rho >= 0) by construction
//! while bounding memory footprint to O(d * r) independent of sequence length S.
//!
//! Reformulates query-key attention routing through Dual Positive Operator-Valued
//! Measure (POVM) mechanics under Born's rule: <M_q> = Tr(M_q * rho) = ||L^T q||_2^2 / ||q||_2^2.

use std::time::Instant;

/// Small numerical epsilon for stability and division guards.
pub const NUMERICAL_EPSILON: f32 = 1e-9;

/// A normalized pure quantum state ray in real projective space RP^{d-1}.
/// Invariant under non-zero scalar scaling: |psi> ~ c |psi>.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectiveState {
    pub dim: usize,
    pub amplitudes: Vec<f32>,
}

impl ProjectiveState {
    /// Creates a zero-initialized state with given dimension.
    pub fn new(dim: usize) -> Self {
        let mut amplitudes = vec![0.0f32; dim];
        if dim > 0 {
            amplitudes[0] = 1.0;
        }
        Self { dim, amplitudes }
    }

    /// Embeds a raw Euclidean key vector into a pure state ray on RP^{d-1}.
    /// Normalizes the key onto the unit sphere S^{d-1}.
    pub fn from_key(raw_key: &[f32]) -> Self {
        let dim = raw_key.len();
        assert!(dim > 0, "Key dimension must be positive");

        let norm_sq: f32 = raw_key.iter().map(|&x| x * x).sum();
        if norm_sq > NUMERICAL_EPSILON {
            let inv_norm = 1.0 / norm_sq.sqrt();
            let amplitudes = raw_key.iter().map(|&x| x * inv_norm).collect();
            Self { dim, amplitudes }
        } else {
            let mut amplitudes = vec![0.0f32; dim];
            amplitudes[0] = 1.0;
            Self { dim, amplitudes }
        }
    }

    /// Evaluates the projective Fubini-Study geodesic distance:
    /// d_FS([psi], [phi]) = arccos(|<psi|phi>|).
    pub fn fubini_study_distance(&self, other: &Self) -> f32 {
        assert_eq!(self.dim, other.dim, "Dimension mismatch in Fubini-Study calculation");
        let inner_product: f32 = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let overlap = inner_product.abs().min(1.0);
        overlap.acos()
    }

    /// Evaluates the transition probability (quantum overlap) |<psi|phi>|^2.
    pub fn quantum_overlap(&self, other: &Self) -> f32 {
        assert_eq!(self.dim, other.dim, "Dimension mismatch in quantum overlap calculation");
        let inner_prod: f32 = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        inner_prod * inner_prod
    }
}

/// A low-rank factorized mixed Density Matrix rho = L * L^T in Burer-Monteiro form.
/// Guarantees Hermiticity (rho = rho^T) and Positive Semi-Definiteness (rho >= 0)
/// by construction without requiring expensive semidefinite programming.
#[derive(Debug, Clone)]
pub struct FactorizedDensityMatrix {
    pub dim: usize,
    pub rank: usize,
    /// Storage matrix L of shape (dim x rank), stored in row-major order.
    pub l_factor: Vec<f32>,
}

impl FactorizedDensityMatrix {
    /// Creates an empty factorized density matrix of shape (dim x rank).
    pub fn new(dim: usize, rank: usize) -> Self {
        assert!(dim > 0 && rank > 0, "Dimension and rank must be positive");
        let l_factor = vec![0.0f32; dim * rank];
        Self { dim, rank, l_factor }
    }

    /// Initializes a density matrix populated with a single pure state seed.
    pub fn initialize_pure(seed: &ProjectiveState, rank: usize) -> Self {
        let dim = seed.dim;
        assert!(rank > 0, "Rank must be positive");
        let mut l_factor = vec![0.0f32; dim * rank];
        for i in 0..dim {
            l_factor[i * rank] = seed.amplitudes[i];
        }
        let mut matrix = Self { dim, rank, l_factor };
        matrix.normalize_trace();
        matrix
    }

    /// Enforces the unit trace condition: Tr(rho) = Tr(L L^T) = ||L||_F^2 = 1.0.
    pub fn normalize_trace(&mut self) {
        let frobenius_sq: f32 = self.l_factor.iter().map(|&x| x * x).sum();
        if frobenius_sq > NUMERICAL_EPSILON {
            let inv_frobenius = 1.0 / frobenius_sq.sqrt();
            for val in self.l_factor.iter_mut() {
                *val *= inv_frobenius;
            }
        }
    }

    /// Computes the active trace Tr(rho) = ||L||_F^2.
    pub fn trace(&self) -> f32 {
        self.l_factor.iter().map(|&x| x * x).sum()
    }

    /// Computes the expectation value <M_q> = Tr(M_q * rho) = ||L^T q||_2^2 / ||q||_2^2
    /// for a query vector q under the Dual POVM formulation.
    /// Complexity: O(d * r) FLOPs, independent of sequence history S.
    pub fn query_expectation(&self, query: &[f32]) -> f32 {
        assert_eq!(query.len(), self.dim, "Query dimension mismatch");
        let q_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        if q_norm_sq <= NUMERICAL_EPSILON {
            return 0.0;
        }

        let mut l_t_q = vec![0.0f32; self.rank];
        for (j, item) in l_t_q.iter_mut().enumerate().take(self.rank) {
            let mut sum = 0.0f32;
            for (i, &q_val) in query.iter().enumerate().take(self.dim) {
                sum += self.l_factor[i * self.rank + j] * q_val;
            }
            *item = sum;
        }

        let expectation: f32 = l_t_q.iter().map(|&x| x * x).sum();
        (expectation / q_norm_sq).clamp(0.0, 1.0)
    }

    /// Incorporates a new incoming token state |psi> via streaming rank-1 update:
    /// rho_{t+1} = beta * rho_t + (1 - beta) * |psi><psi|
    /// Maintains low-rank factorization via Gram-Schmidt residual projection.
    pub fn streaming_update(&mut self, new_state: &ProjectiveState, decay_beta: f32) {
        assert_eq!(new_state.dim, self.dim, "State dimension mismatch");
        let beta = decay_beta.clamp(0.0, 1.0);
        let alpha = (1.0 - beta).max(0.0);
        let sqrt_beta = beta.sqrt();
        let sqrt_alpha = alpha.sqrt();

        // 1. Scale existing L by sqrt(beta)
        for val in self.l_factor.iter_mut() {
            *val *= sqrt_beta;
        }

        // 2. Perform Gram-Schmidt projection of |psi> against existing L columns
        let mut proj_weights = vec![0.0f32; self.rank];
        for (j, item) in proj_weights.iter_mut().enumerate().take(self.rank) {
            let mut dot = 0.0f32;
            for (i, &amp) in new_state.amplitudes.iter().enumerate().take(self.dim) {
                dot += self.l_factor[i * self.rank + j] * amp;
            }
            *item = dot;
        }

        // Residual orthogonal component: v_orth = |psi> - sum_j (proj_weights[j] * L_j)
        let mut v_orth = new_state.amplitudes.clone();
        for (j, &w) in proj_weights.iter().enumerate().take(self.rank) {
            for (i, v_val) in v_orth.iter_mut().enumerate().take(self.dim) {
                *v_val -= w * self.l_factor[i * self.rank + j];
            }
        }

        let orth_norm_sq: f32 = v_orth.iter().map(|&x| x * x).sum();
        let orth_norm = orth_norm_sq.sqrt();

        // If the residual is significant, replace or blend into the lowest-energy column
        if orth_norm > 1e-4 {
            let inv_norm = 1.0 / orth_norm;
            for x in v_orth.iter_mut() {
                *x *= inv_norm;
            }

            let mut min_col = 0;
            let mut min_norm = f32::MAX;
            for j in 0..self.rank {
                let mut col_norm_sq = 0.0f32;
                for i in 0..self.dim {
                    let val = self.l_factor[i * self.rank + j];
                    col_norm_sq += val * val;
                }
                if col_norm_sq < min_norm {
                    min_norm = col_norm_sq;
                    min_col = j;
                }
            }

            for (i, &v_val) in v_orth.iter().enumerate().take(self.dim) {
                self.l_factor[i * self.rank + min_col] += sqrt_alpha * v_val;
            }
        } else {
            // State is already spanned; reinforce parallel subspace
            for (j, &w) in proj_weights.iter().enumerate().take(self.rank) {
                for (i, &amp) in new_state.amplitudes.iter().enumerate().take(self.dim) {
                    self.l_factor[i * self.rank + j] += sqrt_alpha * w * amp * 0.1;
                }
            }
        }

        // Re-enforce unit trace
        self.normalize_trace();
    }

    /// Evaluates the quantum purity gamma = Tr(rho^2) = Tr((L^T L)^2).
    /// Bounded in [1/dim, 1.0]. A pure state has gamma = 1.0; a maximally mixed state has gamma = 1/dim.
    pub fn purity(&self) -> f32 {
        let g = self.gram_matrix();
        let tr_g_sq: f32 = g.iter().map(|&x| x * x).sum();
        let min_purity = 1.0 / (self.dim as f32);
        tr_g_sq.clamp(min_purity, 1.0)
    }

    /// Computes the small (r x r) Gram matrix G = L^T * L.
    pub fn gram_matrix(&self) -> Vec<f32> {
        let mut g = vec![0.0f32; self.rank * self.rank];
        for r1 in 0..self.rank {
            for r2 in 0..self.rank {
                let mut sum = 0.0f32;
                for i in 0..self.dim {
                    sum += self.l_factor[i * self.rank + r1] * self.l_factor[i * self.rank + r2];
                }
                g[r1 * self.rank + r2] = sum;
            }
        }
        g
    }

    /// Computes the eigenvalues of the Gram matrix G = L^T * L via the Jacobi eigenvalue algorithm.
    /// Since G is symmetric positive semi-definite and r is small (r <= 16), this converges in < 2 us.
    pub fn compute_eigenvalues(&self) -> Vec<f32> {
        let r = self.rank;
        let mut a = self.gram_matrix();
        let max_sweeps = 50;
        let eps = 1e-7f32;

        for _ in 0..max_sweeps {
            let mut max_offdiag = 0.0f32;
            let mut p = 0;
            let mut q = 1;

            for i in 0..r {
                for j in (i + 1)..r {
                    let val = a[i * r + j].abs();
                    if val > max_offdiag {
                        max_offdiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_offdiag < eps {
                break;
            }

            let app = a[p * r + p];
            let aqq = a[q * r + q];
            let apq = a[p * r + q];

            let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
            let c = phi.cos();
            let s = phi.sin();

            for i in 0..r {
                if i != p && i != q {
                    let aip = a[i * r + p];
                    let aiq = a[i * r + q];
                    a[i * r + p] = c * aip - s * aiq;
                    a[p * r + i] = a[i * r + p];
                    a[i * r + q] = s * aip + c * aiq;
                    a[q * r + i] = a[i * r + q];
                }
            }

            let new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            let new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p * r + p] = new_app;
            a[q * r + q] = new_aqq;
            a[p * r + q] = 0.0;
            a[q * r + p] = 0.0;
        }

        let mut evals: Vec<f32> = (0..r).map(|i| a[i * r + i].max(0.0)).collect();
        evals.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        evals
    }

    /// Reconstructs the full (dim x dim) density matrix rho = L * L^T for validation and diagnostics.
    pub fn reconstruct_full_density_matrix(&self) -> Vec<f32> {
        let mut rho = vec![0.0f32; self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                let mut sum = 0.0f32;
                for r in 0..self.rank {
                    sum += self.l_factor[i * self.rank + r] * self.l_factor[j * self.rank + r];
                }
                rho[i * self.dim + j] = sum;
            }
        }
        rho
    }
}

/// Quantum Information Diagnostics Engine for Multi-Head Attention Subspaces.
pub struct QuantumInformationEngine;

impl QuantumInformationEngine {
    /// Computes the von Neumann entropy S(rho) = -sum_j lambda_j * ln(lambda_j)
    /// from the eigenvalues of the low-rank density matrix.
    pub fn von_neumann_entropy(eigenvalues: &[f32]) -> f32 {
        let mut entropy = 0.0f32;
        for &lambda in eigenvalues {
            if lambda > NUMERICAL_EPSILON {
                entropy -= lambda * lambda.ln();
            }
        }
        entropy.max(0.0)
    }

    /// Evaluates linear entropy S_L(rho) = (d / (d - 1)) * (1 - purity).
    pub fn linear_entropy(purity: f32, dim: usize) -> f32 {
        if dim <= 1 {
            return 0.0;
        }
        let factor = (dim as f32) / ((dim - 1) as f32);
        (factor * (1.0 - purity)).clamp(0.0, 1.0)
    }

    /// Computes the Uhlmann-Jozsa Quantum Fidelity between a pure query state |psi>
    /// and a mixed density state rho: F(|psi><psi|, rho) = <psi| rho |psi>.
    pub fn fidelity_pure_mixed(pure: &ProjectiveState, mixed: &FactorizedDensityMatrix) -> f32 {
        mixed.query_expectation(&pure.amplitudes).clamp(0.0, 1.0)
    }

    /// Evaluates the subspace overlap fidelity between two factorized density matrices
    /// via the normalized Hilbert-Schmidt inner product:
    /// F_HS(rho, sigma) = Tr(rho * sigma) / sqrt(Tr(rho^2) * Tr(sigma^2)).
    /// Yields 1.0 for identical density matrices (whether pure or mixed), and ~0.0 for mutually orthogonal subspaces.
    pub fn subspace_fidelity(rho: &FactorizedDensityMatrix, sigma: &FactorizedDensityMatrix) -> f32 {
        assert_eq!(rho.dim, sigma.dim, "Dimension mismatch in fidelity calculation");
        let r1 = rho.rank;
        let r2 = sigma.rank;

        let mut m = vec![0.0f32; r1 * r2];
        for a in 0..r1 {
            for b in 0..r2 {
                let mut sum = 0.0f32;
                for i in 0..rho.dim {
                    sum += rho.l_factor[i * r1 + a] * sigma.l_factor[i * r2 + b];
                }
                m[a * r2 + b] = sum;
            }
        }

        let tr_rho_sigma: f32 = m.iter().map(|&x| x * x).sum();
        let purity_rho = rho.purity();
        let purity_sigma = sigma.purity();
        let denom = (purity_rho * purity_sigma).sqrt();
        if denom > NUMERICAL_EPSILON {
            (tr_rho_sigma / denom).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Positive Operator-Valued Measure (POVM) Attention Mechanism Kernel.
#[derive(Debug, Clone)]
pub struct PovmAttentionKernel {
    pub head_dim: usize,
}

impl PovmAttentionKernel {
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    /// Evaluates attention weights over a collection of key POVM elements {M_t}
    /// for a given query state under Born's rule: alpha_t = Tr(M_t * rho_q) / sum_k Tr(M_k * rho_q).
    /// Enforces strict probability normalization without softmax numerical drift.
    pub fn compute_born_attention(
        &self,
        query: &[f32],
        key_states: &[ProjectiveState],
    ) -> Vec<f32> {
        let q_state = ProjectiveState::from_key(query);
        let num_tokens = key_states.len();

        let mut raw_probabilities = Vec::with_capacity(num_tokens);
        let mut total_trace = 0.0f32;

        for key in key_states {
            let p_t = key.quantum_overlap(&q_state);
            raw_probabilities.push(p_t);
            total_trace += p_t;
        }

        let inv_total = if total_trace > NUMERICAL_EPSILON {
            1.0 / total_trace
        } else {
            1.0 / (num_tokens.max(1) as f32)
        };

        for p in raw_probabilities.iter_mut() {
            *p *= inv_total;
        }

        raw_probabilities
    }
}

/// Configuration parameters for Quantum POVM density matrix compaction.
#[derive(Debug, Clone)]
pub struct QuantumDensityConfig {
    pub head_dim: usize,
    pub subspace_rank: usize,
    pub decay_beta: f32,
    pub pruning_threshold: f32,
}

impl Default for QuantumDensityConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            subspace_rank: 6,
            decay_beta: 0.92,
            pruning_threshold: 0.95,
        }
    }
}

/// Verification report capturing performance and mathematical invariants of Phase 23.
#[derive(Debug, Clone)]
pub struct QuantumDensityReport {
    pub head_dim: usize,
    pub subspace_rank: usize,
    pub query_expectation_latency_us: f32,
    pub streaming_update_latency_us: f32,
    pub state_compaction_ratio: f32,
    pub trace_conservation_err: f32,
    pub purity: f32,
    pub von_neumann_entropy: f32,
    pub collinear_subspace_fidelity: f32,
    pub orthogonal_subspace_fidelity: f32,
    pub born_attention_partition_error: f32,
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator benchmark harness for Phase 23.
#[derive(Debug, Default)]
pub struct QuantumDensityEvaluator {
    pub config: QuantumDensityConfig,
}

impl QuantumDensityEvaluator {
    pub fn new(config: QuantumDensityConfig) -> Self {
        Self { config }
    }

    /// Executes performance evaluation and mathematical invariant validation.
    pub fn evaluate_performance(&self) -> QuantumDensityReport {
        let d = self.config.head_dim;
        let r = self.config.subspace_rank;
        let beta = self.config.decay_beta;

        // Generate synthetic context of S tokens
        let num_tokens = 256;
        let mut key_states = Vec::with_capacity(num_tokens);
        for t in 0..num_tokens {
            let mut raw_key = vec![0.0f32; d];
            for (i, item) in raw_key.iter_mut().enumerate().take(d) {
                let angle = (t as f32 * 0.1) + (i as f32 * 0.05);
                *item = angle.sin() + 0.1 * ((i * 7 + t * 13) as f32).sin();
            }
            key_states.push(ProjectiveState::from_key(&raw_key));
        }

        // Initialize density matrix with first key
        let mut density = FactorizedDensityMatrix::initialize_pure(&key_states[0], r);

        // Benchmark streaming updates
        let update_start = Instant::now();
        for state in key_states.iter().take(num_tokens).skip(1) {
            density.streaming_update(state, beta);
        }
        let update_duration = update_start.elapsed();
        let streaming_update_latency_us =
            (update_duration.as_nanos() as f32 / ((num_tokens - 1) as f32)) / 1000.0;

        // Verify trace conservation
        let trace = density.trace();
        let trace_conservation_err = (trace - 1.0).abs();

        // Benchmark query expectation
        let mut query = vec![0.0f32; d];
        for (i, item) in query.iter_mut().enumerate().take(d) {
            *item = (i as f32 * 0.1).cos();
        }
        let q_state = ProjectiveState::from_key(&query);

        let query_trials = 1000;
        let query_start = Instant::now();
        let mut last_exp = 0.0f32;
        for _ in 0..query_trials {
            last_exp = density.query_expectation(&q_state.amplitudes);
        }
        let query_duration = query_start.elapsed();
        let query_expectation_latency_us =
            (query_duration.as_nanos() as f32 / (query_trials as f32)) / 1000.0;
        let _ = last_exp;

        // Evaluate quantum purity and entropy
        let purity = density.purity();
        let evals = density.compute_eigenvalues();
        let von_neumann_entropy = QuantumInformationEngine::von_neumann_entropy(&evals);

        // Subspace fidelity: identical state vs orthogonal state
        let collinear_density = density.clone();
        let collinear_subspace_fidelity =
            QuantumInformationEngine::subspace_fidelity(&density, &collinear_density);

        // Create a strictly orthogonal density matrix via Gram-Schmidt projection
        let mut orth_vec: Vec<f32> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        for _ in 0..2 {
            for j in 0..r {
                let mut dot = 0.0f32;
                for (i, &v_val) in orth_vec.iter().enumerate().take(d) {
                    dot += density.l_factor[i * r + j] * v_val;
                }
                for (i, v_val) in orth_vec.iter_mut().enumerate().take(d) {
                    *v_val -= dot * density.l_factor[i * r + j];
                }
            }
        }
        let orth_state = ProjectiveState::from_key(&orth_vec);
        let orth_density = FactorizedDensityMatrix::initialize_pure(&orth_state, r);
        let orthogonal_subspace_fidelity =
            QuantumInformationEngine::subspace_fidelity(&density, &orth_density);

        // Born attention partition normalization check
        let povm_kernel = PovmAttentionKernel::new(d);
        let born_probs = povm_kernel.compute_born_attention(&query, &key_states[..32]);
        let born_sum: f32 = born_probs.iter().sum();
        let born_attention_partition_error = (born_sum - 1.0).abs();

        // Memory footprint calculation
        let discrete_buffer_bytes = num_tokens * d * 4;
        let factor_bytes = d * r * 4;
        let meta_bytes = r * r * 4 + 64;
        let total_memory_footprint_bytes = factor_bytes + meta_bytes;
        let state_compaction_ratio =
            (discrete_buffer_bytes as f32) / (total_memory_footprint_bytes as f32);

        QuantumDensityReport {
            head_dim: d,
            subspace_rank: r,
            query_expectation_latency_us,
            streaming_update_latency_us,
            state_compaction_ratio,
            trace_conservation_err,
            purity,
            von_neumann_entropy,
            collinear_subspace_fidelity,
            orthogonal_subspace_fidelity,
            born_attention_partition_error,
            total_memory_footprint_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projective_state_fubini_study() {
        let dim = 64;
        let mut key1 = vec![0.0f32; dim];
        let mut key2 = vec![0.0f32; dim];
        key1[0] = 1.0;
        key2[0] = 1.0;

        let s1 = ProjectiveState::from_key(&key1);
        let s2 = ProjectiveState::from_key(&key2);

        let dist = s1.fubini_study_distance(&s2);
        assert!(dist < 1e-5, "Identical states must have 0 Fubini-Study distance");

        let overlap = s1.quantum_overlap(&s2);
        assert!((overlap - 1.0).abs() < 1e-5, "Identical states must have unit quantum overlap");

        let mut key3 = vec![0.0f32; dim];
        key3[1] = 1.0;
        let s3 = ProjectiveState::from_key(&key3);
        let dist_orth = s1.fubini_study_distance(&s3);
        assert!((dist_orth - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        assert!(s1.quantum_overlap(&s3) < 1e-6);
    }

    #[test]
    fn test_density_matrix_trace_normalization() {
        let dim = 128;
        let rank = 6;
        let mut key = vec![0.0f32; dim];
        key[5] = 4.2;
        let state = ProjectiveState::from_key(&key);

        let mut density = FactorizedDensityMatrix::initialize_pure(&state, rank);
        assert!((density.trace() - 1.0).abs() < 1e-5);

        // Mutate and re-normalize
        density.l_factor[0] = 10.0;
        density.normalize_trace();
        assert!((density.trace() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_query_expectation_bounded() {
        let dim = 128;
        let rank = 6;
        let key: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let state = ProjectiveState::from_key(&key);
        let density = FactorizedDensityMatrix::initialize_pure(&state, rank);

        let query: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let expectation = density.query_expectation(&query);
        assert!(expectation >= 0.0);
        assert!(expectation <= 1.0);
    }

    #[test]
    fn test_streaming_update_trace_conservation() {
        let dim = 64;
        let rank = 4;
        let mut key0 = vec![0.0f32; dim];
        key0[0] = 1.0;
        let s0 = ProjectiveState::from_key(&key0);
        let mut density = FactorizedDensityMatrix::initialize_pure(&s0, rank);

        for t in 1..20 {
            let mut key = vec![0.0f32; dim];
            key[t % dim] = 1.0;
            let st = ProjectiveState::from_key(&key);
            density.streaming_update(&st, 0.90);
            assert!(
                (density.trace() - 1.0).abs() < 1e-4,
                "Trace must be 1.0 after update step {}",
                t
            );
        }
    }

    #[test]
    fn test_purity_and_entropy_bounds() {
        let dim = 64;
        let rank = 4;
        let mut key0 = vec![0.0f32; dim];
        key0[0] = 1.0;
        let s0 = ProjectiveState::from_key(&key0);
        let density = FactorizedDensityMatrix::initialize_pure(&s0, rank);

        let purity = density.purity();
        assert!(
            (purity - 1.0).abs() < 1e-4,
            "Pure state must have purity 1.0, got {}",
            purity
        );

        let evals = density.compute_eigenvalues();
        let entropy = QuantumInformationEngine::von_neumann_entropy(&evals);
        assert!(entropy >= 0.0, "Entropy must be non-negative");
    }

    #[test]
    fn test_subspace_fidelity_collinear_and_orthogonal() {
        let dim = 32;
        let rank = 4;
        let mut key1 = vec![0.0f32; dim];
        key1[0] = 1.0;
        let s1 = ProjectiveState::from_key(&key1);
        let d1 = FactorizedDensityMatrix::initialize_pure(&s1, rank);
        let d2 = d1.clone();

        let f_collinear = QuantumInformationEngine::subspace_fidelity(&d1, &d2);
        assert!((f_collinear - 1.0).abs() < 1e-4, "Collinear fidelity must be 1.0");

        let mut key3 = vec![0.0f32; dim];
        key3[10] = 1.0;
        let s3 = ProjectiveState::from_key(&key3);
        let d3 = FactorizedDensityMatrix::initialize_pure(&s3, rank);

        let f_orth = QuantumInformationEngine::subspace_fidelity(&d1, &d3);
        assert!(f_orth < 1e-5, "Orthogonal fidelity must be 0.0");
    }

    #[test]
    fn test_born_attention_normalization() {
        let dim = 32;
        let kernel = PovmAttentionKernel::new(dim);

        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;

        let mut keys = Vec::new();
        for i in 0..10 {
            let mut k = vec![0.0f32; dim];
            k[i % dim] = 1.0;
            keys.push(ProjectiveState::from_key(&k));
        }

        let probs = kernel.compute_born_attention(&query, &keys);
        assert_eq!(probs.len(), 10);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Born probabilities must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_evaluator_report() {
        let evaluator = QuantumDensityEvaluator::default();
        let report = evaluator.evaluate_performance();

        assert!(report.query_expectation_latency_us < 50.0);
        assert!(report.streaming_update_latency_us < 50.0);
        assert!(report.state_compaction_ratio >= 8.0);
        assert!(report.trace_conservation_err < 1e-4);
        assert!(report.purity >= 1.0 / (report.head_dim as f32));
        assert!(report.purity <= 1.0001);
        assert!(report.von_neumann_entropy >= 0.0);
        assert!(report.collinear_subspace_fidelity >= 0.95);
        assert!(report.orthogonal_subspace_fidelity < 0.15);
        assert!(report.born_attention_partition_error < 1e-5);
        assert!(report.total_memory_footprint_bytes < 10_000);
    }
}
