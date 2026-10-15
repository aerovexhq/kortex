#![deny(unsafe_code)]

//! Graph Neural Graphons, Continuous Non-Parametric Limits, and Spectral Graph
//! Limits of Attention Key Graphs.
//!
//! Replaces discrete attention matrices with continuous limit kernels
//! W: [0, 1]^2 -> [0, 1], decoupling attention memory and sequence routing from
//! discrete sequence length S. Implements Frieze-Kannan weak regularity step-graphon
//! partitioning, spectral Hilbert-Schmidt operator decomposition, and continuous
//! cross-attention interpolation across variable sequence lengths.

use std::time::Instant;

/// Small numerical epsilon for stability.
pub const GRAPHON_EPSILON: f32 = 1e-9;

/// A continuous graphon limit operator W: [0, 1]^2 -> [0, 1] represented on an
/// N x N discretization grid with bilinear interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousGraphon {
    pub resolution: usize,
    /// Grid values W(u_i, v_j) of shape (resolution x resolution), row-major.
    pub kernel_values: Vec<f32>,
}

impl ContinuousGraphon {
    /// Creates a zero-initialized continuous graphon on an N x N grid.
    pub fn new(resolution: usize) -> Self {
        assert!(resolution > 1, "Resolution must be greater than 1");
        let kernel_values = vec![0.0f32; resolution * resolution];
        Self {
            resolution,
            kernel_values,
        }
    }

    /// Embeds a discrete attention matrix A in [0, 1]^{S x S} into a continuous
    /// step graphon on an N x N resolution grid.
    pub fn from_attention_matrix(matrix: &[f32], seq_len: usize, resolution: usize) -> Self {
        assert_eq!(
            matrix.len(),
            seq_len * seq_len,
            "Matrix size must match seq_len^2"
        );
        assert!(resolution > 1, "Resolution must be greater than 1");
        assert!(seq_len > 0, "Sequence length must be positive");

        let mut kernel_values = vec![0.0f32; resolution * resolution];

        for i in 0..resolution {
            let u = (i as f32 + 0.5) / (resolution as f32);
            let token_u = ((u * seq_len as f32).floor() as usize).min(seq_len - 1);

            for j in 0..resolution {
                let v = (j as f32 + 0.5) / (resolution as f32);
                let token_v = ((v * seq_len as f32).floor() as usize).min(seq_len - 1);

                // Symmetrize empirical attention graph: A_sym = (A + A^T) / 2
                let val1 = matrix[token_u * seq_len + token_v];
                let val2 = matrix[token_v * seq_len + token_u];
                let sym_val = (0.5 * (val1 + val2)).clamp(0.0, 1.0);

                kernel_values[i * resolution + j] = sym_val;
            }
        }

        Self {
            resolution,
            kernel_values,
        }
    }

    /// Evaluates the continuous graphon W(u, v) at coordinates (u, v) in [0, 1]^2
    /// via bilinear interpolation.
    pub fn evaluate(&self, u: f32, v: f32) -> f32 {
        let u_clamped = u.clamp(0.0, 1.0);
        let v_clamped = v.clamp(0.0, 1.0);
        let n = self.resolution;

        let x = u_clamped * ((n - 1) as f32);
        let y = v_clamped * ((n - 1) as f32);

        let i0 = (x.floor() as usize).min(n - 2);
        let j0 = (y.floor() as usize).min(n - 2);
        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let dx = x - (i0 as f32);
        let dy = y - (j0 as f32);

        let w00 = self.kernel_values[i0 * n + j0];
        let w10 = self.kernel_values[i1 * n + j0];
        let w01 = self.kernel_values[i0 * n + j1];
        let w11 = self.kernel_values[i1 * n + j1];

        let interp = (1.0 - dx) * (1.0 - dy) * w00
            + dx * (1.0 - dy) * w10
            + (1.0 - dx) * dy * w01
            + dx * dy * w11;

        interp.clamp(0.0, 1.0)
    }

    /// Computes the L2 Hilbert-Schmidt norm: ||W||_L2 = sqrt( \int_0^1 \int_0^1 W^2(u, v) du dv ).
    pub fn l2_norm(&self) -> f32 {
        let n = self.resolution;
        let sum_sq: f32 = self.kernel_values.iter().map(|&x| x * x).sum();
        (sum_sq / ((n * n) as f32)).sqrt()
    }
}

/// A piecewise-constant step-graphon approximation W_P over an M-step uniform partition
/// of [0, 1] guaranteed by the Frieze-Kannan Weak Regularity Lemma.
#[derive(Debug, Clone, PartialEq)]
pub struct StepGraphonPartition {
    pub num_steps: usize,
    /// Block densities B_{ab} of shape (num_steps x num_steps), row-major.
    pub block_weights: Vec<f32>,
}

impl StepGraphonPartition {
    /// Constructs an M-step graphon approximation from a continuous graphon.
    pub fn from_continuous(graphon: &ContinuousGraphon, num_steps: usize) -> Self {
        assert!(num_steps > 0, "Number of steps must be positive");
        let mut block_weights = vec![0.0f32; num_steps * num_steps];
        let n = graphon.resolution;

        for a in 0..num_steps {
            let u_start = (a as f32) / (num_steps as f32);
            let u_end = ((a + 1) as f32) / (num_steps as f32);

            for b in 0..num_steps {
                let v_start = (b as f32) / (num_steps as f32);
                let v_end = ((b + 1) as f32) / (num_steps as f32);

                // Integrate W(u, v) over block [u_start, u_end] x [v_start, v_end]
                let mut sum = 0.0f32;
                let mut count = 0usize;

                for i in 0..n {
                    let u = (i as f32 + 0.5) / (n as f32);
                    if u >= u_start && u < u_end {
                        for j in 0..n {
                            let v = (j as f32 + 0.5) / (n as f32);
                            if v >= v_start && v < v_end {
                                sum += graphon.kernel_values[i * n + j];
                                count += 1;
                            }
                        }
                    }
                }

                let avg = if count > 0 {
                    sum / (count as f32)
                } else {
                    graphon.evaluate(0.5 * (u_start + u_end), 0.5 * (v_start + v_end))
                };

                block_weights[a * num_steps + b] = avg.clamp(0.0, 1.0);
            }
        }

        Self {
            num_steps,
            block_weights,
        }
    }

    /// Evaluates the step graphon at (u, v) in [0, 1]^2.
    pub fn evaluate(&self, u: f32, v: f32) -> f32 {
        let m = self.num_steps;
        let a = ((u.clamp(0.0, 1.0) * (m as f32)).floor() as usize).min(m - 1);
        let b = ((v.clamp(0.0, 1.0) * (m as f32)).floor() as usize).min(m - 1);
        self.block_weights[a * m + b]
    }

    /// Approximates the Frieze-Kannan cut-metric distance:
    /// d_square(W, W_P) = sup_{S, T \subset [0, 1]} | \int_S \int_T (W(u, v) - W_P(u, v)) du dv |
    /// using 2D prefix sums over interval cuts [u1, u2] x [v1, v2].
    pub fn cut_metric_distance(&self, graphon: &ContinuousGraphon) -> f32 {
        let n = graphon.resolution;
        let mut diff = vec![0.0f32; n * n];

        for i in 0..n {
            let u = (i as f32 + 0.5) / (n as f32);
            for j in 0..n {
                let v = (j as f32 + 0.5) / (n as f32);
                let w = graphon.kernel_values[i * n + j];
                let wp = self.evaluate(u, v);
                diff[i * n + j] = w - wp;
            }
        }

        // Construct 2D prefix sum table of size (n + 1) x (n + 1)
        let mut prefix = vec![0.0f32; (n + 1) * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                prefix[(i + 1) * (n + 1) + (j + 1)] = diff[i * n + j]
                    + prefix[i * (n + 1) + (j + 1)]
                    + prefix[(i + 1) * (n + 1) + j]
                    - prefix[i * (n + 1) + j];
            }
        }

        // Sweep rectangular sub-intervals with dyadic stride
        let stride = (n / 8).max(1);
        let mut max_cut = 0.0f32;
        let inv_n_sq = 1.0 / ((n * n) as f32);

        for i1 in (0..n).step_by(stride) {
            for i2 in ((i1 + stride)..=n).step_by(stride) {
                for j1 in (0..n).step_by(stride) {
                    for j2 in ((j1 + stride)..=n).step_by(stride) {
                        let rect_sum = prefix[i2 * (n + 1) + j2]
                            - prefix[i1 * (n + 1) + j2]
                            - prefix[i2 * (n + 1) + j1]
                            + prefix[i1 * (n + 1) + j1];
                        let cut_norm = (rect_sum.abs()) * inv_n_sq;
                        if cut_norm > max_cut {
                            max_cut = cut_norm;
                        }
                    }
                }
            }
        }

        max_cut
    }
}

/// Compact low-rank spectral decomposition of the continuous graphon operator:
/// W_k(u, v) = \sum_{j=1}^k \lambda_j \phi_j(u) \phi_j(v)
#[derive(Debug, Clone)]
pub struct SpectralGraphon {
    pub rank: usize,
    pub eigenvalues: Vec<f32>,
    /// Orthonormal eigenfunctions on the grid: shape (rank x resolution).
    pub eigenfunctions: Vec<Vec<f32>>,
    pub resolution: usize,
}

impl SpectralGraphon {
    /// Decomposes the continuous graphon into rank-k spectral modes via power iteration
    /// and Hotelling deflation with Gram-Schmidt orthogonalization.
    pub fn decompose(graphon: &ContinuousGraphon, rank: usize) -> Self {
        let n = graphon.resolution;
        assert!(rank > 0 && rank <= n, "Rank must be in [1, resolution]");

        // Discretized integral operator matrix: M = (1 / n) * W_grid
        let inv_n = 1.0 / (n as f32);
        let mut a: Vec<f32> = graphon.kernel_values.iter().map(|&x| x * inv_n).collect();
        let mut unit_eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(rank);
        let mut pairs: Vec<(f32, Vec<f32>)> = Vec::with_capacity(rank);

        for m in 0..rank {
            // Diverse harmonic initialization
            let mut v: Vec<f32> = (0..n)
                .map(|i| {
                    ((m as f32 + 1.0) * std::f32::consts::PI * (i as f32 + 0.5) / (n as f32)).sin()
                })
                .collect();

            // Project out previous unit eigenvectors
            for prev in &unit_eigenvectors {
                let dot: f32 = prev.iter().zip(v.iter()).map(|(&x, &y)| x * y).sum();
                for (vi, &pi) in v.iter_mut().zip(prev.iter()) {
                    *vi -= dot * pi;
                }
            }
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > GRAPHON_EPSILON {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }

            for _ in 0..40 {
                let mut next_v = vec![0.0f32; n];
                for (i, item) in next_v.iter_mut().enumerate().take(n) {
                    let mut sum = 0.0f32;
                    for (j, &v_val) in v.iter().enumerate().take(n) {
                        sum += a[i * n + j] * v_val;
                    }
                    *item = sum;
                }

                for prev in &unit_eigenvectors {
                    let dot: f32 = prev.iter().zip(next_v.iter()).map(|(&x, &y)| x * y).sum();
                    for (nvi, &pi) in next_v.iter_mut().zip(prev.iter()) {
                        *nvi -= dot * pi;
                    }
                }

                let next_norm: f32 = next_v.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if next_norm < GRAPHON_EPSILON {
                    break;
                }
                for x in next_v.iter_mut() {
                    *x /= next_norm;
                }
                v = next_v;
            }

            // Rayleigh quotient: lambda = v^T A v
            let mut av = vec![0.0f32; n];
            for (i, item) in av.iter_mut().enumerate().take(n) {
                let mut sum = 0.0f32;
                for (j, &v_val) in v.iter().enumerate().take(n) {
                    sum += a[i * n + j] * v_val;
                }
                *item = sum;
            }
            let lambda: f32 = v.iter().zip(av.iter()).map(|(&x, &y)| x * y).sum::<f32>().max(0.0);

            // Deflation: A <- A - lambda * (v * v^T)
            for i in 0..n {
                for (j, &v_j) in v.iter().enumerate().take(n) {
                    a[i * n + j] -= lambda * v[i] * v_j;
                }
            }

            let sqrt_n = (n as f32).sqrt();
            let phi: Vec<f32> = v.iter().map(|&x| x * sqrt_n).collect();
            unit_eigenvectors.push(v);
            pairs.push((lambda, phi));
        }

        pairs.sort_by(|(l1, _), (l2, _)| l2.partial_cmp(l1).unwrap_or(std::cmp::Ordering::Equal));
        let eigenvalues: Vec<f32> = pairs.iter().map(|(l, _)| *l).collect();
        let eigenfunctions: Vec<Vec<f32>> = pairs.into_iter().map(|(_, phi)| phi).collect();

        Self {
            rank,
            eigenvalues,
            eigenfunctions,
            resolution: n,
        }
    }

    /// Reconstructs the continuous graphon W_k(u, v) from the top-k spectral modes.
    pub fn reconstruct(&self, u: f32, v: f32) -> f32 {
        let n = self.resolution;
        let x = u.clamp(0.0, 1.0) * ((n - 1) as f32);
        let y = v.clamp(0.0, 1.0) * ((n - 1) as f32);

        let i0 = (x.floor() as usize).min(n - 2);
        let j0 = (y.floor() as usize).min(n - 2);
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let dx = x - (i0 as f32);
        let dy = y - (j0 as f32);

        let mut sum = 0.0f32;
        for m in 0..self.rank {
            let lambda = self.eigenvalues[m];
            let phi = &self.eigenfunctions[m];

            let phi_u = (1.0 - dx) * phi[i0] + dx * phi[i1];
            let phi_v = (1.0 - dy) * phi[j0] + dy * phi[j1];

            sum += lambda * phi_u * phi_v;
        }

        sum.clamp(0.0, 1.0)
    }

    /// Evaluates the fraction of total L2 Frobenius energy captured by the rank-k spectral modes:
    /// E_k = (\sum_{j=1}^k \lambda_j^2) / ||W||_L2^2
    pub fn spectral_energy_ratio(&self, graphon: &ContinuousGraphon) -> f32 {
        let total_l2 = graphon.l2_norm();
        if total_l2 < GRAPHON_EPSILON {
            return 1.0;
        }
        let captured_sq: f32 = self.eigenvalues.iter().map(|&x| x * x).sum();
        let total_sq = total_l2 * total_l2;
        (captured_sq / total_sq).clamp(0.0, 1.0)
    }

    /// Verifies the orthonormality of the empirical eigenfunctions:
    /// \int_0^1 \phi_a(u) \phi_b(u) du = \delta_{ab}
    pub fn max_orthogonality_error(&self) -> f32 {
        let n = self.resolution;
        let inv_n = 1.0 / (n as f32);
        let mut max_err = 0.0f32;

        for a in 0..self.rank {
            for b in 0..self.rank {
                let mut dot = 0.0f32;
                for i in 0..n {
                    dot += self.eigenfunctions[a][i] * self.eigenfunctions[b][i];
                }
                let integral = dot * inv_n;
                let expected = if a == b { 1.0 } else { 0.0 };
                let err = (integral - expected).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        max_err
    }
}

/// Continuous Attention Resampling and Cross-Attention Interpolator Engine.
pub struct GraphonAttentionInterpolator;

impl GraphonAttentionInterpolator {
    /// Resamples the continuous graphon W(u, v) into a target attention matrix
    /// of arbitrary size S_target x S_target.
    pub fn resample(graphon: &ContinuousGraphon, target_len: usize) -> Vec<f32> {
        assert!(target_len > 0, "Target sequence length must be positive");
        let mut resampled = vec![0.0f32; target_len * target_len];

        for i in 0..target_len {
            let u = (i as f32 + 0.5) / (target_len as f32);
            for j in 0..target_len {
                let v = (j as f32 + 0.5) / (target_len as f32);
                resampled[i * target_len + j] = graphon.evaluate(u, v);
            }
        }

        resampled
    }

    /// Evaluates the transferability distortion between two sequence representations
    /// S_1 and S_2 embedded in the continuous graphon domain:
    /// ||W_{A^{(S_1)}} - W_{A^{(S_2)}}||_L2
    pub fn transferability_error(
        graphon1: &ContinuousGraphon,
        graphon2: &ContinuousGraphon,
    ) -> f32 {
        let n = graphon1.resolution.min(graphon2.resolution);
        let mut sum_sq = 0.0f32;

        for i in 0..n {
            let u = (i as f32 + 0.5) / (n as f32);
            for j in 0..n {
                let v = (j as f32 + 0.5) / (n as f32);
                let w1 = graphon1.evaluate(u, v);
                let w2 = graphon2.evaluate(u, v);
                let diff = w1 - w2;
                sum_sq += diff * diff;
            }
        }

        (sum_sq / ((n * n) as f32)).sqrt()
    }
}

/// Configuration settings for Graphon Continuous Limit Attention.
#[derive(Debug, Clone)]
pub struct GraphonAttentionConfig {
    pub grid_resolution: usize,
    pub num_steps: usize,
    pub spectral_rank: usize,
}

impl Default for GraphonAttentionConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 32,
            num_steps: 8,
            spectral_rank: 4,
        }
    }
}

/// Verification report capturing performance and mathematical invariants of Phase 24.
#[derive(Debug, Clone)]
pub struct GraphonAttentionReport {
    pub grid_resolution: usize,
    pub num_steps: usize,
    pub spectral_rank: usize,
    pub cut_metric_error: f32,
    pub spectral_energy_ratio: f32,
    pub eigenfunction_orthogonality_err: f32,
    pub transferability_error: f32,
    pub resampling_latency_us: f32,
    pub compression_ratio: f32,
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator benchmark harness for Phase 24.
#[derive(Debug, Default)]
pub struct GraphonAttentionEvaluator {
    pub config: GraphonAttentionConfig,
}

impl GraphonAttentionEvaluator {
    pub fn new(config: GraphonAttentionConfig) -> Self {
        Self { config }
    }

    /// Executes performance evaluation and mathematical invariant validation.
    pub fn evaluate_performance(&self) -> GraphonAttentionReport {
        let n = self.config.grid_resolution;
        let m = self.config.num_steps;
        let k = self.config.spectral_rank;

        // Generate synthetic long-context attention matrix (S = 256)
        let seq_len = 256;
        let mut raw_attention = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                let u = (i as f32) / (seq_len as f32);
                let v = (j as f32) / (seq_len as f32);
                // Continuous limit kernel: W(u, v) = exp(-4 * |u - v|) + 0.3 * cos(2*pi*u)*cos(2*pi*v)
                let dist = (u - v).abs();
                let diag = (-4.0 * dist).exp();
                let global_mode = 0.3 * (2.0 * std::f32::consts::PI * u).cos()
                    * (2.0 * std::f32::consts::PI * v).cos();
                raw_attention[i * seq_len + j] = (diag + global_mode).clamp(0.0, 1.0);
            }
        }

        // 1. Construct continuous graphon
        let continuous_graphon =
            ContinuousGraphon::from_attention_matrix(&raw_attention, seq_len, n);

        // 2. Compute Frieze-Kannan step-graphon partition
        let step_partition =
            StepGraphonPartition::from_continuous(&continuous_graphon, m);
        let cut_metric_error = step_partition.cut_metric_distance(&continuous_graphon);

        // 3. Compute spectral graphon decomposition
        let spectral_graphon = SpectralGraphon::decompose(&continuous_graphon, k);
        let spectral_energy_ratio =
            spectral_graphon.spectral_energy_ratio(&continuous_graphon);
        let eigenfunction_orthogonality_err = spectral_graphon.max_orthogonality_error();

        // 4. Benchmark continuous attention resampling to target sequence length (e.g. S_target = 64)
        let target_len = 64;
        let resample_trials = 200;
        let resample_start = Instant::now();
        let mut last_resampled = Vec::new();
        for _ in 0..resample_trials {
            last_resampled =
                GraphonAttentionInterpolator::resample(&continuous_graphon, target_len);
        }
        let resample_duration = resample_start.elapsed();
        let resampling_latency_us =
            (resample_duration.as_nanos() as f32 / (resample_trials as f32)) / 1000.0;
        let _ = last_resampled;

        // 5. Transferability verification between S = 64 and S = 128
        let graphon_64 = ContinuousGraphon::from_attention_matrix(&last_resampled, target_len, n);
        let resampled_128 = GraphonAttentionInterpolator::resample(&continuous_graphon, 128);
        let graphon_128 = ContinuousGraphon::from_attention_matrix(&resampled_128, 128, n);
        let transferability_error =
            GraphonAttentionInterpolator::transferability_error(&graphon_64, &graphon_128);

        // Memory footprint: discrete matrix (seq_len x seq_len x 4 B) vs step partition (m x m x 4 B) + spectral modes
        let raw_attention_bytes = seq_len * seq_len * 4;
        let step_partition_bytes = m * m * 4;
        let spectral_bytes = (k + k * n) * 4;
        let total_memory_footprint_bytes = step_partition_bytes + spectral_bytes + 64;
        let compression_ratio =
            (raw_attention_bytes as f32) / (total_memory_footprint_bytes as f32);

        GraphonAttentionReport {
            grid_resolution: n,
            num_steps: m,
            spectral_rank: k,
            cut_metric_error,
            spectral_energy_ratio,
            eigenfunction_orthogonality_err,
            transferability_error,
            resampling_latency_us,
            compression_ratio,
            total_memory_footprint_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_graphon_bilinear_interpolation() {
        let n = 16;
        let mut graphon = ContinuousGraphon::new(n);
        for i in 0..n {
            for j in 0..n {
                graphon.kernel_values[i * n + j] = ((i + j) as f32) / (2.0 * (n - 1) as f32);
            }
        }

        assert!((graphon.evaluate(0.0, 0.0) - 0.0).abs() < 1e-4);
        assert!((graphon.evaluate(1.0, 1.0) - 1.0).abs() < 1e-4);
        let mid = graphon.evaluate(0.5, 0.5);
        assert!((mid - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_step_graphon_partition_averaging() {
        let n = 16;
        let mut graphon = ContinuousGraphon::new(n);
        for val in graphon.kernel_values.iter_mut() {
            *val = 0.75;
        }

        let partition = StepGraphonPartition::from_continuous(&graphon, 4);
        assert_eq!(partition.num_steps, 4);
        for &w in &partition.block_weights {
            assert!((w - 0.75).abs() < 1e-4);
        }

        let cut_dist = partition.cut_metric_distance(&graphon);
        assert!(cut_dist < 1e-5);
    }

    #[test]
    fn test_spectral_graphon_orthonormality() {
        let n = 16;
        let mut graphon = ContinuousGraphon::new(n);
        for i in 0..n {
            for j in 0..n {
                graphon.kernel_values[i * n + j] = if i == j { 0.8 } else { 0.2 };
            }
        }

        let spectral = SpectralGraphon::decompose(&graphon, 3);
        assert_eq!(spectral.rank, 3);
        assert!(spectral.max_orthogonality_error() < 1e-3);
    }

    #[test]
    fn test_spectral_energy_decay() {
        let n = 24;
        let mut graphon = ContinuousGraphon::new(n);
        for i in 0..n {
            for j in 0..n {
                let diff = (i as f32 - j as f32).abs();
                graphon.kernel_values[i * n + j] = (-0.2 * diff).exp();
            }
        }

        let spectral = SpectralGraphon::decompose(&graphon, 4);
        for i in 0..3 {
            assert!(spectral.eigenvalues[i] >= spectral.eigenvalues[i + 1]);
        }
        assert!(spectral.spectral_energy_ratio(&graphon) > 0.80);
    }

    #[test]
    fn test_attention_resampling_transferability() {
        let n = 16;
        let mut graphon = ContinuousGraphon::new(n);
        for i in 0..n {
            for j in 0..n {
                graphon.kernel_values[i * n + j] = 0.5;
            }
        }

        let resampled_32 = GraphonAttentionInterpolator::resample(&graphon, 32);
        let resampled_64 = GraphonAttentionInterpolator::resample(&graphon, 64);
        assert_eq!(resampled_32.len(), 32 * 32);
        assert_eq!(resampled_64.len(), 64 * 64);

        let g32 = ContinuousGraphon::from_attention_matrix(&resampled_32, 32, n);
        let g64 = ContinuousGraphon::from_attention_matrix(&resampled_64, 64, n);
        let err = GraphonAttentionInterpolator::transferability_error(&g32, &g64);
        assert!(err < 1e-4);
    }

    #[test]
    fn test_evaluator_report() {
        let evaluator = GraphonAttentionEvaluator::default();
        let report = evaluator.evaluate_performance();

        assert!(report.cut_metric_error < 0.15);
        assert!(report.spectral_energy_ratio >= 0.85);
        assert!(report.eigenfunction_orthogonality_err < 1e-3);
        assert!(report.transferability_error < 0.15);
        assert!(report.resampling_latency_us < 100.0);
        assert!(report.compression_ratio >= 4.0);
        assert!(report.total_memory_footprint_bytes < 10_000);
    }
}
