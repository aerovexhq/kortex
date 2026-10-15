#![deny(unsafe_code)]

//! Differential Privacy Sketches, Functional Sensitivity Encoders, and Local DP
//! Key-Cache Protections for Edge Robotics Foundation Models.
//!
//! Subsystems:
//! 1. `LipschitzSensitivityEstimator`: Tracks query/key norm bounds and solves exact Gaussian scale.
//! 2. `DifferentiallyPrivateKeyCache`: Manages privatized keys with Box-Muller Gaussian noise.
//! 3. `DpCountMinSketch`: Sub-linear differentially private frequency tracker with harmonic debiasing.
//! 4. `RandomizedLinearSketch`: Fast Johnson-Lindenstrauss random projection engine for key energy.
//! 5. `PrivacySketchesEvaluator`: Comprehensive empirical benchmark and invariant validation harness.

use std::time::Instant;

/// Small numerical epsilon for stability.
pub const DP_EPSILON: f32 = 1e-9;

/// Target differential privacy noise mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpNoiseMechanism {
    ClassicalGaussian,
    AnalyticalGaussian,
    CoordinateLaplace,
}

/// Configuration settings for Local Differential Privacy on attention keys.
#[derive(Debug, Clone)]
pub struct DpKeyConfig {
    pub epsilon: f32,
    pub delta: f32,
    pub key_radius_bound: f32,
    pub query_radius_bound: f32,
    pub value_radius_bound: f32,
    pub head_dim: usize,
    pub mechanism: DpNoiseMechanism,
    pub sketch_depth: usize,
    pub sketch_width: usize,
    pub sketch_dim: usize,
}

impl Default for DpKeyConfig {
    fn default() -> Self {
        Self {
            epsilon: 2.0,
            delta: 1e-5,
            key_radius_bound: 1.0,
            query_radius_bound: 1.0,
            value_radius_bound: 1.0,
            head_dim: 128,
            mechanism: DpNoiseMechanism::AnalyticalGaussian,
            sketch_depth: 5,
            sketch_width: 64,
            sketch_dim: 16,
        }
    }
}

/// Lipschitz Sensitivity Estimator and Analytical Noise Calibrator.
#[derive(Debug, Clone)]
pub struct LipschitzSensitivityEstimator {
    pub config: DpKeyConfig,
    pub delta_2_key: f32,
    pub delta_2_logit: f32,
    pub calibrated_sigma: f32,
    pub calibrated_laplace_b: f32,
}

impl LipschitzSensitivityEstimator {
    pub fn new(config: DpKeyConfig) -> Self {
        assert!(config.epsilon > 0.0, "Epsilon must be strictly positive");
        assert!(
            config.delta > 0.0 && config.delta < 1.0,
            "Delta must be in (0, 1)"
        );
        assert!(config.head_dim > 0, "Head dimension must be non-zero");

        let delta_2_key = 2.0 * config.key_radius_bound;
        let sqrt_d = (config.head_dim as f32).sqrt();
        let delta_2_logit = (2.0 * config.query_radius_bound * config.key_radius_bound) / sqrt_d;

        let delta_1_key = 2.0 * sqrt_d * config.key_radius_bound;
        let calibrated_laplace_b = delta_1_key / config.epsilon;

        let calibrated_sigma = match config.mechanism {
            DpNoiseMechanism::ClassicalGaussian => {
                let log_term = (1.25f32 / config.delta).ln();
                (delta_2_key * (2.0 * log_term).sqrt()) / config.epsilon
            }
            DpNoiseMechanism::AnalyticalGaussian => {
                Self::solve_analytical_gaussian_sigma(delta_2_key, config.epsilon, config.delta)
            }
            DpNoiseMechanism::CoordinateLaplace => 0.0,
        };

        Self {
            config,
            delta_2_key,
            delta_2_logit,
            calibrated_sigma,
            calibrated_laplace_b,
        }
    }

    /// Numerical root finder for the Analytical Gaussian Mechanism (Balle & Wang, 2018).
    pub fn solve_analytical_gaussian_sigma(delta_2: f32, eps: f32, delta: f32) -> f32 {
        let mut low = delta_2 / (2.0 * eps);
        let mut high = (delta_2 * (2.0 * (1.25f32 / delta).ln()).sqrt()) / eps;

        for _ in 0..32 {
            let mid = 0.5 * (low + high);
            let s = mid;
            let val1 = (delta_2 / (2.0 * s)) - (eps * s / delta_2);
            let val2 = -(delta_2 / (2.0 * s)) - (eps * s / delta_2);

            let lhs =
                Self::standard_normal_cdf(val1) - eps.exp() * Self::standard_normal_cdf(val2);
            if lhs > delta {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }

    /// High-precision approximation of standard normal cumulative distribution Phi(z).
    #[inline]
    pub fn standard_normal_cdf(z: f32) -> f32 {
        0.5 * (1.0 + Self::erf(z / std::f32::consts::SQRT_2))
    }

    /// Chebyshev polynomial approximation of error function erf(x).
    #[inline]
    pub fn erf(x: f32) -> f32 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254_829_6
                + t * (-0.284_496_72
                    + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
        let result = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 {
            result
        } else {
            -result
        }
    }

    /// Radial contraction projection: k_proj = k * min(1, R_K / ||k||_2).
    #[inline]
    pub fn project_key(&self, key: &mut [f32]) -> f32 {
        let norm_sq: f32 = key.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        let r_k = self.config.key_radius_bound;

        if norm > r_k && norm > DP_EPSILON {
            let scale = r_k / norm;
            for x in key.iter_mut() {
                *x *= scale;
            }
            r_k
        } else {
            norm
        }
    }
}

/// Differentially Private Key Cache Manager with Local DP Noise Injection.
#[derive(Debug, Clone)]
pub struct DifferentiallyPrivateKeyCache {
    pub estimator: LipschitzSensitivityEstimator,
    pub max_sequence_length: usize,
    pub keys: Vec<f32>,
    pub token_count: usize,
    pub rng_state: u64,
}

impl DifferentiallyPrivateKeyCache {
    pub fn new(
        estimator: LipschitzSensitivityEstimator,
        max_sequence_length: usize,
        rng_seed: u64,
    ) -> Self {
        let d = estimator.config.head_dim;
        let keys = vec![0.0f32; max_sequence_length * d];
        let rng_state = if rng_seed == 0 {
            0x853c49e6748fea9b
        } else {
            rng_seed
        };

        Self {
            estimator,
            max_sequence_length,
            keys,
            token_count: 0,
            rng_state,
        }
    }

    /// Fast 64-bit Xorshift PRNG returning uniform float in (0, 1].
    #[inline]
    pub fn next_uniform(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let bits = (self.rng_state >> 40) as u32;
        ((bits as f32) + 1.0) / 16777217.0
    }

    /// Box-Muller transform for generating standard Gaussian variates.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Appends a key vector, applies norm projection, injects calibrated noise, and stores it.
    pub fn append_key(&mut self, clean_key: &[f32]) -> usize {
        let d = self.estimator.config.head_dim;
        assert_eq!(clean_key.len(), d, "Key dimension mismatch");
        assert!(
            self.token_count < self.max_sequence_length,
            "Cache capacity exceeded"
        );

        let mut projected_key = clean_key.to_vec();
        self.estimator.project_key(&mut projected_key);

        match self.estimator.config.mechanism {
            DpNoiseMechanism::ClassicalGaussian | DpNoiseMechanism::AnalyticalGaussian => {
                let sigma = self.estimator.calibrated_sigma;
                for item in projected_key.iter_mut().take(d) {
                    let noise = self.next_gaussian() * sigma;
                    *item += noise;
                }
            }
            DpNoiseMechanism::CoordinateLaplace => {
                let b = self.estimator.calibrated_laplace_b;
                for item in projected_key.iter_mut().take(d) {
                    let u = self.next_uniform() - 0.5;
                    let noise = if u < 0.0 {
                        b * (2.0 * (0.5 + u)).ln()
                    } else {
                        -b * (2.0 * (0.5 - u)).ln()
                    };
                    *item += noise;
                }
            }
        }

        let target_idx = self.token_count * d;
        self.keys[target_idx..target_idx + d].copy_from_slice(&projected_key);

        let assigned_pos = self.token_count;
        self.token_count += 1;
        assigned_pos
    }

    /// Evaluates scalar attention logits: s_tau = (q^T k_tau) / sqrt(d) across all active tokens.
    pub fn compute_attention_logits(&self, query: &[f32], out_logits: &mut [f32]) {
        let d = self.estimator.config.head_dim;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        assert_eq!(query.len(), d);
        assert!(out_logits.len() >= self.token_count);

        for (t, out_logit) in out_logits.iter_mut().enumerate().take(self.token_count) {
            let offset = t * d;
            let mut dot = 0.0f32;
            for (i, &q_val) in query.iter().enumerate().take(d) {
                dot += q_val * self.keys[offset + i];
            }
            *out_logit = dot * inv_sqrt_d;
        }
    }
}

/// Differentially Private Count-Min Sketch (DP-CMS) with Harmonic Debiasing.
#[derive(Debug, Clone)]
pub struct DpCountMinSketch {
    pub depth: usize,
    pub width: usize,
    pub table: Vec<f32>,
    pub hash_seeds_a: Vec<u64>,
    pub hash_seeds_b: Vec<u64>,
    pub noise_scale: f32,
    pub harmonic_correction: f32,
}

impl DpCountMinSketch {
    pub fn new(depth: usize, width: usize, epsilon: f32, max_mass: f32) -> Self {
        assert!(depth > 0 && width > 0);
        let table = vec![0.0f32; depth * width];

        let mut hash_seeds_a = Vec::with_capacity(depth);
        let mut hash_seeds_b = Vec::with_capacity(depth);
        let mut seed = 0x9e3779b97f4a7c15u64;

        for _ in 0..depth {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            hash_seeds_a.push((seed | 1) % 1000000007);
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            hash_seeds_b.push(seed % 1000000007);
        }

        let delta_1 = (depth as f32) * max_mass;
        let noise_scale = delta_1 / epsilon;

        let mut h_d = 0.0f32;
        for k in 1..=depth {
            h_d += 1.0 / (k as f32);
        }
        let harmonic_correction = noise_scale * h_d;

        Self {
            depth,
            width,
            table,
            hash_seeds_a,
            hash_seeds_b,
            noise_scale,
            harmonic_correction,
        }
    }

    #[inline]
    pub fn hash(&self, row: usize, token_id: usize) -> usize {
        let a = self.hash_seeds_a[row];
        let b = self.hash_seeds_b[row];
        let x = token_id as u64;
        let p = 1000000007u64;
        (((a.wrapping_mul(x)).wrapping_add(b) % p) as usize) % self.width
    }

    pub fn update(&mut self, token_id: usize, mass: f32) {
        for row in 0..self.depth {
            let col = self.hash(row, token_id);
            self.table[row * self.width + col] += mass;
        }
    }

    pub fn privatize_sketch(&mut self, rng_seed: u64) {
        let mut state = if rng_seed == 0 {
            0xdeadbeefcafebabe
        } else {
            rng_seed
        };
        let b = self.noise_scale;

        for entry in self.table.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (((state >> 40) as f32) + 1.0) / 16777217.0 - 0.5;
            let laplace_noise = if u < 0.0 {
                b * (2.0 * (0.5 + u)).ln()
            } else {
                -b * (2.0 * (0.5 - u)).ln()
            };
            *entry += laplace_noise;
        }
    }

    pub fn query_debiased(&self, token_id: usize, total_stream_mass: f32) -> f32 {
        let mut min_val = f32::INFINITY;
        for row in 0..self.depth {
            let col = self.hash(row, token_id);
            let val = self.table[row * self.width + col];
            if val < min_val {
                min_val = val;
            }
        }

        let background_collision = total_stream_mass / (self.width as f32);
        (min_val + self.harmonic_correction - background_collision).max(0.0)
    }
}

/// Randomized Linear Projection Sketch (Fast Johnson-Lindenstrauss Transform).
#[derive(Debug, Clone)]
pub struct RandomizedLinearSketch {
    pub original_dim: usize,
    pub sketch_dim: usize,
    pub projection_seed: u64,
}

impl RandomizedLinearSketch {
    pub fn new(original_dim: usize, sketch_dim: usize, seed: u64) -> Self {
        assert!(sketch_dim <= original_dim);
        Self {
            original_dim,
            sketch_dim,
            projection_seed: if seed == 0 {
                0x517cc1b727220a95
            } else {
                seed
            },
        }
    }

    /// Projects key vector k into sketch z = Phi * k in R^m without materializing dense matrices in RAM.
    pub fn project(&self, key: &[f32], out_sketch: &mut [f32]) {
        let d = self.original_dim;
        let m = self.sketch_dim;
        let norm_factor = 1.0 / (m as f32).sqrt();
        assert_eq!(key.len(), d);
        assert_eq!(out_sketch.len(), m);

        let mut rng = self.projection_seed;

        for item in out_sketch.iter_mut().take(m) {
            let mut dot = 0.0f32;
            for &k_val in key.iter().take(d) {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u1 = (((rng >> 40) as f32) + 1.0) / 16777217.0;
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u2 = (((rng >> 40) as f32) + 1.0) / 16777217.0;
                let phi_ij = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

                dot += phi_ij * k_val;
            }
            *item = dot * norm_factor;
        }
    }

    /// Evaluates approximate L2 energy ||k||_2^2 from compact sketch.
    #[inline]
    pub fn estimate_energy(&self, sketch: &[f32]) -> f32 {
        sketch.iter().map(|&z| z * z).sum()
    }
}

/// Verification report capturing performance and mathematical invariants of Phase 25.
#[derive(Debug, Clone)]
pub struct PrivacySketchesReport {
    pub epsilon: f32,
    pub delta: f32,
    pub calibrated_sigma: f32,
    pub lipschitz_logit_sensitivity: f32,
    pub key_norm_bound_verified: bool,
    pub mutual_information_reduction_pct: f32,
    pub sketch_query_latency_us: f32,
    pub sketch_energy_rel_error: f32,
    pub count_min_debiased_mse: f32,
    pub critical_privacy_epsilon: f32,
    pub action_safety_override_triggered: bool,
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator benchmark harness for Phase 25.
#[derive(Debug, Default)]
pub struct PrivacySketchesEvaluator {
    pub config: DpKeyConfig,
}

impl PrivacySketchesEvaluator {
    pub fn new(config: DpKeyConfig) -> Self {
        Self { config }
    }

    /// Executes performance evaluation and mathematical invariant validation.
    pub fn evaluate_performance(&self) -> PrivacySketchesReport {
        let d = self.config.head_dim;
        let eps = self.config.epsilon;
        let delta = self.config.delta;

        // 1. Initialize sensitivity estimator and calibrate sigma
        let estimator = LipschitzSensitivityEstimator::new(self.config.clone());
        let calibrated_sigma = estimator.calibrated_sigma;

        // Verify norm projection
        let mut outlier_key = vec![2.5f32; d];
        let post_proj_norm = estimator.project_key(&mut outlier_key);
        let key_norm_bound_verified = (post_proj_norm - self.config.key_radius_bound).abs() < 1e-5;

        // 2. Differentially Private Key Cache evaluation
        let seq_len = 128;
        let mut dp_cache =
            DifferentiallyPrivateKeyCache::new(estimator.clone(), seq_len, 0x12345678abcdef01);

        for t in 0..seq_len {
            let key: Vec<f32> = (0..d)
                .map(|i| ((t * 7 + i * 13) as f32 * 0.05).sin())
                .collect();
            dp_cache.append_key(&key);
        }

        // Mutual Information under Gaussian LDP channel:
        // By Data Processing Inequality: I(X; K~) <= I(K; K~)
        // For signal power P = R_K^2 / d and noise variance sigma^2 per coordinate:
        // I(K; K~) = (d / 2) * ln(1 + R_K^2 / (d * sigma^2))
        let r_k = self.config.key_radius_bound;
        let snr = (r_k * r_k) / ((d as f32) * calibrated_sigma * calibrated_sigma);
        let mi_channel = 0.5 * (d as f32) * (1.0 + snr).ln();
        let max_entropy = (32000.0f32).ln();
        let mutual_information_reduction_pct =
            ((1.0 - (mi_channel / max_entropy).min(1.0)) * 100.0).max(0.0);

        // 3. DP Count-Min Sketch evaluation
        let mut cms = DpCountMinSketch::new(
            self.config.sketch_depth,
            self.config.sketch_width,
            eps,
            1.0,
        );

        let mut total_mass = 0.0f32;
        for i in 0..seq_len {
            let mass = 1.0 + (i as f32 * 0.1);
            cms.update(i, mass);
            total_mass += mass;
        }
        cms.privatize_sketch(0x9876543210fedcba);

        // Benchmark query latency and compute MSE
        let query_trials = 1000;
        let t_start = Instant::now();
        let mut mse_sum = 0.0f32;
        for t in 0..query_trials {
            let target_tok = t % seq_len;
            let est_freq = cms.query_debiased(target_tok, total_mass);
            let true_freq = 1.0 + (target_tok as f32 * 0.1);
            let diff = est_freq - true_freq;
            mse_sum += diff * diff;
        }
        let elapsed = t_start.elapsed();
        let sketch_query_latency_us =
            (elapsed.as_nanos() as f32 / (query_trials as f32)) / 1000.0;
        let count_min_debiased_mse = mse_sum / (query_trials as f32);

        // 4. Randomized Linear Sketch evaluation (JL Energy Preservation)
        let jl_sketch = RandomizedLinearSketch::new(d, self.config.sketch_dim, 0xabcdef0123456789);
        let mut test_key = vec![0.0f32; d];
        for (i, item) in test_key.iter_mut().enumerate().take(d) {
            *item = ((i as f32) * 0.1).sin();
        }
        let true_energy: f32 = test_key.iter().map(|&x| x * x).sum();
        let mut out_sketch = vec![0.0f32; self.config.sketch_dim];
        jl_sketch.project(&test_key, &mut out_sketch);
        let est_energy = jl_sketch.estimate_energy(&out_sketch);
        let sketch_energy_rel_error = ((est_energy - true_energy) / true_energy).abs();

        // 5. Critical Privacy Budget and Action Safety Check
        let delta_z_min = 0.85f32; // Typical logit margin between primary and competing command
        let d_model = 2048.0f32;
        let w_u_norm = 1.0f32;
        let r_q = self.config.query_radius_bound;
        let r_k = self.config.key_radius_bound;
        let r_v = self.config.value_radius_bound;
        let eta_fail_quantile = 3.719f32; // Phi^-1(1 - 10^-4)

        let critical_privacy_epsilon = ((4.0 * r_q * r_k * r_v * w_u_norm
            * (1.25f32 / delta).ln().sqrt())
            / ((d as f32 * d_model).sqrt() * delta_z_min))
            * eta_fail_quantile;

        let action_safety_override_triggered = eps < critical_privacy_epsilon;

        // Memory footprint: cache (seq x d x 4 B) + CMS table + JL metadata
        let cache_bytes = seq_len * d * 4;
        let cms_bytes = self.config.sketch_depth * self.config.sketch_width * 4;
        let jl_bytes = self.config.sketch_dim * 4 + 64;
        let total_memory_footprint_bytes = cache_bytes + cms_bytes + jl_bytes;

        PrivacySketchesReport {
            epsilon: eps,
            delta,
            calibrated_sigma,
            lipschitz_logit_sensitivity: estimator.delta_2_logit,
            key_norm_bound_verified,
            mutual_information_reduction_pct,
            sketch_query_latency_us,
            sketch_energy_rel_error,
            count_min_debiased_mse,
            critical_privacy_epsilon,
            action_safety_override_triggered,
            total_memory_footprint_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lipschitz_norm_projection() {
        let config = DpKeyConfig::default();
        let estimator = LipschitzSensitivityEstimator::new(config);

        let mut key = vec![3.0f32; 128];
        let norm = estimator.project_key(&mut key);
        assert!((norm - 1.0).abs() < 1e-5);

        let post_norm: f32 = key.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((post_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_analytical_gaussian_solver() {
        let delta_2 = 2.0;
        let eps = 1.0;
        let delta = 1e-5;
        let sigma =
            LipschitzSensitivityEstimator::solve_analytical_gaussian_sigma(delta_2, eps, delta);

        assert!(sigma > 0.0);
        let s = sigma;
        let val1 = (delta_2 / (2.0 * s)) - (eps * s / delta_2);
        let val2 = -(delta_2 / (2.0 * s)) - (eps * s / delta_2);
        let lhs = LipschitzSensitivityEstimator::standard_normal_cdf(val1)
            - eps.exp() * LipschitzSensitivityEstimator::standard_normal_cdf(val2);
        assert!((lhs - delta).abs() < 1e-4);
    }

    #[test]
    fn test_dp_key_cache_noise_injection() {
        let config = DpKeyConfig::default();
        let estimator = LipschitzSensitivityEstimator::new(config);
        let mut cache =
            DifferentiallyPrivateKeyCache::new(estimator, 32, 0xdeadbeefcafebabe);

        let clean_key = vec![0.5f32; 128];
        let idx = cache.append_key(&clean_key);
        assert_eq!(idx, 0);

        let d = 128;
        let cached_key = &cache.keys[0..d];
        let diff: f32 = cached_key
            .iter()
            .zip(clean_key.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "Noise must be non-zero");
    }

    #[test]
    fn test_count_min_sketch_debiasing() {
        let mut cms = DpCountMinSketch::new(5, 64, 2.0, 1.0);
        for _ in 0..10 {
            cms.update(7, 1.0);
        }
        cms.privatize_sketch(0x123456789abcdef0);
        let debiased = cms.query_debiased(7, 10.0);
        assert!(debiased > 0.0);
    }

    #[test]
    fn test_randomized_linear_sketch_energy() {
        let d = 128;
        let m = 32;
        let sketch = RandomizedLinearSketch::new(d, m, 0x1122334455667788);

        let key: Vec<f32> = (0..d).map(|i| ((i as f32) * 0.1).cos()).collect();
        let true_energy: f32 = key.iter().map(|&x| x * x).sum();

        let mut out_sketch = vec![0.0f32; m];
        sketch.project(&key, &mut out_sketch);
        let est_energy = sketch.estimate_energy(&out_sketch);

        let rel_err = ((est_energy - true_energy) / true_energy).abs();
        assert!(
            rel_err < 0.60,
            "JL projection energy rel error must be bounded, got {}",
            rel_err
        );
    }

    #[test]
    fn test_evaluator_report() {
        let evaluator = PrivacySketchesEvaluator::default();
        let report = evaluator.evaluate_performance();

        assert!(report.key_norm_bound_verified);
        assert!(report.mutual_information_reduction_pct >= 85.0);
        assert!(report.sketch_query_latency_us < 10.0);
        assert!(report.sketch_energy_rel_error < 0.60);
        assert!(report.count_min_debiased_mse < 50.0);
        assert!(report.total_memory_footprint_bytes < 100_000);
    }
}
