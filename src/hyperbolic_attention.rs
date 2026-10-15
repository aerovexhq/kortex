#![deny(unsafe_code)]

//! Non-Euclidean Hyperbolic Manifolds, Poincaré Ball Models, and Lorentz / Minkowski
//! Geometry for Hierarchical Attention Key Embeddings and KV Cache Compaction.
//!
//! Subsystems:
//! 1. `LorentzVector`: Upper sheet of hyperboloid L_c^d in R^{1, d} with Lorentzian metric.
//! 2. `PoincareVector`: Conformal Poincare ball B_c^d with Mobius addition and geodesic distance.
//! 3. `HyperbolicPolarQuantizer`: Decoupled radial-angular compression codec.
//! 4. `HyperbolicAttentionCache`: Distance-based hyperbolic softmax attention and key storage.
//! 5. `HyperbolicEvaluator`: Empirical benchmark and invariant validation harness for Gate 19.

use std::time::Instant;

/// Default hyperbolic curvature parameter c > 0 (Sectional curvature kappa = -c).
pub const DEFAULT_HYPERBOLIC_CURVATURE: f32 = 1.0;

/// Numerical epsilon preventing floating-point boundary singularities.
pub const HYPERBOLIC_EPSILON: f32 = 1e-6;

// =============================================================================
// 1. Minkowski / Lorentz Hyperboloid Vector Space (R^{1, d})
// =============================================================================

/// Point residing on the Lorentz Hyperboloid L_c^d in R^{1, d}.
/// Constraint: -x_0^2 + sum_{i=1}^d x_i^2 = -1/c with x_0 > 0.
#[derive(Clone, Debug, PartialEq)]
pub struct LorentzVector {
    /// Time-like coordinate x_0 >= 1/sqrt(c).
    pub x0: f32,
    /// Space-like coordinates x_s in R^d.
    pub xs: Vec<f32>,
    /// Curvature scale c > 0.
    pub c: f32,
}

impl LorentzVector {
    /// Constructs a Lorentz vector from space-like coordinates by solving for x0:
    /// x0 = sqrt(1/c + ||xs||_2^2).
    pub fn from_spacelike(xs: Vec<f32>, c: f32) -> Self {
        assert!(c > 0.0, "Curvature c must be strictly positive.");
        let norm_sq: f32 = xs.iter().map(|&v| v * v).sum();
        let x0 = ((1.0 / c) + norm_sq).max(1e-12).sqrt();
        Self { x0, xs, c }
    }

    /// Evaluates the Lorentzian (Minkowski) inner product: <x, y>_L = -x0*y0 + xs^T ys.
    #[inline]
    pub fn lorentz_inner_product(&self, other: &Self) -> f32 {
        assert_eq!(self.xs.len(), other.xs.len(), "Dimension mismatch in Lorentz inner product.");
        let mut spacelike_dot = 0.0f32;
        for (a, b) in self.xs.iter().zip(other.xs.iter()) {
            spacelike_dot += a * b;
        }
        -self.x0 * other.x0 + spacelike_dot
    }

    /// Geodesic distance on the Lorentz hyperboloid:
    /// d_L(x, y) = (1/sqrt(c)) * arcosh(-c * <x, y>_L).
    pub fn geodesic_distance(&self, other: &Self) -> f32 {
        let inner = self.lorentz_inner_product(other);
        let arg = (-self.c * inner).max(1.0 + 1e-7);
        // arcosh(z) = ln(z + sqrt(z^2 - 1))
        let arcosh_val = (arg + (arg * arg - 1.0).max(0.0).sqrt()).ln();
        arcosh_val / self.c.sqrt()
    }

    /// Stereographic projection to the Poincare ball B_c^d:
    /// u = xs / (sqrt(c)*x0 + 1).
    pub fn to_poincare(&self) -> PoincareVector {
        let denom = self.c.sqrt() * self.x0 + 1.0;
        let scale = 1.0 / denom.max(HYPERBOLIC_EPSILON);
        let u: Vec<f32> = self.xs.iter().map(|&v| v * scale).collect();
        PoincareVector::new(u, self.c)
    }
}

// =============================================================================
// 2. Poincaré Ball Model Vector Space (B_c^d)
// =============================================================================

/// Point residing in the Poincare Ball B_c^d: { u in R^d | c * ||u||^2 < 1 }.
#[derive(Clone, Debug, PartialEq)]
pub struct PoincareVector {
    /// Coordinate vector u in R^d.
    pub coords: Vec<f32>,
    /// Curvature parameter c > 0.
    pub c: f32,
}

impl PoincareVector {
    /// Constructs a Poincare vector and clamps strictly within the open ball:
    /// ||u|| < (1/sqrt(c)) - epsilon.
    pub fn new(coords: Vec<f32>, c: f32) -> Self {
        assert!(c > 0.0, "Curvature c must be strictly positive.");
        let max_norm = (1.0 / c.sqrt()) - HYPERBOLIC_EPSILON;
        let norm_sq: f32 = coords.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();

        let clamped_coords = if norm >= max_norm {
            let scale = max_norm / norm.max(1e-12);
            coords.iter().map(|&v| v * scale).collect()
        } else {
            coords
        };

        Self { coords: clamped_coords, c }
    }

    /// Dimension of the space d.
    #[inline]
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Euclidean norm ||u||_2.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.coords.iter().map(|&v| v * v).sum::<f32>().sqrt()
    }

    /// Conformal factor lambda_u^c = 2 / (1 - c * ||u||^2).
    #[inline]
    pub fn conformal_factor(&self) -> f32 {
        let norm_sq = self.coords.iter().map(|&v| v * v).sum::<f32>();
        let denom = (1.0 - self.c * norm_sq).max(HYPERBOLIC_EPSILON);
        2.0 / denom
    }

    /// Geodesic distance from the origin: R = (2/sqrt(c)) * artanh(sqrt(c) * ||u||).
    pub fn radius_from_origin(&self) -> f32 {
        let sc = self.c.sqrt();
        let r = (sc * self.norm()).min(1.0 - HYPERBOLIC_EPSILON);
        // artanh(x) = 0.5 * ln((1 + x) / (1 - x))
        let artanh_val = 0.5 * ((1.0 + r) / (1.0 - r).max(1e-12)).ln();
        (2.0 / sc) * artanh_val
    }

    /// Mobius vector addition: u (+) v.
    pub fn mobius_add(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "Dimension mismatch in mobius_add.");
        let u_dot_v: f32 = self.coords.iter().zip(other.coords.iter()).map(|(&a, &b)| a * b).sum();
        let u_sq: f32 = self.coords.iter().map(|&a| a * a).sum();
        let v_sq: f32 = other.coords.iter().map(|&b| b * b).sum();

        let denom = 1.0 + 2.0 * self.c * u_dot_v + self.c * self.c * u_sq * v_sq;
        let denom_inv = 1.0 / denom.max(HYPERBOLIC_EPSILON);

        let alpha = 1.0 + 2.0 * self.c * u_dot_v + self.c * v_sq;
        let beta = 1.0 - self.c * u_sq;

        let mut res = vec![0.0f32; self.dim()];
        for (i, r) in res.iter_mut().enumerate().take(self.dim()) {
            *r = (alpha * self.coords[i] + beta * other.coords[i]) * denom_inv;
        }

        Self::new(res, self.c)
    }

    /// Mobius vector subtraction: u (-) v = u (+) (-v).
    pub fn mobius_sub(&self, other: &Self) -> Self {
        let neg_other_coords: Vec<f32> = other.coords.iter().map(|&v| -v).collect();
        let neg_other = Self::new(neg_other_coords, other.c);
        self.mobius_add(&neg_other)
    }

    /// Poincare geodesic distance:
    /// d_B(u, v) = (1/sqrt(c)) * arcosh(1 + 2*c*||u - v||^2 / ((1 - c*||u||^2)*(1 - c*||v||^2))).
    #[inline]
    pub fn geodesic_distance(&self, other: &Self) -> f32 {
        let u_sq: f32 = self.coords.iter().map(|&x| x * x).sum();
        let v_sq: f32 = other.coords.iter().map(|&x| x * x).sum();
        let mut diff_sq = 0.0f32;
        for (&a, &b) in self.coords.iter().zip(other.coords.iter()) {
            let diff = a - b;
            diff_sq += diff * diff;
        }

        let denom = ((1.0 - self.c * u_sq) * (1.0 - self.c * v_sq)).max(HYPERBOLIC_EPSILON);
        let arg = (1.0 + (2.0 * self.c * diff_sq) / denom).max(1.0);
        let arcosh_val = (arg + (arg * arg - 1.0).max(0.0).sqrt()).ln();
        arcosh_val / self.c.sqrt()
    }

    /// Inverse stereographic projection to the Lorentz Hyperboloid:
    /// x0 = (1 + c*||u||^2) / (sqrt(c)*(1 - c*||u||^2))
    /// xs = 2*u / (1 - c*||u||^2)
    pub fn to_lorentz(&self) -> LorentzVector {
        let norm_sq = self.coords.iter().map(|&v| v * v).sum::<f32>();
        let denom = (1.0 - self.c * norm_sq).max(HYPERBOLIC_EPSILON);
        let sc = self.c.sqrt();

        let x0 = (1.0 + self.c * norm_sq) / (sc * denom);
        let scale = 2.0 / denom;
        let xs: Vec<f32> = self.coords.iter().map(|&v| v * scale).collect();

        LorentzVector { x0, xs, c: self.c }
    }

    /// Exponential map from the origin: exp_0(v) = (1/sqrt(c)) * tanh(sqrt(c) * ||v||) * (v / ||v||).
    pub fn exp_0(v: &[f32], c: f32) -> Self {
        assert!(c > 0.0, "Curvature c must be strictly positive.");
        let sc = c.sqrt();
        let norm_v = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let scale = if norm_v > 1e-12 {
            (1.0 / sc) * (sc * norm_v).tanh() / norm_v
        } else {
            0.0
        };
        let coords: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        Self::new(coords, c)
    }

    /// Logarithmic map to the origin: log_0(u) = (1/sqrt(c)) * artanh(sqrt(c) * ||u||) * (u / ||u||).
    pub fn log_0(&self) -> Vec<f32> {
        let sc = self.c.sqrt();
        let norm_u = self.norm();
        let scale = if norm_u > 1e-12 {
            let r = (sc * norm_u).min(1.0 - HYPERBOLIC_EPSILON);
            let artanh_val = 0.5 * ((1.0 + r) / (1.0 - r).max(1e-12)).ln();
            (1.0 / sc) * artanh_val / norm_u
        } else {
            0.0
        };
        self.coords.iter().map(|&x| x * scale).collect()
    }
}

// =============================================================================
// 3. Hyperbolic Polar Quantizer (Decoupled Radial-Angular Codec)
// =============================================================================

/// Compact Quantized Representation of a Hyperbolic Point.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedHyperbolicKey {
    /// Quantized radial coordinate (scalar integer in [0, 255]).
    pub radial_bin: u8,
    /// Quantized angular unit direction vector (INT8 scalar quantized components).
    pub angular_codes: Vec<i8>,
    /// Global angular normalization scale.
    pub angular_scale: f32,
    /// Token depth / hierarchy index (for hierarchy-aware routing).
    pub hierarchy_level: u8,
}

/// Quantizer that decouples hyperbolic radius R from spherical direction theta.
#[derive(Clone, Debug)]
pub struct HyperbolicPolarQuantizer {
    pub dim: usize,
    pub curvature: f32,
    pub max_radius: f32,
    pub radial_bits: u32,
    pub num_radial_bins: u32,
    pub lut_r_euc: Vec<f32>,
    pub lut_denom_v: Vec<f32>,
}

impl HyperbolicPolarQuantizer {
    pub fn new(dim: usize, curvature: f32, max_radius: f32, radial_bits: u32) -> Self {
        assert!(radial_bits <= 8, "Radial bits constrained to <= 8 for compact byte packing.");
        let num_radial_bins = 1 << radial_bits;
        let sc = curvature.sqrt();
        let r_step = max_radius / ((num_radial_bins - 1) as f32);

        let mut lut_r_euc = Vec::with_capacity(num_radial_bins as usize);
        let mut lut_denom_v = Vec::with_capacity(num_radial_bins as usize);
        for bin in 0..num_radial_bins {
            let r_hyp = (bin as f32) * r_step;
            let r_euc = (1.0 / sc) * (0.5 * sc * r_hyp).tanh();
            let v_sq = r_euc * r_euc;
            let denom_v = (1.0 - curvature * v_sq).max(HYPERBOLIC_EPSILON);
            lut_r_euc.push(r_euc);
            lut_denom_v.push(denom_v);
        }

        Self {
            dim,
            curvature,
            max_radius,
            radial_bits,
            num_radial_bins,
            lut_r_euc,
            lut_denom_v,
        }
    }

    /// Compresses a continuous Poincare key vector into a QuantizedHyperbolicKey.
    pub fn quantize(&self, point: &PoincareVector, hierarchy_level: u8) -> QuantizedHyperbolicKey {
        let r_hyperbolic = point.radius_from_origin();

        // 1. Uniform quantization of hyperbolic radius R in [0, max_radius]
        let clamped_r = r_hyperbolic.clamp(0.0, self.max_radius);
        let r_step = self.max_radius / ((self.num_radial_bins - 1) as f32);
        let radial_bin = ((clamped_r / r_step).round() as u32).min(self.num_radial_bins - 1) as u8;

        // 2. Extract unit angular direction vector theta in S^{d-1}
        let euclidean_norm = point.norm();
        let mut theta = vec![0.0f32; self.dim];
        if euclidean_norm > 1e-7 {
            let inv_norm = 1.0 / euclidean_norm;
            for (th, &c) in theta.iter_mut().zip(point.coords.iter()) {
                *th = c * inv_norm;
            }
        } else {
            // Degenerate vector at origin: assign canonical unit vector
            theta[0] = 1.0;
        }

        // 3. Symmetric INT8 quantization of spherical direction vector
        let max_abs = theta.iter().fold(0.0f32, |acc, &v| acc.max(v.abs())).max(1e-6);
        let scale = max_abs / 127.0f32;
        let inv_scale = 1.0 / scale;

        let mut angular_codes = vec![0i8; self.dim];
        for (code, &th) in angular_codes.iter_mut().zip(theta.iter()) {
            *code = (th * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }

        QuantizedHyperbolicKey {
            radial_bin,
            angular_codes,
            angular_scale: scale,
            hierarchy_level,
        }
    }

    /// Reconstructs the continuous Poincare key vector from its quantized representation.
    pub fn dequantize(&self, q_key: &QuantizedHyperbolicKey) -> PoincareVector {
        let r_euc = self.lut_r_euc[q_key.radial_bin as usize];

        // Dequantize and normalize direction vector theta
        let mut theta = vec![0.0f32; self.dim];
        let mut norm_sq = 0.0f32;
        for (th, &code) in theta.iter_mut().zip(q_key.angular_codes.iter()) {
            let val = (code as f32) * q_key.angular_scale;
            *th = val;
            norm_sq += val * val;
        }

        let norm = norm_sq.sqrt().max(1e-12);
        let inv_norm = 1.0 / norm;

        // 4. Assemble final reconstructed Poincare vector: u_hat = r_euc * (theta / ||theta||)
        let mut reconstructed_coords = vec![0.0f32; self.dim];
        for (rc, &th) in reconstructed_coords.iter_mut().zip(theta.iter()) {
            *rc = r_euc * (th * inv_norm);
        }

        PoincareVector::new(reconstructed_coords, self.curvature)
    }
}

// =============================================================================
// 4. Hyperbolic Attention Cache & Routing Engine
// =============================================================================

/// Compressed KV-cache key store for hyperbolic attention.
#[derive(Clone, Debug)]
pub struct HyperbolicAttentionCache {
    pub head_dim: usize,
    pub curvature: f32,
    pub beta: f32,
    pub gamma: f32,
    pub quantizer: HyperbolicPolarQuantizer,
    pub keys: Vec<QuantizedHyperbolicKey>,
}

impl HyperbolicAttentionCache {
    pub fn new(head_dim: usize, curvature: f32, max_radius: f32, radial_bits: u32, beta: f32) -> Self {
        let quantizer = HyperbolicPolarQuantizer::new(head_dim, curvature, max_radius, radial_bits);
        let gamma = 0.1f32;
        Self {
            head_dim,
            curvature,
            beta,
            gamma,
            quantizer,
            keys: Vec::new(),
        }
    }

    /// Appends a raw Euclidean key vector, projecting it into the Poincare ball.
    pub fn append_key(&mut self, raw_key: &[f32], hierarchy_depth: u8) {
        assert_eq!(raw_key.len(), self.head_dim, "Key dimension mismatch.");
        let scaled: Vec<f32> = raw_key.iter().map(|&v| v * self.gamma).collect();
        let p_point = PoincareVector::exp_0(&scaled, self.curvature);
        let q_key = self.quantizer.quantize(&p_point, hierarchy_depth);
        self.keys.push(q_key);
    }

    /// Reconstructs Euclidean key vector for active token index.
    pub fn reconstruct_key(&self, idx: usize) -> Vec<f32> {
        let q_key = &self.keys[idx];
        let p_point = self.quantizer.dequantize(q_key);
        let unscaled = p_point.log_0();
        let inv_gamma = 1.0 / self.gamma;
        unscaled.iter().map(|&v| v * inv_gamma).collect()
    }

    /// Computes distance-based hyperbolic softmax attention weights:
    /// alpha_j = exp(-beta * d_B(q, k_j)) / sum_l exp(-beta * d_B(q, k_l)).
    pub fn compute_hyperbolic_attention(
        &self,
        query: &[f32],
        out_weights: &mut [f32],
    ) {
        let seq_len = self.keys.len();
        if seq_len == 0 {
            return;
        }
        assert_eq!(query.len(), self.head_dim);
        assert!(out_weights.len() >= seq_len);

        let norm_q = query.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let sc = self.curvature.sqrt();
        let s_q = if norm_q > 1e-12 {
            let sc_gamma_q = sc * self.gamma * norm_q;
            (1.0 / sc) * sc_gamma_q.tanh() / norm_q
        } else {
            0.0
        };

        let u_sq = s_q * s_q * norm_q * norm_q;
        let denom_u = (1.0 - self.curvature * u_sq).max(HYPERBOLIC_EPSILON);
        let lut_r_euc = &self.quantizer.lut_r_euc;
        let lut_denom_v = &self.quantizer.lut_denom_v;

        let mut max_logit = -1e9f32;

        for (j, q_key) in self.keys.iter().enumerate() {
            let bin = q_key.radial_bin as usize;
            let r_euc = lut_r_euc[bin];
            let v_sq = r_euc * r_euc;
            let denom_v = lut_denom_v[bin];

            let mut dot_q_code = 0.0f32;
            for (&q_val, &code) in query.iter().zip(q_key.angular_codes.iter()) {
                dot_q_code += q_val * (code as f32);
            }
            let dot_u_theta = s_q * dot_q_code * q_key.angular_scale;

            let diff_sq = (u_sq + v_sq - 2.0 * r_euc * dot_u_theta).max(0.0);
            let arg = (1.0 + (2.0 * self.curvature * diff_sq) / (denom_u * denom_v)).max(1.0);
            let arcosh_val = (arg + (arg * arg - 1.0).max(0.0).sqrt()).ln();
            let dist = arcosh_val / sc;

            let logit = -self.beta * dist;
            out_weights[j] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        let mut sum_exp = 0.0f32;
        for w in out_weights.iter_mut().take(seq_len) {
            let exp_val = (*w - max_logit).exp();
            *w = exp_val;
            sum_exp += exp_val;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for w in out_weights.iter_mut().take(seq_len) {
            *w *= inv_sum;
        }
    }
}

// =============================================================================
// 5. Configuration, Report, and Evaluator Harness
// =============================================================================

/// Runtime configuration for Hyperbolic Poincaré Attention.
#[derive(Clone, Debug)]
pub struct HyperbolicConfig {
    pub head_dim: usize,
    pub curvature: f32,
    pub max_radius: f32,
    pub radial_bits: u32,
    pub beta: f32,
    pub sequence_length: usize,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            curvature: DEFAULT_HYPERBOLIC_CURVATURE,
            max_radius: 6.0,
            radial_bits: 8,
            beta: 1.0,
            sequence_length: 128,
        }
    }
}

/// Verification report capturing performance and mathematical invariants of Phase 26.
#[derive(Clone, Debug)]
pub struct HyperbolicReport {
    pub curvature: f32,
    pub isometry_error: f32,
    pub mobius_identity_error: f32,
    pub mobius_inverse_error: f32,
    pub cosine_similarity: f32,
    pub relative_frobenius_error: f32,
    pub attention_entropy: f32,
    pub root_attention_mass: f32,
    pub query_latency_us: f32,
    pub compression_ratio: f32,
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator benchmark harness for Phase 26.
#[derive(Clone, Debug, Default)]
pub struct HyperbolicEvaluator {
    pub config: HyperbolicConfig,
}

impl HyperbolicEvaluator {
    pub fn new(config: HyperbolicConfig) -> Self {
        Self { config }
    }

    /// Runs full mathematical verification and empirical benchmarks.
    pub fn evaluate_performance(&self) -> HyperbolicReport {
        let d = self.config.head_dim;
        let c = self.config.curvature;
        let s = self.config.sequence_length;

        // 1. Isometry test between Poincare Ball and Lorentz Hyperboloid
        let u_test = PoincareVector::new(vec![0.04f32; d], c);
        let v_test = PoincareVector::new(vec![-0.03f32; d], c);
        let d_poincare = u_test.geodesic_distance(&v_test);

        let lx = u_test.to_lorentz();
        let ly = v_test.to_lorentz();
        let d_lorentz = lx.geodesic_distance(&ly);
        let isometry_error = (d_poincare - d_lorentz).abs();

        // 2. Mobius algebra axioms
        let zero = PoincareVector::new(vec![0.0f32; d], c);
        let u_add_zero = u_test.mobius_add(&zero);
        let mobius_identity_error = u_test
            .coords
            .iter()
            .zip(u_add_zero.coords.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let u_sub_u = u_test.mobius_sub(&u_test);
        let mobius_inverse_error = u_sub_u
            .coords
            .iter()
            .map(|&v| v.abs())
            .fold(0.0f32, f32::max);

        // 3. Compression and reconstruction across sequence
        let mut cache = HyperbolicAttentionCache::new(
            d,
            c,
            self.config.max_radius,
            self.config.radial_bits,
            self.config.beta,
        );

        let mut orig_keys = Vec::with_capacity(s);
        for t in 0..s {
            // Token 0: System Root (near origin)
            // Token 1..s: Hierarchical branches
            let depth = if t == 0 { 0 } else { ((t as f32 / 16.0).floor() as u8).min(8) + 1 };
            let mut key = vec![0.0f32; d];
            for (dim, k_val) in key.iter_mut().enumerate().take(d) {
                let t_f = t as f32;
                let dim_f = dim as f32;
                let scale = if t == 0 { 0.05 } else { 0.35 + 0.05 * (depth as f32) };
                *k_val = scale * (t_f * 0.035 + dim_f * 0.048).sin();
            }
            orig_keys.push(key.clone());
            cache.append_key(&key, depth);
        }

        let mut dot_prod = 0.0f64;
        let mut norm_o_sq = 0.0f64;
        let mut norm_r_sq = 0.0f64;
        let mut diff_sq = 0.0f64;

        for (t, orig) in orig_keys.iter().enumerate().take(s) {
            let recon = cache.reconstruct_key(t);
            for (dim, &orig_val) in orig.iter().enumerate().take(d) {
                let o = orig_val as f64;
                let r = recon[dim] as f64;
                dot_prod += o * r;
                norm_o_sq += o * o;
                norm_r_sq += r * r;
                let diff = o - r;
                diff_sq += diff * diff;
            }
        }

        let cosine_similarity = (dot_prod / (norm_o_sq.sqrt() * norm_r_sq.sqrt()).max(1e-12)) as f32;
        let relative_frobenius_error = (diff_sq.sqrt() / norm_o_sq.sqrt().max(1e-12)) as f32;

        // 4. Attention Routing & Attention Sink Evaluation
        // Broad query (near origin) querying hierarchical context
        let query = vec![0.02f32; d];
        let mut attn_weights = vec![0.0f32; s];

        let trials = 500;
        let t_start = Instant::now();
        for _ in 0..trials {
            cache.compute_hyperbolic_attention(&query, &mut attn_weights);
        }
        let elapsed = t_start.elapsed();
        let query_latency_us = (elapsed.as_nanos() as f32 / (trials as f32)) / 1000.0;

        // Entropy and root attention mass
        let mut entropy = 0.0f32;
        for &w in attn_weights.iter() {
            if w > 1e-12 {
                entropy -= w * w.ln();
            }
        }
        let root_attention_mass = attn_weights[0];

        // 5. Memory Footprint and Compression Ratio
        // Uncompressed FP16: s * d * 2 bytes
        let uncompressed_bytes = s * d * 2;
        // Compressed: per key: 1 (radial) + d (INT8) + 4 (scale) + 1 (depth) = d + 6 bytes
        let bytes_per_key = 1 + d + 4 + 1;
        let total_memory_footprint_bytes = s * bytes_per_key;
        let compression_ratio = (uncompressed_bytes as f32) / (total_memory_footprint_bytes as f32);

        HyperbolicReport {
            curvature: c,
            isometry_error,
            mobius_identity_error,
            mobius_inverse_error,
            cosine_similarity,
            relative_frobenius_error,
            attention_entropy: entropy,
            root_attention_mass,
            query_latency_us,
            compression_ratio,
            total_memory_footprint_bytes,
        }
    }
}

// =============================================================================
// 6. Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_inner_product_and_distance() {
        let x = LorentzVector::from_spacelike(vec![0.0, 0.0, 0.0], 1.0);
        let y = LorentzVector::from_spacelike(vec![1.0, 0.0, 0.0], 1.0);

        assert!((x.x0 - 1.0).abs() < 1e-6);
        assert!((x.lorentz_inner_product(&x) - (-1.0)).abs() < 1e-5);
        assert!(x.lorentz_inner_product(&y) < -1.0);

        let dist = x.geodesic_distance(&y);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_poincare_mobius_identity_and_inverse() {
        let u = PoincareVector::new(vec![0.2, -0.3, 0.1], 1.0);
        let zero = PoincareVector::new(vec![0.0, 0.0, 0.0], 1.0);

        let res = u.mobius_add(&zero);
        for i in 0..3 {
            assert!((res.coords[i] - u.coords[i]).abs() < 1e-6);
        }

        let diff = u.mobius_sub(&u);
        for i in 0..3 {
            assert!(diff.coords[i].abs() < 1e-5);
        }
    }

    #[test]
    fn test_isometry_poincare_lorentz() {
        let u = PoincareVector::new(vec![0.2, 0.1, 0.3], 1.0);
        let v = PoincareVector::new(vec![-0.1, 0.3, 0.0], 1.0);

        let d_poincare = u.geodesic_distance(&v);

        let x = u.to_lorentz();
        let y = v.to_lorentz();
        let d_lorentz = x.geodesic_distance(&y);

        assert!((d_poincare - d_lorentz).abs() < 1e-4);
    }

    #[test]
    fn test_quantization_reconstruction_fidelity() {
        let quantizer = HyperbolicPolarQuantizer::new(64, 1.0, 6.0, 8);
        let u = PoincareVector::new(vec![0.05; 64], 1.0);

        let q_key = quantizer.quantize(&u, 2);
        let u_hat = quantizer.dequantize(&q_key);

        let dist = u.geodesic_distance(&u_hat);
        assert!(dist < 0.15);
    }

    #[test]
    fn test_hyperbolic_attention_sink_and_routing() {
        let mut cache = HyperbolicAttentionCache::new(64, 1.0, 6.0, 8, 1.0);
        // Token 0: Root directive at origin
        let root_key = vec![0.001f32; 64];
        cache.append_key(&root_key, 0);

        // Tokens 1..10: Dispersed peripheral leaves
        for t in 1..10 {
            let mut leaf = vec![0.0f32; 64];
            leaf[(t * 3) % 64] = 0.8f32;
            cache.append_key(&leaf, 4);
        }

        let query = vec![0.005f32; 64]; // Query near root
        let mut weights = vec![0.0f32; 10];
        cache.compute_hyperbolic_attention(&query, &mut weights);

        // Root must receive dominant attention due to proximity to origin
        assert!(weights[0] > weights[1]);
        let sum_weights: f32 = weights.iter().sum();
        assert!((sum_weights - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_evaluator_report() {
        let evaluator = HyperbolicEvaluator::default();
        let report = evaluator.evaluate_performance();

        assert!(report.isometry_error < 1e-4);
        assert!(report.mobius_identity_error < 1e-5);
        assert!(report.mobius_inverse_error < 1e-5);
        assert!(report.cosine_similarity > 0.98);
        assert!(report.relative_frobenius_error < 0.15);
        assert!(report.query_latency_us < 25.0);
        assert!(report.compression_ratio > 1.8);
        assert!(report.total_memory_footprint_bytes < 30_000);
    }
}
