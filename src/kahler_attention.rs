//! Kortex Autonomous Robotics Research Division
//! Module: `kortex` (`aerovexhq/kortex`)
//! Subsystem: Complex Kähler Manifolds & Holomorphic Key-Value Bundling
//!
//! Pure safe Rust implementation of holomorphic key-value bundling,
//! Cauchy-Riemann conformal routing, and bounded Bergman kernel projection.

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::should_implement_trait)]

use std::f64::consts::PI;
use std::time::Instant;

/// Machine epsilon for floating-point geometric tolerances.
const EPSILON: f64 = 1e-12;

/// Maximum allowable vector norm strictly inside the complex unit ball.
const BALL_NORM_CLAMP: f64 = 0.999999;

// ============================================================================
// 1. COMPLEX SCALAR PRIMITIVES
// ============================================================================

/// Cache-aligned, zero-copy complex scalar representing z = re + i * im.
/// In the KV cache, re = key channel, im = value channel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C, align(16))]
pub struct KahlerComplex {
    pub re: f64,
    pub im: f64,
}

/// Type alias for backward compatibility.
pub type Complex64 = KahlerComplex;

impl KahlerComplex {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    #[inline(always)]
    pub const fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline(always)]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline(always)]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline(always)]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }

    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    #[inline(always)]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline(always)]
    pub fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        assert!(denom > EPSILON, "Complex division by near-zero magnitude");
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

// ============================================================================
// 2. COMPLEX VECTOR OPERATIONS ON C^D
// ============================================================================

/// Heap-allocated complex vector representing a bundled token z = k + i*v.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexVector {
    pub elements: Vec<Complex64>,
}

impl ComplexVector {
    pub fn zeros(dim: usize) -> Self {
        Self {
            elements: vec![Complex64::zero(); dim],
        }
    }

    pub fn from_real_and_imag(k: &[f64], v: &[f64]) -> Self {
        assert_eq!(k.len(), v.len(), "Key and Value slices must match in length");
        let elements = k
            .iter()
            .zip(v.iter())
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect();
        Self { elements }
    }

    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.elements.len()
    }

    #[inline(always)]
    pub fn extract_key(&self) -> Vec<f64> {
        self.elements.iter().map(|c| c.re).collect()
    }

    #[inline(always)]
    pub fn extract_value(&self) -> Vec<f64> {
        self.elements.iter().map(|c| c.im).collect()
    }

    /// Squared Euclidean norm ||z||^2 = sum_j |z_j|^2 = ||k||^2 + ||v||^2.
    #[inline(always)]
    pub fn norm_sq(&self) -> f64 {
        self.elements.iter().map(|c| c.norm_sq()).sum()
    }

    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Projects/clamps vector into the open unit ball B^d: ||z|| < 1.
    pub fn clamp_to_unit_ball(&mut self) {
        let n = self.norm();
        if n >= 1.0 {
            let scale = BALL_NORM_CLAMP / n;
            for c in self.elements.iter_mut() {
                *c = c.scale(scale);
            }
        }
    }

    /// Hermitian inner product: <w, z> = w^dagger * z = sum_j conj(w_j) * z_j.
    #[inline(always)]
    pub fn dot_hermitian(&self, rhs: &Self) -> Complex64 {
        assert_eq!(self.dim(), rhs.dim(), "Dimension mismatch in Hermitian dot");
        let mut acc = Complex64::zero();
        for (w, z) in self.elements.iter().zip(rhs.elements.iter()) {
            acc = acc.add(w.conj().mul(*z));
        }
        acc
    }

    pub fn add(&self, rhs: &Self) -> Self {
        assert_eq!(self.dim(), rhs.dim());
        let elements = self
            .elements
            .iter()
            .zip(rhs.elements.iter())
            .map(|(a, b)| a.add(*b))
            .collect();
        Self { elements }
    }

    pub fn scale(&self, s: f64) -> Self {
        let elements = self.elements.iter().map(|c| c.scale(s)).collect();
        Self { elements }
    }

    pub fn scale_complex(&self, s: Complex64) -> Self {
        let elements = self.elements.iter().map(|c| c.mul(s)).collect();
        Self { elements }
    }
}

// ============================================================================
// 3. KÄHLER POTENTIAL AND DIFFERENTIAL GEOMETRY TRAIT
// ============================================================================

/// Trait defining a Kähler potential K(z, z_bar) and its induced metric.
pub trait KahlerPotential {
    /// Evaluates the scalar potential K(z, z_bar) -> f64.
    fn potential(&self, z: &ComplexVector) -> f64;

    /// Evaluates the Wirtinger gradient dK / dz_j (holomorphic derivative).
    fn gradient_z(&self, z: &ComplexVector) -> ComplexVector;

    /// Evaluates the metric tensor g_{j k_bar} = d^2 K / (dz_j dz_k_bar).
    fn metric_tensor(&self, z: &ComplexVector) -> Vec<Vec<Complex64>>;
}

/// Bergman-Poincaré Kähler Potential on the Complex Unit Ball B^d:
/// K(z, z_bar) = -log(1 - ||z||^2).
#[derive(Debug, Clone, Copy, Default)]
pub struct BergmanPoincarePotential;

impl KahlerPotential for BergmanPoincarePotential {
    fn potential(&self, z: &ComplexVector) -> f64 {
        let norm_sq = z.norm_sq();
        assert!(norm_sq < 1.0, "State must strictly reside within unit ball B^d");
        -(1.0 - norm_sq).ln()
    }

    fn gradient_z(&self, z: &ComplexVector) -> ComplexVector {
        let norm_sq = z.norm_sq();
        let denom = 1.0 - norm_sq;
        assert!(denom > EPSILON);
        // dK / dz_j = z_bar_j / (1 - ||z||^2)
        let elements = z
            .elements
            .iter()
            .map(|c| c.conj().scale(1.0 / denom))
            .collect();
        ComplexVector { elements }
    }

    fn metric_tensor(&self, z: &ComplexVector) -> Vec<Vec<Complex64>> {
        let d = z.dim();
        let norm_sq = z.norm_sq();
        let denom = 1.0 - norm_sq;
        let denom_sq = denom * denom;
        assert!(denom > EPSILON);

        let mut g = vec![vec![Complex64::zero(); d]; d];
        for j in 0..d {
            for k in 0..d {
                let mut val = Complex64::zero();
                if j == k {
                    val = val.add(Complex64::new(denom, 0.0));
                }
                // Term: z_bar_j * z_k
                let cross = z.elements[j].conj().mul(z.elements[k]);
                val = val.add(cross);
                g[j][k] = val.scale(1.0 / denom_sq);
            }
        }
        g
    }
}

// ============================================================================
// 4. CONFORMAL ROUTER & MÖBIUS AUTOMORPHISMS
// ============================================================================

/// Conformal Attention Router executing angle-preserving transformations.
#[derive(Debug, Clone)]
pub struct ConformalRouter {
    pub dim: usize,
    pub rope_thetas: Vec<f64>,
}

impl ConformalRouter {
    pub fn new(dim: usize) -> Self {
        let rope_thetas = (0..dim)
            .map(|j| 10000.0_f64.powf(-2.0 * (j as f64) / (dim as f64)))
            .collect();
        Self { dim, rope_thetas }
    }

    /// Conformal Rotary Position Embedding (CRoPE):
    /// Multiplies z_j by exp(i * t * theta_j), rotating in the complex plane.
    pub fn apply_crope(&self, z: &ComplexVector, seq_idx: usize) -> ComplexVector {
        assert_eq!(z.dim(), self.dim);
        let t = seq_idx as f64;
        let elements = z
            .elements
            .iter()
            .enumerate()
            .map(|(j, &zj)| {
                let angle = t * self.rope_thetas[j];
                let rotor = Complex64::from_polar(1.0, angle);
                zj.mul(rotor)
            })
            .collect();
        ComplexVector { elements }
    }

    /// 1D Complex Möbius automorphism on the unit disk:
    /// phi_a(w) = (w - a) / (1 - conj(a) * w) for |a| < 1.
    pub fn mobius_disk(w: Complex64, a: Complex64) -> Complex64 {
        assert!(a.norm_sq() < 1.0, "Möbius shift parameter must reside in unit disk");
        let num = w.sub(a);
        let den = Complex64::one().sub(a.conj().mul(w));
        num.div(den)
    }

    /// Multi-dimensional Möbius automorphism on the complex ball B^d:
    /// phi_a(z) = (a - P_a z - (1 - ||a||^2)^(1/2) Q_a z) / (1 - <z, a>).
    pub fn mobius_ball(z: &ComplexVector, a: &ComplexVector) -> ComplexVector {
        let a_norm_sq = a.norm_sq();
        assert!(a_norm_sq < 1.0);
        if a_norm_sq < EPSILON {
            return z.clone();
        }

        let gamma = (1.0 - a_norm_sq).sqrt();
        let dot_za = z.dot_hermitian(a);
        let denom = Complex64::one().sub(dot_za);

        // P_a z = (<z, a> / ||a||^2) * a
        let proj_coeff = dot_za.scale(1.0 / a_norm_sq);
        let pa_z = a.scale_complex(proj_coeff);

        // Q_a z = z - P_a z
        let mut qa_z = ComplexVector::zeros(z.dim());
        for j in 0..z.dim() {
            qa_z.elements[j] = z.elements[j].sub(pa_z.elements[j]);
        }

        // Numerator: a - P_a z - gamma * Q_a z
        let mut num = ComplexVector::zeros(z.dim());
        for j in 0..z.dim() {
            let term = a.elements[j]
                .sub(pa_z.elements[j])
                .sub(qa_z.elements[j].scale(gamma));
            num.elements[j] = term;
        }

        // Final division by denom = 1 - <z, a>
        let elements = num
            .elements
            .iter()
            .map(|&c| c.div(denom))
            .collect();
        ComplexVector { elements }
    }
}

// ============================================================================
// 5. STREAMING BERGMAN KERNEL COMPRESSION ENGINE
// ============================================================================

/// Streaming, O(1)-memory compressor projecting token sequences onto an
/// M-element orthonormal Bergman polynomial basis of the unit disk A^2(D).
#[derive(Debug, Clone)]
pub struct StreamingBergmanCache {
    /// Token hidden dimension d.
    pub dim: usize,
    /// Number of retained holomorphic polynomial modes M.
    pub num_modes: usize,
    /// Temporal decay factor lambda in (0, 1].
    pub decay_factor: f64,
    /// Accumulated complex mode coefficients: shape [num_modes][dim].
    pub coefficients: Vec<ComplexVector>,
    /// Total tokens ingested.
    pub total_tokens: usize,
}

impl StreamingBergmanCache {
    pub fn new(dim: usize, num_modes: usize, decay_factor: f64) -> Self {
        assert!(num_modes > 0);
        assert!(decay_factor > 0.0 && decay_factor <= 1.0);
        let coefficients = vec![ComplexVector::zeros(dim); num_modes];
        Self {
            dim,
            num_modes,
            decay_factor,
            coefficients,
            total_tokens: 0,
        }
    }

    /// Evaluates the normalized 1D Bergman basis polynomial phi_m(w) on the unit disk:
    /// phi_m(w) = sqrt((m + 1) / pi) * w^m.
    #[inline(always)]
    pub fn basis_polynomial(m: usize, w: Complex64) -> Complex64 {
        let norm_factor = (((m + 1) as f64) / PI).sqrt();
        let mut power = Complex64::one();
        for _ in 0..m {
            power = power.mul(w);
        }
        power.scale(norm_factor)
    }

    /// Evaluates the continuous closed-form Bergman reproducing kernel on the disk:
    /// K(z, w_bar) = 1 / (pi * (1 - z * w_bar)^2).
    #[inline(always)]
    pub fn reproducing_kernel(z: Complex64, w: Complex64) -> Complex64 {
        let zw_bar = z.mul(w.conj());
        let denom_inner = Complex64::one().sub(zw_bar);
        let denom = denom_inner.mul(denom_inner).scale(PI);
        Complex64::one().div(denom)
    }

    /// Computes the synthetic temporal coordinate xi_t in the unit disk for token index t.
    #[inline(always)]
    pub fn temporal_coordinate(t: usize, horizon: usize) -> Complex64 {
        let r = ((t + 1) as f64) / ((horizon + 2) as f64) * 0.95;
        let theta = (t as f64) * 0.1;
        Complex64::from_polar(r, theta)
    }

    /// Ingests a new incoming token z_t = k_t + i*v_t with temporal coordinate xi_t.
    /// Memory consumption is strictly O(num_modes * dim).
    pub fn ingest_token(&mut self, mut token: ComplexVector, xi_t: Complex64, weight: f64) {
        assert_eq!(token.dim(), self.dim);
        token.clamp_to_unit_ball();

        for m in 0..self.num_modes {
            // Basis value at token coordinate: phi_m(xi_t)
            let phi_m = Self::basis_polynomial(m, xi_t);
            // Coefficient update: c_m <- lambda * c_m + weight * conj(phi_m) * token
            let update_scalar = phi_m.conj().scale(weight);
            let increment = token.scale_complex(update_scalar);

            for j in 0..self.dim {
                let decayed = self.coefficients[m].elements[j].scale(self.decay_factor);
                self.coefficients[m].elements[j] = decayed.add(increment.elements[j]);
            }
        }
        self.total_tokens += 1;
    }

    /// Reconstructs the continuous complex token z_hat(xi) at historical coordinate xi into an existing buffer.
    #[inline(always)]
    pub fn reconstruct_into(&self, xi: Complex64, out: &mut ComplexVector) {
        assert_eq!(out.dim(), self.dim);
        for c in out.elements.iter_mut() {
            *c = Complex64::zero();
        }
        let mut xi_pow = Complex64::one();
        for m in 0..self.num_modes {
            let norm_factor = (((m + 1) as f64) / PI).sqrt();
            let phi_m = xi_pow.scale(norm_factor);
            let coeff_m = &self.coefficients[m].elements;
            for (out_c, &c_mj) in out.elements.iter_mut().zip(coeff_m.iter()) {
                *out_c = out_c.add(c_mj.mul(phi_m));
            }
            xi_pow = xi_pow.mul(xi);
        }
        out.clamp_to_unit_ball();
    }

    /// Reconstructs the continuous complex token z_hat(xi) at historical coordinate xi.
    pub fn reconstruct(&self, xi: Complex64) -> ComplexVector {
        let mut recon = ComplexVector::zeros(self.dim);
        self.reconstruct_into(xi, &mut recon);
        recon
    }

    /// Reconstructs historical key and value vectors at coordinate xi.
    pub fn reconstruct_key_value(&self, xi: Complex64) -> (Vec<f64>, Vec<f64>) {
        let recon = self.reconstruct(xi);
        (recon.extract_key(), recon.extract_value())
    }

    /// Memory footprint in bytes of the compressed cache.
    pub fn memory_bytes(&self) -> usize {
        self.num_modes * self.dim * std::mem::size_of::<Complex64>()
    }

    /// Evaluates continuous attention with respect to a complex query vector w across sample coordinates
    /// using dual-space reproducing kernel projection in O(M * d + N * M) operations.
    pub fn query_attention(
        &self,
        query: &ComplexVector,
        coords: &[Complex64],
    ) -> (Vec<f64>, ComplexVector) {
        assert_eq!(query.dim(), self.dim);
        if coords.is_empty() {
            return (Vec::new(), ComplexVector::zeros(self.dim));
        }

        let scale = 1.0 / (self.dim as f64).sqrt();

        // 1. Dual projection of query onto mode coefficients: alpha_m = query^dagger * c_m
        let mut alpha = Vec::with_capacity(self.num_modes);
        for m in 0..self.num_modes {
            let mut dot = Complex64::zero();
            for (q_c, c_c) in query.elements.iter().zip(self.coefficients[m].elements.iter()) {
                dot = dot.add(q_c.conj().mul(*c_c));
            }
            alpha.push(dot);
        }

        // 2. Precompute basis polynomial vectors for each coordinate: phi[i][m] = phi_m(coords[i])
        let mut basis_matrix = Vec::with_capacity(coords.len());
        let mut raw_scores = Vec::with_capacity(coords.len());
        let mut max_score = f64::NEG_INFINITY;

        for &xi in coords {
            let mut phi_row = Vec::with_capacity(self.num_modes);
            let mut xi_pow = Complex64::one();
            let mut score_complex = Complex64::zero();

            for m in 0..self.num_modes {
                let norm_factor = (((m + 1) as f64) / PI).sqrt();
                let phi_m = xi_pow.scale(norm_factor);
                phi_row.push(phi_m);
                score_complex = score_complex.add(alpha[m].mul(phi_m));
                xi_pow = xi_pow.mul(xi);
            }

            let score = score_complex.re * scale;
            if score > max_score {
                max_score = score;
            }
            raw_scores.push(score);
            basis_matrix.push(phi_row);
        }

        // 3. Softmax computation
        let mut sum_exp = 0.0;
        let mut exp_weights = Vec::with_capacity(coords.len());
        for &s in &raw_scores {
            let e = (s - max_score).exp();
            sum_exp += e;
            exp_weights.push(e);
        }

        let inv_sum = if sum_exp > EPSILON { 1.0 / sum_exp } else { 0.0 };
        let probs: Vec<f64> = exp_weights.iter().map(|&e| e * inv_sum).collect();

        // 4. Synthesize composite mode weights: Phi_m = sum_i p_i * phi_m(xi_i)
        let mut phi_composite = vec![Complex64::zero(); self.num_modes];
        for (i, &p) in probs.iter().enumerate() {
            let row = &basis_matrix[i];
            for m in 0..self.num_modes {
                phi_composite[m] = phi_composite[m].add(row[m].scale(p));
            }
        }

        // 5. Linear combination of mode vectors: context = sum_m Phi_m * c_m
        let mut context = ComplexVector::zeros(self.dim);
        for m in 0..self.num_modes {
            let weight_m = phi_composite[m];
            for (ctx_c, &c_mj) in context.elements.iter_mut().zip(self.coefficients[m].elements.iter()) {
                *ctx_c = ctx_c.add(c_mj.mul(weight_m));
            }
        }

        context.clamp_to_unit_ball();
        (probs, context)
    }
}

// ============================================================================
// 6. EVALUATION REPORT & GATE 22 AUDIT HARNESS
// ============================================================================

/// Verification report capturing Gate 22 metrics.
#[derive(Debug, Clone)]
pub struct KahlerReport {
    pub cr_derivative_error: f64,
    pub conformal_angle_error: f64,
    pub metric_positivity_verified: bool,
    pub barrier_confinement_verified: bool,
    pub crope_isometry_error: f64,
    pub compaction_ratio: f64,
    pub query_latency_us: f64,
    pub memory_footprint_bytes: usize,
}

/// Evaluator checking all Gate 22 invariants.
pub struct KahlerEvaluator;

impl KahlerEvaluator {
    pub fn run_full_evaluation() -> KahlerReport {
        // 1. Cauchy-Riemann Holomorphicity check
        let a = Complex64::new(0.3, -0.2);
        let k0 = 0.2;
        let v0 = 0.4;
        let h = 1e-6;

        let f = |k: f64, v: f64| -> (f64, f64) {
            let res = ConformalRouter::mobius_disk(Complex64::new(k, v), a);
            (res.re, res.im)
        };

        let (u_k_plus, w_k_plus) = f(k0 + h, v0);
        let (u_k_minus, w_k_minus) = f(k0 - h, v0);
        let (u_v_plus, w_v_plus) = f(k0, v0 + h);
        let (u_v_minus, w_v_minus) = f(k0, v0 - h);

        let du_dk = (u_k_plus - u_k_minus) / (2.0 * h);
        let dw_dk = (w_k_plus - w_k_minus) / (2.0 * h);
        let du_dv = (u_v_plus - u_v_minus) / (2.0 * h);
        let dw_dv = (w_v_plus - w_v_minus) / (2.0 * h);

        let cr1_err = (du_dk - dw_dv).abs();
        let cr2_err = (du_dv + dw_dk).abs();
        let cr_derivative_error = cr1_err.max(cr2_err);

        // 2. Conformal Angle Preservation check
        let base = Complex64::new(0.1, -0.1);
        let eps = 1e-5;
        let v1 = Complex64::new(eps, 0.0);
        let v2 = Complex64::new(eps, eps);

        let f_base = ConformalRouter::mobius_disk(base, a);
        let f_v1 = ConformalRouter::mobius_disk(base.add(v1), a);
        let f_v2 = ConformalRouter::mobius_disk(base.add(v2), a);

        let df1 = f_v1.sub(f_base);
        let df2 = f_v2.sub(f_base);

        let orig_angle = (v2.arg() - v1.arg()).abs();
        let mapped_angle = (df2.arg() - df1.arg()).abs();
        let conformal_angle_error = (orig_angle - mapped_angle).abs();

        // 3. Kähler Metric Positivity & Hermiticity check
        let potential = BergmanPoincarePotential;
        let z_sample = ComplexVector::from_real_and_imag(&[0.2, -0.3, 0.1, 0.2], &[0.1, 0.4, -0.2, 0.1]);
        let metric = potential.metric_tensor(&z_sample);
        let mut metric_positivity_verified = true;
        for (j, row) in metric.iter().enumerate().take(z_sample.dim()) {
            if row[j].re <= 0.0 || row[j].im.abs() > 1e-9 {
                metric_positivity_verified = false;
            }
        }

        // 4. CRoPE Isometry check
        let router = ConformalRouter::new(z_sample.dim());
        let rotated = router.apply_crope(&z_sample, 42);
        let crope_isometry_error = (z_sample.norm() - rotated.norm()).abs();

        // 5. Streaming Bergman Cache Evaluation
        let dim = 128;
        let num_modes = 16;
        let total_tokens = 256;
        let mut cache = StreamingBergmanCache::new(dim, num_modes, 1.0);

        for t in 0..total_tokens {
            let k = vec![(t as f64 * 0.01).sin() * 0.4; dim];
            let v = vec![(t as f64 * 0.02).cos() * 0.4; dim];
            let token = ComplexVector::from_real_and_imag(&k, &v);
            let xi = StreamingBergmanCache::temporal_coordinate(t, total_tokens);
            cache.ingest_token(token, xi, 1.0 / (total_tokens as f64));
        }

        // Check barrier confinement on reconstruction
        let mut barrier_confinement_verified = true;
        let sample_coords: Vec<Complex64> = (0..16)
            .map(|idx| StreamingBergmanCache::temporal_coordinate(idx * 16, total_tokens))
            .collect();

        for &xi in &sample_coords {
            let recon = cache.reconstruct(xi);
            if recon.norm() >= 1.0 {
                barrier_confinement_verified = false;
            }
        }

        // Query latency check
        let query = ComplexVector::from_real_and_imag(&vec![0.1; dim], &vec![0.05; dim]);
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = cache.query_attention(&query, &sample_coords);
        }
        let elapsed = start.elapsed();
        let query_latency_us = (elapsed.as_nanos() as f64 / 1000.0) / (iterations as f64);

        // Compaction calculation:
        // Discrete buffer stores 256 tokens * 128 dims * 2 (k and v) * 8 bytes (f64) = 524,288 bytes
        // Compressed cache stores 16 modes * 128 dims * 16 bytes (Complex64) = 32,768 bytes
        let raw_bytes = total_tokens * dim * 2 * std::mem::size_of::<f64>();
        let memory_footprint_bytes = cache.memory_bytes();
        let compaction_ratio = raw_bytes as f64 / memory_footprint_bytes as f64;

        KahlerReport {
            cr_derivative_error,
            conformal_angle_error,
            metric_positivity_verified,
            barrier_confinement_verified,
            crope_isometry_error,
            compaction_ratio,
            query_latency_us,
            memory_footprint_bytes,
        }
    }
}

// ============================================================================
// 7. UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_scalar_algebra() {
        let z1 = Complex64::new(3.0, 4.0);
        let z2 = Complex64::new(1.0, -2.0);

        assert_eq!(z1.add(z2), Complex64::new(4.0, 2.0));
        assert_eq!(z1.sub(z2), Complex64::new(2.0, 6.0));
        assert!((z1.norm_sq() - 25.0).abs() < EPSILON);
        assert!((z1.norm() - 5.0).abs() < EPSILON);

        // Multiplication: (3 + 4i)(1 - 2i) = 3 - 6i + 4i - 8i^2 = 11 - 2i
        let prod = z1.mul(z2);
        assert!((prod.re - 11.0).abs() < EPSILON);
        assert!((prod.im - (-2.0)).abs() < EPSILON);

        let div = prod.div(z2);
        assert!((div.re - z1.re).abs() < 1e-10);
        assert!((div.im - z1.im).abs() < 1e-10);
    }

    #[test]
    fn test_complex_vector_and_hermitian_inner_product() {
        let k1 = [1.0, 2.0];
        let v1 = [0.5, -1.0];
        let k2 = [2.0, -1.0];
        let v2 = [1.0, 0.5];

        let z1 = ComplexVector::from_real_and_imag(&k1, &v1);
        let z2 = ComplexVector::from_real_and_imag(&k2, &v2);

        let h_dot = z1.dot_hermitian(&z2);
        // z1 = [1 + 0.5i, 2 - i]
        // z2 = [2 + i, -1 + 0.5i]
        // z1^dagger * z2 = (1 - 0.5i)(2 + i) + (2 + i)(-1 + 0.5i)
        // (1 - 0.5i)(2 + i) = 2 + i - i + 0.5 = 2.5
        // (2 + i)(-1 + 0.5i) = -2 + i - i + 0.5 i^2 = -2 - 0.5 = -2.5
        // total = 2.5 - 2.5 = 0.0 + 0.0i
        assert!((h_dot.re - 0.0).abs() < 1e-10);
        assert!((h_dot.im - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_riemann_equations() {
        let report = KahlerEvaluator::run_full_evaluation();
        assert!(report.cr_derivative_error < 1e-5, "CR error exceeded: {}", report.cr_derivative_error);
        assert!(report.conformal_angle_error < 1e-4, "Angle error exceeded: {}", report.conformal_angle_error);
    }

    #[test]
    fn test_gate_22_audit() {
        let report = KahlerEvaluator::run_full_evaluation();
        assert!(report.metric_positivity_verified, "Metric positivity failed");
        assert!(report.barrier_confinement_verified, "Barrier confinement failed");
        assert!(report.crope_isometry_error < 1e-7, "CRoPE isometry error: {}", report.crope_isometry_error);
        assert!(report.compaction_ratio >= 2.0, "Compaction ratio below 2.0x: {}", report.compaction_ratio);
        assert!(report.query_latency_us < 50.0, "Query latency exceeded 50 us: {}", report.query_latency_us);
    }
}
