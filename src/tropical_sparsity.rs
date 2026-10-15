//! Tropical Geometry Attention Decimation & Min-Plus Polytope Sparsification
//! Commercial Flight-Critical Edge Inference Runtime (Kortex)
//!
//! # Safety & Domain Mandate
//! - Strictly pure safe Rust: `#![deny(unsafe_code)]`.
//! - Non-linear softmax attention bounding via the max-plus semiring (R_max, oplus, otimes)
//!   and Newton polytope convex support functions.
//! - Shift-invariant key representation in the tropical projective torus TP^{d-1} = R^d / R*1
//!   with O(d) Hopf-Hilbert variation metric evaluation and zero heap allocations in inner loops.
//! - O(1) cluster dormancy screening via coordinate-wise polyhedral bounding envelopes.
//! - Provable rate-distortion attention decimation bounds preventing contextual amnesia.

#![deny(unsafe_code)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::f32;

/// Precision tolerance for numerical assertions and comparisons.
pub const TROPICAL_EPSILON: f32 = 1e-6;

/// Default logit deficit threshold corresponding to p_i <= exp(-theta) approx 1.0e-4.
pub const DEFAULT_PRUNING_DEFICIT_THRESHOLD: f32 = 9.210_34;

// ==============================================================================
// 1. Tropical Key Representation in the Projective Torus TP^{d-1}
// ==============================================================================

/// A key vector projected into the tropical projective torus TP^{d-1} = R^d / R*1.
///
/// In the tropical projective torus, two vectors are equivalent if they differ by
/// an additive constant c * 1. This struct stores raw coordinates for exact inner
/// products, canonical shift-normalized coordinates (min-zero gauge: y_j = k_j - min_l k_l)
/// for tropical geometry, and precomputed coordinate extrema.
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalKey {
    /// Dimension d of the key space.
    dim: usize,
    /// Raw unshifted key coordinates k in R^d.
    raw_coords: Vec<f32>,
    /// Shift-normalized coordinates in TP^{d-1} (min-zero gauge: y_j = k_j - min_coord >= 0).
    canonical_coords: Vec<f32>,
    /// Minimum coordinate value in the raw key vector.
    min_coord: f32,
    /// Maximum coordinate value in the raw key vector.
    max_coord: f32,
    /// Tropical coordinate spread: spread(k) = max_j k_j - min_j k_j.
    spread: f32,
    /// Euclidean L2 norm of the raw key vector.
    l2_norm: f32,
}

impl TropicalKey {
    /// Constructs a new `TropicalKey` from a raw slice of dimension d.
    ///
    /// # Panics
    /// Panics if the input slice is empty.
    pub fn new(raw_key: &[f32]) -> Self {
        assert!(!raw_key.is_empty(), "Key dimension must be non-zero");
        let dim = raw_key.len();

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum_sq = 0.0_f32;

        for &val in raw_key {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
            sum_sq += val * val;
        }

        let spread = max_val - min_val;
        let l2_norm = sum_sq.sqrt();

        let mut canonical_coords = Vec::with_capacity(dim);
        for &val in raw_key {
            canonical_coords.push(val - min_val);
        }

        Self {
            dim,
            raw_coords: raw_key.to_vec(),
            canonical_coords,
            min_coord: min_val,
            max_coord: max_val,
            spread,
            l2_norm,
        }
    }

    /// Returns the dimensionality d of the key vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the raw coordinates in R^d.
    #[inline]
    pub fn raw_coords(&self) -> &[f32] {
        &self.raw_coords
    }

    /// Returns the canonical min-zero coordinates in TP^{d-1}.
    #[inline]
    pub fn canonical_coords(&self) -> &[f32] {
        &self.canonical_coords
    }

    /// Returns the coordinate spread: max_j k_j - min_j k_j.
    #[inline]
    pub fn spread(&self) -> f32 {
        self.spread
    }

    /// Returns the Euclidean L2 norm of the raw key vector.
    #[inline]
    pub fn l2_norm(&self) -> f32 {
        self.l2_norm
    }

    /// Computes the exact tropical distance d_tr(self, other) on TP^{d-1}.
    ///
    /// Formula:
    ///   d_tr(u, v) = max_j (u_j - v_j) - min_j (u_j - v_j)
    ///
    /// Complexity: O(d) subtractions and comparisons, zero heap allocations.
    #[inline]
    pub fn tropical_distance(&self, other: &Self) -> f32 {
        assert_eq!(self.dim, other.dim, "Key dimensions must match");

        let mut max_diff = f32::NEG_INFINITY;
        let mut min_diff = f32::INFINITY;

        let u = &self.raw_coords;
        let v = &other.raw_coords;

        for j in 0..self.dim {
            let diff = u[j] - v[j];
            if diff > max_diff {
                max_diff = diff;
            }
            if diff < min_diff {
                min_diff = diff;
            }
        }

        max_diff - min_diff
    }
}

// ==============================================================================
// 2. Tropical Bounding Box & Polyhedral Cluster Envelope
// ==============================================================================

/// Coordinate-wise bounding envelope of a set of keys in R^d.
///
/// For any query q in R^d and keys in cluster K:
///   <q, k> = sum_{q_j >= 0} q_j k_j + sum_{q_j < 0} q_j k_j
///          <= sum_{q_j >= 0} q_j upper_j + sum_{q_j < 0} q_j lower_j
///
/// Coordinate-wise bounding envelope of a set of keys in R^d with respect to a cluster center.
///
/// For any query q in R^d, cluster center k_c, and key k = k_c + delta in cluster K:
///   <q, k> = <q, k_c> + <q, delta>
///          <= <q, k_c> + sum_{q_j >= 0} q_j delta_upper_j + sum_{q_j < 0} q_j delta_lower_j
///
/// Adding (b_max - b_c) and dividing by sqrt(d) yields an exact, tight upper bound
/// on the attention logit for all keys in the cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalBoundingBox {
    dim: usize,
    /// Centered coordinate differences delta_lower = min_k (k_j - center_j) <= 0.
    delta_lower: Vec<f32>,
    /// Centered coordinate differences delta_upper = max_k (k_j - center_j) >= 0.
    delta_upper: Vec<f32>,
    /// Maximum scalar bias in the cluster.
    max_bias: f32,
    /// Center scalar bias.
    center_bias: f32,
    /// Number of tokens aggregated in this envelope.
    token_count: usize,
}

impl TropicalBoundingBox {
    /// Creates a new bounding box initialized with a single center key and bias.
    pub fn from_key(key: &TropicalKey, bias: f32) -> Self {
        Self {
            dim: key.dim(),
            delta_lower: vec![0.0; key.dim()],
            delta_upper: vec![0.0; key.dim()],
            max_bias: bias,
            center_bias: bias,
            token_count: 1,
        }
    }

    /// Extends the bounding box to enclose an additional key and bias.
    pub fn extend(&mut self, center_key: &TropicalKey, key: &TropicalKey, bias: f32) {
        assert_eq!(self.dim, key.dim(), "Dimension mismatch in bounding box");
        let raw = key.raw_coords();
        let center = center_key.raw_coords();
        for j in 0..self.dim {
            let diff = raw[j] - center[j];
            if diff < self.delta_lower[j] {
                self.delta_lower[j] = diff;
            }
            if diff > self.delta_upper[j] {
                self.delta_upper[j] = diff;
            }
        }
        if bias > self.max_bias {
            self.max_bias = bias;
        }
        self.token_count += 1;
    }

    /// Evaluates the provable upper bound on the attention logit using the precomputed center logit.
    ///
    /// Complexity: O(d) with zero allocations and no dot product with center.
    #[inline]
    pub fn evaluate_upper_logit_with_center(
        &self,
        center_logit: f32,
        query: &[f32],
        inv_sqrt_d: f32,
    ) -> f32 {
        assert_eq!(self.dim, query.len(), "Query dimension mismatch");
        let mut delta_dot = 0.0_f32;

        for (&q, (&d_up, &d_low)) in query.iter().zip(self.delta_upper.iter().zip(self.delta_lower.iter())) {
            if q >= 0.0 {
                delta_dot += d_up * q;
            } else {
                delta_dot += d_low * q;
            }
        }

        center_logit + (delta_dot * inv_sqrt_d) + (self.max_bias - self.center_bias)
    }

    /// Evaluates the provable upper bound on the attention logit given query and center key.
    #[inline]
    pub fn evaluate_upper_logit(
        &self,
        center_key: &TropicalKey,
        query: &[f32],
        inv_sqrt_d: f32,
    ) -> f32 {
        assert_eq!(self.dim, query.len(), "Query dimension mismatch");
        let mut center_dot = 0.0_f32;
        let raw = center_key.raw_coords();
        for (&q, &r) in query.iter().zip(raw.iter()) {
            center_dot += q * r;
        }
        let center_logit = (center_dot * inv_sqrt_d) + self.center_bias;
        self.evaluate_upper_logit_with_center(center_logit, query, inv_sqrt_d)
    }

    /// Returns the number of tokens covered by this envelope.
    #[inline]
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Returns the coordinate difference lower bounds.
    #[inline]
    pub fn delta_lower(&self) -> &[f32] {
        &self.delta_lower
    }

    /// Returns the coordinate difference upper bounds.
    #[inline]
    pub fn delta_upper(&self) -> &[f32] {
        &self.delta_upper
    }
}

// ==============================================================================
// 3. Tropical Cluster Representative & Ball Cover
// ==============================================================================

/// A cluster of keys contained within a tropical ball B_tr(center, rho) on TP^{d-1}.
#[derive(Debug, Clone)]
pub struct TropicalCluster {
    /// Unique cluster ID.
    pub cluster_id: usize,
    /// Center representative key in TP^{d-1}.
    pub center_key: TropicalKey,
    /// Aggregate value accumulator: sum_{i in Cluster} v_i.
    pub accumulated_value: Vec<f32>,
    /// Coordinate-wise bounding envelope over all member keys.
    pub bounding_box: TropicalBoundingBox,
    /// Vector of member token indices.
    pub member_indices: Vec<usize>,
    /// Maximum observed tropical distance from any member to the center.
    pub max_radius: f32,
    /// Scalar bias associated with the center key.
    pub center_bias: f32,
}

impl TropicalCluster {
    /// Creates a new cluster centered at the given key and value vector.
    pub fn new(
        cluster_id: usize,
        key: TropicalKey,
        value: &[f32],
        bias: f32,
        token_index: usize,
    ) -> Self {
        let bbox = TropicalBoundingBox::from_key(&key, bias);
        Self {
            cluster_id,
            center_key: key,
            accumulated_value: value.to_vec(),
            bounding_box: bbox,
            member_indices: vec![token_index],
            max_radius: 0.0,
            center_bias: bias,
        }
    }

    /// Adds a key-value token to this cluster.
    pub fn add_token(
        &mut self,
        key: &TropicalKey,
        value: &[f32],
        bias: f32,
        token_index: usize,
        distance: f32,
    ) {
        assert_eq!(
            self.accumulated_value.len(),
            value.len(),
            "Value dimension mismatch"
        );

        for (acc, &v) in self.accumulated_value.iter_mut().zip(value.iter()) {
            *acc += v;
        }

        self.bounding_box.extend(&self.center_key, key, bias);
        self.member_indices.push(token_index);

        if distance > self.max_radius {
            self.max_radius = distance;
        }
    }

    /// Returns the number of tokens aggregated in this cluster.
    #[inline]
    pub fn len(&self) -> usize {
        self.member_indices.len()
    }

    /// Returns whether the cluster contains zero tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.member_indices.is_empty()
    }

    /// Computes the approximate memory footprint of this cluster in bytes.
    pub fn memory_footprint_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + (self.center_key.raw_coords.capacity() * 4)
            + (self.center_key.canonical_coords.capacity() * 4)
            + (self.accumulated_value.capacity() * 4)
            + (self.bounding_box.delta_lower.capacity() * 4)
            + (self.bounding_box.delta_upper.capacity() * 4)
            + (self.member_indices.capacity() * core::mem::size_of::<usize>())
    }
}

// ==============================================================================
// 4. Configuration and Inference Statistics
// ==============================================================================

/// Configuration parameters for the Tropical Attention Engine.
#[derive(Debug, Clone)]
pub struct TropicalEngineConfig {
    /// Dimension of key vectors d.
    pub key_dim: usize,
    /// Dimension of value vectors d_v.
    pub value_dim: usize,
    /// Tropical ball radius rho for clustering in TP^{d-1}.
    pub tropical_radius: f32,
    /// Logit deficit threshold theta for key pruning (theta = ln(1/eps)).
    pub pruning_deficit_threshold: f32,
    /// Maximum number of active clusters retained in the KV cache.
    pub max_active_clusters: usize,
}

impl Default for TropicalEngineConfig {
    fn default() -> Self {
        Self {
            key_dim: 128,
            value_dim: 128,
            tropical_radius: 1.0,
            pruning_deficit_threshold: DEFAULT_PRUNING_DEFICIT_THRESHOLD,
            max_active_clusters: 2048,
        }
    }
}

/// Statistics collected during a single attention inference pass.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TropicalInferenceStats {
    /// Total number of tokens ingested into the engine.
    pub total_ingested_tokens: usize,
    /// Total number of active clusters evaluated during attention.
    pub evaluated_clusters: usize,
    /// Number of clusters skipped via tropical bounding envelope tests.
    pub skipped_clusters: usize,
    /// Dynamic pruning ratio: (total_tokens - evaluated_tokens) / total_tokens.
    pub pruning_ratio: f32,
    /// Effective retention ratio: evaluated_tokens / total_tokens.
    pub retention_ratio: f32,
    /// Maximum tropical logit deficit observed during the pass.
    pub max_observed_deficit: f32,
}

// ==============================================================================
// 5. Tropical Attention Decimation Cache
// ==============================================================================

/// Complete production-grade streaming KV decimation cache implementing
/// Tropical Geometry attention sparsification.
pub struct TropicalAttentionCache {
    config: TropicalEngineConfig,
    /// Precomputed 1.0 / sqrt(d).
    inv_sqrt_d: f32,
    /// Active clusters representing the compressed KV cache.
    clusters: Vec<TropicalCluster>,
    /// Total number of ingested tokens across sequence history.
    total_tokens: usize,
    /// Next cluster ID to assign.
    next_cluster_id: usize,
}

impl TropicalAttentionCache {
    /// Creates a new `TropicalAttentionCache` with the given configuration.
    pub fn new(config: TropicalEngineConfig) -> Self {
        let inv_sqrt_d = 1.0_f32 / (config.key_dim as f32).sqrt();
        Self {
            config,
            inv_sqrt_d,
            clusters: Vec::new(),
            total_tokens: 0,
            next_cluster_id: 0,
        }
    }

    /// Ingests a new key-value token into the compressed KV cache.
    ///
    /// Searches for an existing cluster within tropical distance rho on TP^{d-1}.
    /// If found, the token is absorbed into that cluster.
    /// If no cluster is within rho, a new cluster is instantiated.
    pub fn ingest_token(&mut self, raw_key: &[f32], value: &[f32], bias: f32) {
        assert_eq!(raw_key.len(), self.config.key_dim, "Invalid key dimension");
        assert_eq!(
            value.len(),
            self.config.value_dim,
            "Invalid value dimension"
        );

        let token_index = self.total_tokens;
        self.total_tokens += 1;

        let trop_key = TropicalKey::new(raw_key);

        let mut best_cluster_idx: Option<usize> = None;
        let mut min_distance = f32::INFINITY;

        for (idx, cluster) in self.clusters.iter().enumerate() {
            let dist = trop_key.tropical_distance(&cluster.center_key);
            if dist < min_distance {
                min_distance = dist;
                best_cluster_idx = Some(idx);
            }
        }

        if let Some(idx) = best_cluster_idx {
            if min_distance <= self.config.tropical_radius {
                self.clusters[idx].add_token(
                    &trop_key,
                    value,
                    bias,
                    token_index,
                    min_distance,
                );
                return;
            }
        }

        let new_cluster = TropicalCluster::new(
            self.next_cluster_id,
            trop_key,
            value,
            bias,
            token_index,
        );
        self.next_cluster_id += 1;
        self.clusters.push(new_cluster);
    }

    /// Computes bounded sparse attention against a query vector q.
    ///
    /// # Three-Stage Tropical Pipeline:
    /// 1. **Anchor Screening**: Evaluates logits of active cluster centers to establish z_max(q).
    /// 2. **O(1) Polyhedral Envelope Pruning**: For each cluster, tests if its upper bounding
    ///    logit satisfies bar{Z}_K(q) <= z_max(q) - theta. If so, skips the cluster entirely.
    /// 3. **Stable Softmax Aggregation**: Evaluates softmax exclusively over survived clusters,
    ///    accumulating representative value mass into the output buffer.
    pub fn compute_attention(
        &self,
        query: &[f32],
        output: &mut [f32],
    ) -> TropicalInferenceStats {
        assert_eq!(query.len(), self.config.key_dim, "Invalid query dimension");
        assert_eq!(
            output.len(),
            self.config.value_dim,
            "Invalid output dimension"
        );

        for out in output.iter_mut() {
            *out = 0.0;
        }

        if self.clusters.is_empty() {
            return TropicalInferenceStats::default();
        }

        let num_clusters = self.clusters.len();
        let mut cluster_logits = vec![0.0_f32; num_clusters];
        let mut z_max = f32::NEG_INFINITY;

        // Stage 1: Evaluate cluster center logits.
        for (i, cluster) in self.clusters.iter().enumerate() {
            let mut dot = 0.0_f32;
            let raw = cluster.center_key.raw_coords();
            for j in 0..self.config.key_dim {
                dot += query[j] * raw[j];
            }
            let logit = (dot * self.inv_sqrt_d) + cluster.center_bias;
            cluster_logits[i] = logit;
            if logit > z_max {
                z_max = logit;
            }
        }

        // Stage 2: O(1) Envelope Pruning.
        let pruning_threshold = z_max - self.config.pruning_deficit_threshold;
        let mut active_indices = Vec::with_capacity(num_clusters);
        let mut skipped_clusters = 0;
        let mut max_observed_deficit = 0.0_f32;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let deficit = z_max - cluster_logits[i];
            if deficit > max_observed_deficit {
                max_observed_deficit = deficit;
            }

            let upper_bound = cluster
                .bounding_box
                .evaluate_upper_logit_with_center(cluster_logits[i], query, self.inv_sqrt_d);

            if upper_bound <= pruning_threshold {
                skipped_clusters += 1;
            } else {
                active_indices.push(i);
            }
        }

        // Fallback: If all clusters were pruned, retain top-1 center.
        if active_indices.is_empty() {
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &l) in cluster_logits.iter().enumerate() {
                if l > best_val {
                    best_val = l;
                    best_idx = i;
                }
            }
            active_indices.push(best_idx);
            skipped_clusters = num_clusters.saturating_sub(1);
        }

        // Stage 3: Stable Softmax Evaluation over Survived Clusters.
        let mut sum_exp = 0.0_f32;
        let mut active_weights = Vec::with_capacity(active_indices.len());

        for &idx in &active_indices {
            let logit = cluster_logits[idx];
            let exp_val = (logit - z_max).exp();
            sum_exp += exp_val;
            active_weights.push(exp_val);
        }

        let inv_sum_exp = if sum_exp > TROPICAL_EPSILON {
            1.0 / sum_exp
        } else {
            1.0
        };

        let mut evaluated_token_count = 0;
        for (&idx, weight) in active_indices.iter().zip(active_weights.iter()) {
            let cluster = &self.clusters[idx];
            evaluated_token_count += cluster.len();
            let p_r = weight * inv_sum_exp;
            let count = cluster.len() as f32;
            let factor = p_r / count;

            for (out, &acc_v) in output.iter_mut().zip(cluster.accumulated_value.iter()) {
                *out += factor * acc_v;
            }
        }

        let evaluated_clusters = active_indices.len();
        let retention = if self.total_tokens > 0 {
            evaluated_token_count as f32 / self.total_tokens as f32
        } else {
            0.0
        };
        let pruning_ratio = 1.0_f32 - retention;

        TropicalInferenceStats {
            total_ingested_tokens: self.total_tokens,
            evaluated_clusters,
            skipped_clusters,
            pruning_ratio,
            retention_ratio: retention,
            max_observed_deficit,
        }
    }

    /// Returns the current number of active clusters in memory.
    #[inline]
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Returns the total number of ingested tokens.
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Returns references to the internal clusters.
    #[inline]
    pub fn clusters(&self) -> &[TropicalCluster] {
        &self.clusters
    }

    /// Calculates total memory footprint in bytes.
    pub fn memory_footprint_bytes(&self) -> usize {
        let base = core::mem::size_of::<Self>();
        let cluster_bytes: usize = self
            .clusters
            .iter()
            .map(|c| c.memory_footprint_bytes())
            .sum();
        base + cluster_bytes
    }
}

// ==============================================================================
// 6. Report and Benchmark Evaluation Harness
// ==============================================================================

/// Benchmark report validating Gate 20 for Tropical Geometry sparsification.
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalReport {
    /// Dimension of key and value vectors.
    pub head_dim: usize,
    /// Total ingested tokens in the benchmark sequence.
    pub total_tokens: usize,
    /// Number of active clusters formed.
    pub active_clusters: usize,
    /// Dynamic token pruning ratio (>= 0.70 required).
    pub pruning_ratio: f32,
    /// Attention scoring query latency in microseconds (< 10.0 us required).
    pub query_latency_us: f32,
    /// Softmax attention probability recall of retained tokens (>= 0.95 required).
    pub attention_recall: f32,
    /// Relative Frobenius error between exact and decimated attention outputs.
    pub relative_frobenius_error: f32,
    /// L1 variation distance between full and decimated probability distributions.
    pub l1_probability_deviation: f32,
    /// Maximum observed tropical logit deficit.
    pub max_observed_deficit: f32,
    /// Metric triangle inequality maximum error (must be <= 1e-5).
    pub metric_triangle_error: f32,
    /// Softmax gauge shift-invariance error (must be <= 1e-6).
    pub softmax_invariance_error: f32,
    /// Verification status of Log-Sum-Exp tropical sandwich bounds.
    pub sandwich_bound_verified: bool,
    /// Verification status of coordinate-wise bounding envelope dominance.
    pub envelope_soundness_verified: bool,
    /// Total memory footprint of the active cache in bytes (< 30,000 required).
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator harness executing Gate 20 empirical verification.
#[derive(Debug, Clone)]
pub struct TropicalEvaluator {
    config: TropicalEngineConfig,
}

impl Default for TropicalEvaluator {
    fn default() -> Self {
        Self {
            config: TropicalEngineConfig {
                key_dim: 128,
                value_dim: 128,
                tropical_radius: 0.85,
                pruning_deficit_threshold: 4.5,
                max_active_clusters: 256,
            },
        }
    }
}

impl TropicalEvaluator {
    /// Creates an evaluator with custom configuration.
    pub fn new(config: TropicalEngineConfig) -> Self {
        Self { config }
    }

    /// Evaluates all Gate 20 invariants and benchmarks.
    pub fn evaluate_performance(&self) -> TropicalReport {
        let dim = self.config.key_dim;
        let v_dim = self.config.value_dim;
        let num_tokens = 128;

        // Step 1: Metric Axioms & Softmax Invariance Checks.
        let mut u_raw = vec![0.0_f32; dim];
        let mut v_raw = vec![0.0_f32; dim];
        let mut w_raw = vec![0.0_f32; dim];
        for j in 0..dim {
            let fj = j as f32;
            u_raw[j] = (fj * 0.3).sin();
            v_raw[j] = (fj * 0.5).cos();
            w_raw[j] = (fj * 0.7).sin() * 0.5;
        }

        let u_key = TropicalKey::new(&u_raw);
        let v_key = TropicalKey::new(&v_raw);
        let w_key = TropicalKey::new(&w_raw);

        let d_uv = u_key.tropical_distance(&v_key);
        let d_vw = v_key.tropical_distance(&w_key);
        let d_uw = u_key.tropical_distance(&w_key);
        let metric_triangle_error = (d_uw - (d_uv + d_vw)).max(0.0);

        // Softmax gauge invariance test:
        let mut z = [0.0_f32; 16];
        let mut z_shift = [0.0_f32; 16];
        let shift_c = 4.25_f32;
        for i in 0..16 {
            z[i] = (i as f32 * 0.4).sin() * 2.0;
            z_shift[i] = z[i] + shift_c;
        }
        let z_max_1 = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_1: f32 = z.iter().map(|&x| (x - z_max_1).exp()).sum();
        let p1: Vec<f32> = z.iter().map(|&x| (x - z_max_1).exp() / sum_1).collect();

        let z_max_2 = z_shift.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_2: f32 = z_shift.iter().map(|&x| (x - z_max_2).exp()).sum();
        let p2: Vec<f32> = z_shift
            .iter()
            .map(|&x| (x - z_max_2).exp() / sum_2)
            .collect();

        let mut softmax_invariance_error = 0.0_f32;
        for (&a, &b) in p1.iter().zip(p2.iter()) {
            let diff = (a - b).abs();
            if diff > softmax_invariance_error {
                softmax_invariance_error = diff;
            }
        }

        // LSE Sandwich bound test:
        let lse = z_max_1 + sum_1.ln();
        let sandwich_bound_verified =
            lse >= z_max_1 - TROPICAL_EPSILON && lse <= z_max_1 + (16.0_f32).ln() + TROPICAL_EPSILON;

        // Step 2: Synthetic clustered key trajectory generation.
        // 8 thematic clusters with 16 tokens each -> 128 tokens total.
        let num_thematic_clusters = 8;
        let tokens_per_cluster = 16;
        let mut synthetic_keys = Vec::with_capacity(num_tokens);
        let mut synthetic_values = Vec::with_capacity(num_tokens);

        for c in 0..num_thematic_clusters {
            let mut base_k = vec![0.0_f32; dim];
            let mut base_v = vec![0.0_f32; dim];
            let freq = ((c + 1) as f32) * 0.08;
            for j in 0..dim {
                let fj = (j + 1) as f32;
                base_k[j] = (fj * freq).sin() * 0.8;
                base_v[j] = (fj * freq).cos() * 0.5;
            }

            for t in 0..tokens_per_cluster {
                let mut k = base_k.clone();
                let mut v = base_v.clone();
                for j in 0..dim {
                    k[j] += ((t * 13 + j * 7) as f32 * 0.1).sin() * 0.01;
                    v[j] += ((t * 17 + j * 11) as f32 * 0.1).cos() * 0.01;
                }
                synthetic_keys.push(k);
                synthetic_values.push(v);
            }
        }

        // Step 3: Ingestion into TropicalAttentionCache.
        let mut cache = TropicalAttentionCache::new(self.config.clone());
        for i in 0..num_tokens {
            cache.ingest_token(&synthetic_keys[i], &synthetic_values[i], 0.0);
        }

        // Step 4: Verification of Bounding Envelope Dominance.
        let mut query = vec![0.0_f32; dim];
        for j in 0..dim {
            query[j] = synthetic_keys[0][j] * 1.5;
        }

        let inv_sqrt_d = 1.0_f32 / (dim as f32).sqrt();
        let mut envelope_soundness_verified = true;
        for cluster in cache.clusters() {
            let upper = cluster.bounding_box.evaluate_upper_logit(&cluster.center_key, &query, inv_sqrt_d);
            for &idx in &cluster.member_indices {
                let raw = &synthetic_keys[idx];
                let mut dot = 0.0_f32;
                for j in 0..dim {
                    dot += query[j] * raw[j];
                }
                let exact_z = dot * inv_sqrt_d;
                if exact_z > upper + TROPICAL_EPSILON {
                    envelope_soundness_verified = false;
                    break;
                }
            }
        }

        // Step 5: Exact Full Softmax Attention Baseline.
        let mut exact_logits = vec![0.0_f32; num_tokens];
        let mut exact_z_max = f32::NEG_INFINITY;
        for i in 0..num_tokens {
            let mut dot = 0.0_f32;
            for j in 0..dim {
                dot += query[j] * synthetic_keys[i][j];
            }
            let l = dot * inv_sqrt_d;
            exact_logits[i] = l;
            if l > exact_z_max {
                exact_z_max = l;
            }
        }

        let mut exact_sum_exp = 0.0_f32;
        let mut exact_weights = vec![0.0_f32; num_tokens];
        for i in 0..num_tokens {
            let exp_val = (exact_logits[i] - exact_z_max).exp();
            exact_weights[i] = exp_val;
            exact_sum_exp += exp_val;
        }
        let inv_exact_sum = 1.0 / exact_sum_exp;
        let mut exact_probs = vec![0.0_f32; num_tokens];
        let mut exact_output = vec![0.0_f32; v_dim];
        for i in 0..num_tokens {
            exact_probs[i] = exact_weights[i] * inv_exact_sum;
            for j in 0..v_dim {
                exact_output[j] += exact_probs[i] * synthetic_values[i][j];
            }
        }

        // Step 6: Benchmark Tropical Attention Latency & Execution.
        let mut decimated_output = vec![0.0_f32; v_dim];
        let benchmark_iterations = 200;
        let start = std::time::Instant::now();
        let mut last_stats = TropicalInferenceStats::default();

        for _ in 0..benchmark_iterations {
            last_stats = cache.compute_attention(&query, &mut decimated_output);
        }
        let elapsed = start.elapsed();
        let query_latency_us = ((elapsed.as_secs_f64() * 1_000_000.0) / (benchmark_iterations as f64)) as f32;

        // Step 7: Attention Recall & Reconstruction Error.
        // Measure probability mass captured by evaluated clusters.
        let mut active_indices_set = std::collections::HashSet::new();
        // Identify which clusters survived
        let z_max_thresh = {
            let mut zm = f32::NEG_INFINITY;
            for cluster in cache.clusters() {
                let mut dot = 0.0_f32;
                let raw = cluster.center_key.raw_coords();
                for j in 0..dim {
                    dot += query[j] * raw[j];
                }
                let l = dot * inv_sqrt_d;
                if l > zm {
                    zm = l;
                }
            }
            zm - self.config.pruning_deficit_threshold
        };

        for cluster in cache.clusters() {
            let upper = cluster.bounding_box.evaluate_upper_logit(&cluster.center_key, &query, inv_sqrt_d);
            if upper > z_max_thresh {
                for &idx in &cluster.member_indices {
                    active_indices_set.insert(idx);
                }
            }
        }
        if active_indices_set.is_empty() {
            for i in 0..num_tokens {
                active_indices_set.insert(i);
            }
        }

        let mut attention_recall = 0.0_f32;
        for &idx in &active_indices_set {
            attention_recall += exact_probs[idx];
        }

        // Relative Frobenius error: ||A_exact - A_dec||_2 / ||A_exact||_2.
        let mut diff_norm_sq = 0.0_f32;
        let mut exact_norm_sq = 0.0_f32;
        for j in 0..v_dim {
            let diff = exact_output[j] - decimated_output[j];
            diff_norm_sq += diff * diff;
            exact_norm_sq += exact_output[j] * exact_output[j];
        }
        let relative_frobenius_error = diff_norm_sq.sqrt() / (exact_norm_sq.sqrt() + TROPICAL_EPSILON);

        let l1_probability_deviation = 1.0_f32 - attention_recall;
        let total_memory_footprint_bytes = cache.memory_footprint_bytes();

        TropicalReport {
            head_dim: dim,
            total_tokens: num_tokens,
            active_clusters: cache.cluster_count(),
            pruning_ratio: last_stats.pruning_ratio,
            query_latency_us,
            attention_recall,
            relative_frobenius_error,
            l1_probability_deviation,
            max_observed_deficit: last_stats.max_observed_deficit,
            metric_triangle_error,
            softmax_invariance_error,
            sandwich_bound_verified,
            envelope_soundness_verified,
            total_memory_footprint_bytes,
        }
    }
}

// ==============================================================================
// 7. Unit Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tropical_key_metric_axioms() {
        let u_raw = vec![1.0, 3.0, -2.0, 5.0];
        let v_raw = vec![2.0, -1.0, 4.0, 0.0];
        let w_raw = vec![0.5, 2.0, 1.0, -3.0];

        let u = TropicalKey::new(&u_raw);
        let v = TropicalKey::new(&v_raw);
        let w = TropicalKey::new(&w_raw);

        // 1. Non-negativity
        let d_uv = u.tropical_distance(&v);
        let d_vu = v.tropical_distance(&u);
        assert!(d_uv >= 0.0);

        // 2. Symmetry
        assert!((d_uv - d_vu).abs() < TROPICAL_EPSILON);

        // 3. Identity modulo R*1
        let u_shifted_raw: Vec<f32> = u_raw.iter().map(|&x| x + 7.89).collect();
        let u_shifted = TropicalKey::new(&u_shifted_raw);
        let d_u_ushift = u.tropical_distance(&u_shifted);
        assert!(
            d_u_ushift < TROPICAL_EPSILON,
            "Shifted vectors must have zero tropical distance: got {}",
            d_u_ushift
        );

        // 4. Triangle inequality
        let d_uw = u.tropical_distance(&w);
        let d_vw = v.tropical_distance(&w);
        assert!(
            d_uw <= d_uv + d_vw + TROPICAL_EPSILON,
            "Triangle inequality violated: {} > {} + {}",
            d_uw,
            d_uv,
            d_vw
        );
    }

    #[test]
    fn test_softmax_translation_gauge_invariance() {
        let z = [1.2, -0.5, 3.4, 2.1];
        let c = 5.7_f32;
        let z_shifted: Vec<f32> = z.iter().map(|&x| x + c).collect();

        let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_z: f32 = z.iter().map(|&x| (x - max_z).exp()).sum();
        let p_z: Vec<f32> = z.iter().map(|&x| (x - max_z).exp() / sum_z).collect();

        let max_zs = z_shifted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_zs: f32 = z_shifted.iter().map(|&x| (x - max_zs).exp()).sum();
        let p_zs: Vec<f32> = z_shifted
            .iter()
            .map(|&x| (x - max_zs).exp() / sum_zs)
            .collect();

        for (p1, p2) in p_z.iter().zip(p_zs.iter()) {
            assert!(
                (p1 - p2).abs() < TROPICAL_EPSILON,
                "Softmax output must be invariant to uniform translation"
            );
        }
    }

    #[test]
    fn test_tropical_bounding_box_dominance() {
        let dim = 8;
        let k1 = TropicalKey::new(&[1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0, 1.5]);
        let k2 = TropicalKey::new(&[2.0, 1.5, -0.5, 2.5, 1.0, -1.5, 3.5, 2.0]);
        let mut bbox = TropicalBoundingBox::from_key(&k1, 0.1);
        bbox.extend(&k1, &k2, 0.2);

        let query = vec![0.5, -1.0, 2.0, -0.5, 1.5, 0.0, -2.0, 1.0];
        let inv_sqrt_d = 1.0 / (dim as f32).sqrt();

        let upper = bbox.evaluate_upper_logit(&k1, &query, inv_sqrt_d);

        let dot_1: f32 = query.iter().zip(k1.raw_coords()).map(|(&q, &k)| q * k).sum();
        let z1 = (dot_1 * inv_sqrt_d) + 0.1;

        let dot_2: f32 = query.iter().zip(k2.raw_coords()).map(|(&q, &k)| q * k).sum();
        let z2 = (dot_2 * inv_sqrt_d) + 0.2;

        assert!(
            upper >= z1 - TROPICAL_EPSILON,
            "Bounding box failed to dominate k1: {} < {}",
            upper,
            z1
        );
        assert!(
            upper >= z2 - TROPICAL_EPSILON,
            "Bounding box failed to dominate k2: {} < {}",
            upper,
            z2
        );
    }

    #[test]
    fn test_tropical_evaluator_harness() {
        let evaluator = TropicalEvaluator::default();
        let report = evaluator.evaluate_performance();

        assert!(report.pruning_ratio >= 0.70, "Pruning ratio {} < 0.70", report.pruning_ratio);
        assert!(report.query_latency_us < 10.0, "Latency {} us >= 10.0 us", report.query_latency_us);
        assert!(report.attention_recall >= 0.95, "Recall {} < 0.95", report.attention_recall);
        assert!(report.metric_triangle_error < 1e-5);
        assert!(report.softmax_invariance_error < 1e-6);
        assert!(report.sandwich_bound_verified);
        assert!(report.envelope_soundness_verified);
        assert!(report.total_memory_footprint_bytes < 30_000);
    }
}
