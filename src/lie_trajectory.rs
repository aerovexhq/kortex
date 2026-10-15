//! Online Lie-Group Geometric Trajectory Predictor Engine
//!
//! Implements closed-form Lie algebra se(3) matrix exponential and logarithm maps
//! for continuous geodesic trajectory continuation on the Special Euclidean Lie group SE(3),
//! predicting future obstacle poses and injecting trajectory envelopes into transformer KV-space.
//!
//! Subsystem Capabilities:
//! - Closed-form Rodrigues SO(3) matrix exponential and left Jacobian V of SO(3)
//! - Closed-form matrix logarithm mapping SE(3) poses to spatial twist coordinates (v, omega)
//! - Continuous geodesic trajectory rollout T(t + k*dt) = T(t) * exp(k*dt * xi^)
//! - Exact helical screw motion invariance (Chasles' theorem) with zero drift
//! - High-rate linear projection of 16D geometric descriptors into transformer key-value tokens
//!
//! Strict Mandates:
//! - `#![deny(unsafe_code)]`
//! - Zero unicode emojis across all code, tests, and outputs
//! - Bounded static allocations and deterministic execution

#![deny(unsafe_code)]

/// Rigid-body pose in the Special Euclidean Lie group SE(3) represented by a
/// 3x3 rotation matrix R in SO(3) and a 3D translation vector p in R^3.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3Pose {
    /// 3x3 row-major rotation matrix belonging to SO(3).
    pub rotation: [f32; 9],
    /// 3D translation vector (x, y, z) in meters.
    pub translation: [f32; 3],
}

impl Default for Se3Pose {
    fn default() -> Self {
        Self::identity()
    }
}

impl Se3Pose {
    /// Constructs the identity transformation (zero translation, identity rotation).
    pub fn identity() -> Self {
        Self {
            rotation: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Constructs an SE(3) pose from explicit rotation matrix and translation vector.
    pub fn new(rotation: [f32; 9], translation: [f32; 3]) -> Self {
        Self { rotation, translation }
    }

    /// Composes this pose T1 with another pose T2: T_res = T1 * T2.
    ///
    /// Mathematical definition:
    /// R_res = R1 * R2
    /// p_res = R1 * p2 + p1
    pub fn compose(&self, other: &Self) -> Self {
        let r1 = &self.rotation;
        let r2 = &other.rotation;

        let mut r_res = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                r_res[i * 3 + j] = r1[i * 3] * r2[j]
                    + r1[i * 3 + 1] * r2[3 + j]
                    + r1[i * 3 + 2] * r2[6 + j];
            }
        }

        let p1 = &self.translation;
        let p2 = &other.translation;

        let p_res = [
            r1[0] * p2[0] + r1[1] * p2[1] + r1[2] * p2[2] + p1[0],
            r1[3] * p2[0] + r1[4] * p2[1] + r1[5] * p2[2] + p1[1],
            r1[6] * p2[0] + r1[7] * p2[1] + r1[8] * p2[2] + p1[2],
        ];

        Self {
            rotation: r_res,
            translation: p_res,
        }
    }

    /// Computes the exact group inverse: T^{-1} = [R^T, -R^T * p].
    pub fn inverse(&self) -> Self {
        let r = &self.rotation;
        let p = &self.translation;

        // Transpose of 3x3 rotation matrix
        let r_inv = [
            r[0], r[3], r[6],
            r[1], r[4], r[7],
            r[2], r[5], r[8],
        ];

        let p_inv = [
            -(r_inv[0] * p[0] + r_inv[1] * p[1] + r_inv[2] * p[2]),
            -(r_inv[3] * p[0] + r_inv[4] * p[1] + r_inv[5] * p[2]),
            -(r_inv[6] * p[0] + r_inv[7] * p[1] + r_inv[8] * p[2]),
        ];

        Self {
            rotation: r_inv,
            translation: p_inv,
        }
    }

    /// Transforms a 3D coordinate point: p' = R * pt + p.
    pub fn transform_point(&self, pt: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let p = &self.translation;

        [
            r[0] * pt[0] + r[1] * pt[1] + r[2] * pt[2] + p[0],
            r[3] * pt[0] + r[4] * pt[1] + r[5] * pt[2] + p[1],
            r[6] * pt[0] + r[7] * pt[1] + r[8] * pt[2] + p[2],
        ]
    }

    /// Computes the determinant of the 3x3 rotation component.
    pub fn det_rotation(&self) -> f32 {
        let r = &self.rotation;
        r[0] * (r[4] * r[8] - r[5] * r[7])
            - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6])
    }

    /// Computes the Frobenius norm orthogonality error: ||R^T * R - I||_F.
    pub fn orthogonality_error(&self) -> f32 {
        let r = &self.rotation;
        let mut err_sq = 0.0f32;

        for i in 0..3 {
            for j in 0..3 {
                // Dot product of column i and column j of R
                let dot = r[i] * r[j] + r[3 + i] * r[3 + j] + r[6 + i] * r[6 + j];
                let target = if i == j { 1.0f32 } else { 0.0f32 };
                let diff = dot - target;
                err_sq += diff * diff;
            }
        }

        err_sq.sqrt()
    }
}

/// Rigid-body spatial twist vector in the Lie algebra se(3), parameterized as
/// xi = (linear velocity v, angular velocity omega).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3Twist {
    /// Linear velocity vector v = (vx, vy, vz) in m/s.
    pub linear: [f32; 3],
    /// Angular velocity vector omega = (wx, wy, wz) in rad/s.
    pub angular: [f32; 3],
}

impl Default for Se3Twist {
    fn default() -> Self {
        Self::zero()
    }
}

impl Se3Twist {
    /// Zero twist.
    pub fn zero() -> Self {
        Self {
            linear: [0.0, 0.0, 0.0],
            angular: [0.0, 0.0, 0.0],
        }
    }

    /// Constructs a twist from explicit linear and angular vectors.
    pub fn new(linear: [f32; 3], angular: [f32; 3]) -> Self {
        Self { linear, angular }
    }

    /// Multiplies the twist by a scalar (e.g. time step dt).
    pub fn scale(&self, s: f32) -> Self {
        Self {
            linear: [self.linear[0] * s, self.linear[1] * s, self.linear[2] * s],
            angular: [self.angular[0] * s, self.angular[1] * s, self.angular[2] * s],
        }
    }

    /// Magnitude of the angular velocity vector ||omega||_2.
    pub fn norm_angular(&self) -> f32 {
        let w = &self.angular;
        (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt()
    }

    /// Magnitude of the linear velocity vector ||v||_2.
    pub fn norm_linear(&self) -> f32 {
        let v = &self.linear;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Computes the screw pitch: h = (v . omega) / (||omega||^2 + epsilon).
    pub fn helical_pitch(&self) -> f32 {
        let w = &self.angular;
        let v = &self.linear;
        let w_sq = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        let v_dot_w = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
        v_dot_w / (w_sq + 1e-7)
    }
}

/// Constructs the 3x3 skew-symmetric cross-product matrix [omega]_x in so(3).
pub fn skew_symmetric(w: [f32; 3]) -> [f32; 9] {
    [
        0.0, -w[2], w[1],
        w[2], 0.0, -w[0],
        -w[1], w[0], 0.0,
    ]
}

/// Computes the matrix exponential map exp: se(3) -> SE(3) using closed-form
/// Rodrigues' formula for SO(3) and the left Jacobian V of SO(3).
pub fn exp_se3(twist: &Se3Twist) -> Se3Pose {
    let w = &twist.angular;
    let v = &twist.linear;
    let theta_sq = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let theta = theta_sq.sqrt();

    let k = skew_symmetric(*w);

    // K^2 = K * K
    let mut k_sq = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            k_sq[i * 3 + j] = k[i * 3] * k[j] + k[i * 3 + 1] * k[3 + j] + k[i * 3 + 2] * k[6 + j];
        }
    }

    let (a, b, c) = if theta < 1e-5 {
        // Taylor expansions around theta = 0 to avoid division by zero
        let a = 1.0 - (theta_sq / 6.0);
        let b = 0.5 - (theta_sq / 24.0);
        let c = (1.0 / 6.0) - (theta_sq / 120.0);
        (a, b, c)
    } else {
        let a = theta.sin() / theta;
        let b = (1.0 - theta.cos()) / theta_sq;
        let c = (theta - theta.sin()) / (theta_sq * theta);
        (a, b, c)
    };

    // R = I + A * K + B * K^2
    let mut r = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let eye = if i == j { 1.0 } else { 0.0 };
            r[idx] = eye + a * k[idx] + b * k_sq[idx];
        }
    }

    // Left Jacobian V = I + B * K + C * K^2
    let mut v_mat = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let eye = if i == j { 1.0 } else { 0.0 };
            v_mat[idx] = eye + b * k[idx] + c * k_sq[idx];
        }
    }

    // Translation p = V * v
    let p = [
        v_mat[0] * v[0] + v_mat[1] * v[1] + v_mat[2] * v[2],
        v_mat[3] * v[0] + v_mat[4] * v[1] + v_mat[5] * v[2],
        v_mat[6] * v[0] + v_mat[7] * v[1] + v_mat[8] * v[2],
    ];

    Se3Pose {
        rotation: r,
        translation: p,
    }
}

/// Computes the closed-form matrix logarithm map log: SE(3) -> se(3).
pub fn log_se3(pose: &Se3Pose) -> Se3Twist {
    let r = &pose.rotation;
    let p = &pose.translation;

    let tr = r[0] + r[4] + r[8];
    let cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta < 1e-5 {
        // Pure translation / infinitesimal rotation
        let w = [
            0.5 * (r[7] - r[5]),
            0.5 * (r[2] - r[6]),
            0.5 * (r[3] - r[1]),
        ];
        return Se3Twist {
            linear: *p,
            angular: w,
        };
    }

    let alpha = theta / (2.0 * theta.sin());
    let w = [
        alpha * (r[7] - r[5]),
        alpha * (r[2] - r[6]),
        alpha * (r[3] - r[1]),
    ];

    let k = skew_symmetric(w);
    let mut k_sq = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            k_sq[i * 3 + j] = k[i * 3] * k[j] + k[i * 3 + 1] * k[3 + j] + k[i * 3 + 2] * k[6 + j];
        }
    }

    // V^{-1} = I - 0.5 * K + (1 / theta^2) * (1 - (theta * sin(theta)) / (2 * (1 - cos(theta)))) * K^2
    let theta_sq = theta * theta;
    let coeff = (1.0 / theta_sq) * (1.0 - (theta * theta.sin()) / (2.0 * (1.0 - theta.cos())));

    let mut v_inv = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            let eye = if i == j { 1.0 } else { 0.0 };
            v_inv[idx] = eye - 0.5 * k[idx] + coeff * k_sq[idx];
        }
    }

    let v = [
        v_inv[0] * p[0] + v_inv[1] * p[1] + v_inv[2] * p[2],
        v_inv[3] * p[0] + v_inv[4] * p[1] + v_inv[5] * p[2],
        v_inv[6] * p[0] + v_inv[7] * p[1] + v_inv[8] * p[2],
    ];

    Se3Twist {
        linear: v,
        angular: w,
    }
}

/// Token payload emitted for a future lookahead step along the continuous geodesic trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryLookaheadToken {
    /// Lookahead step index (1 ..= lookahead_steps).
    pub step_index: usize,
    /// Future lookahead time offset in seconds: k * dt.
    pub time_offset_seconds: f32,
    /// Exact predicted pose in SE(3).
    pub predicted_pose: Se3Pose,
    /// Projected key embedding vector.
    pub key_embedding: Vec<f32>,
    /// Projected value embedding vector.
    pub value_embedding: Vec<f32>,
}

/// Online Lie-group geometric trajectory predictor.
#[derive(Debug, Clone)]
pub struct LieGroupPredictor {
    /// Number of lookahead steps (e.g. 8).
    pub lookahead_steps: usize,
    /// Lookahead step duration in seconds (e.g. 0.05s = 50ms).
    pub dt_step_seconds: f32,
    /// Embedding dimension for key and value vectors.
    pub embedding_dim: usize,
    /// Linear weights for Key projection [16 * embedding_dim].
    w_key: Vec<f32>,
    /// Linear weights for Value projection [16 * embedding_dim].
    w_val: Vec<f32>,
}

impl LieGroupPredictor {
    /// Allocates a new Lie group trajectory predictor with deterministic linear projection weights.
    pub fn new(lookahead_steps: usize, dt_step_seconds: f32, embedding_dim: usize) -> Self {
        let feat_dim = 16;
        let mut w_key = vec![0.0f32; feat_dim * embedding_dim];
        let mut w_val = vec![0.0f32; feat_dim * embedding_dim];

        for i in 0..feat_dim {
            for j in 0..embedding_dim {
                let idx = i * embedding_dim + j;
                let seed_k = (i * 29 + j * 13 + 5) as f32;
                let seed_v = (i * 19 + j * 17 + 11) as f32;
                w_key[idx] = (seed_k.sin() * 0.35) / (feat_dim as f32).sqrt();
                w_val[idx] = (seed_v.cos() * 0.35) / (feat_dim as f32).sqrt();
            }
        }

        Self {
            lookahead_steps,
            dt_step_seconds,
            embedding_dim,
            w_key,
            w_val,
        }
    }

    /// Propagates continuous geodesic trajectory for `lookahead_steps` into the future
    /// using closed-form Lie algebra integration:
    /// T_k = T_0 * exp(k * dt * xi^)
    pub fn propagate_trajectory(
        &self,
        initial_pose: &Se3Pose,
        twist: &Se3Twist,
    ) -> Vec<TrajectoryLookaheadToken> {
        let mut tokens = Vec::with_capacity(self.lookahead_steps);
        let feat_dim = 16;

        let v_norm = twist.norm_linear();
        let w_norm = twist.norm_angular();
        let pitch = twist.helical_pitch();

        for k in 1..=self.lookahead_steps {
            let t_offset = (k as f32) * self.dt_step_seconds;

            // Scaled twist for lookahead interval
            let step_twist = twist.scale(t_offset);

            // Compute SE(3) transformation along geodesic
            let delta_pose = exp_se3(&step_twist);
            let pred_pose = initial_pose.compose(&delta_pose);

            // Assemble 16-dimensional geometric descriptor
            let mut feat = [0.0f32; 16];
            feat[0] = pred_pose.translation[0];
            feat[1] = pred_pose.translation[1];
            feat[2] = pred_pose.translation[2];
            feat[3] = pred_pose.rotation[0];
            feat[4] = pred_pose.rotation[1];
            feat[5] = pred_pose.rotation[2];
            feat[6] = pred_pose.rotation[3];
            feat[7] = pred_pose.rotation[4];
            feat[8] = pred_pose.rotation[5];
            feat[9] = pred_pose.rotation[6];
            feat[10] = pred_pose.rotation[7];
            feat[11] = pred_pose.rotation[8];
            feat[12] = t_offset;
            feat[13] = v_norm;
            feat[14] = w_norm;
            feat[15] = pitch;

            // Linear projection to Key and Value vectors
            let mut key = vec![0.0f32; self.embedding_dim];
            let mut val = vec![0.0f32; self.embedding_dim];

            for (i, &f_val) in feat.iter().enumerate().take(feat_dim) {
                let row = i * self.embedding_dim;
                for j in 0..self.embedding_dim {
                    key[j] += f_val * self.w_key[row + j];
                    val[j] += f_val * self.w_val[row + j];
                }
            }

            tokens.push(TrajectoryLookaheadToken {
                step_index: k,
                time_offset_seconds: t_offset,
                predicted_pose: pred_pose,
                key_embedding: key,
                value_embedding: val,
            });
        }

        tokens
    }
}

/// Verification report returned by `LieGroupPredictorEvaluator`.
#[derive(Debug, Clone)]
pub struct LieGroupPredictorReport {
    pub propagation_latency_us: f32,
    pub max_orthogonality_error: f32,
    pub log_exp_roundtrip_error: f32,
    pub helical_pitch_drift: f32,
    pub total_lookahead_steps: usize,
    pub total_memory_footprint_bytes: usize,
    pub collision_attention_logit: f32,
}

/// Benchmark evaluator for online Lie group trajectory prediction.
pub struct LieGroupPredictorEvaluator;

impl LieGroupPredictorEvaluator {
    /// Executes end-to-end Lie algebra exponential, logarithm, helical screw motion,
    /// and lookahead token propagation benchmarks.
    pub fn evaluate() -> LieGroupPredictorReport {
        let predictor = LieGroupPredictor::new(8, 0.05, 64);

        // 1. Initial pose: obstacle at (10, 5, -2) heading forward
        let initial_pose = Se3Pose::new(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 5.0, -2.0],
        );

        // Helical screw twist: linear velocity 12 m/s along X, banking turn 0.8 rad/s around Z
        let twist = Se3Twist::new([12.0, 2.0, 1.5], [0.1, 0.2, 0.8]);
        let initial_pitch = twist.helical_pitch();

        // 2. Benchmark propagation latency over 2,000 iterations
        let num_evals = 2000;
        let t_start = std::time::Instant::now();
        let mut tokens = Vec::new();
        for _ in 0..num_evals {
            tokens = predictor.propagate_trajectory(&initial_pose, &twist);
        }
        let elapsed = t_start.elapsed();
        let latency_us = (elapsed.as_nanos() as f32 / 1000.0) / (num_evals as f32);

        // 3. Verify SO(3) orthogonality across all lookaheads
        let mut max_ortho_err = 0.0f32;
        for tok in &tokens {
            let err = tok.predicted_pose.orthogonality_error();
            if err > max_ortho_err {
                max_ortho_err = err;
            }
        }

        // 4. Verify log(exp(twist)) roundtrip error
        let exp_p = exp_se3(&twist);
        let recovered_twist = log_se3(&exp_p);
        let mut roundtrip_err_sq = 0.0f32;
        for i in 0..3 {
            let dv = twist.linear[i] - recovered_twist.linear[i];
            let dw = twist.angular[i] - recovered_twist.angular[i];
            roundtrip_err_sq += dv * dv + dw * dw;
        }
        let roundtrip_err = roundtrip_err_sq.sqrt();

        // 5. Helical pitch drift
        let last_step_twist = log_se3(&initial_pose.inverse().compose(&tokens.last().unwrap().predicted_pose));
        let last_pitch = last_step_twist.scale(1.0 / (8.0 * 0.05)).helical_pitch();
        let pitch_drift = (last_pitch - initial_pitch).abs();

        // 6. Test attention dot-product response with a collision probe query
        let query = vec![0.1f32; predictor.embedding_dim];
        let mut max_logit = f32::NEG_INFINITY;
        let scale = 1.0 / (predictor.embedding_dim as f32).sqrt();

        for tok in &tokens {
            let mut dot = 0.0f32;
            for (j, &q) in query.iter().enumerate().take(predictor.embedding_dim) {
                dot += q * tok.key_embedding[j];
            }
            let logit = dot * scale;
            if logit > max_logit {
                max_logit = logit;
            }
        }

        let mem_bytes = 2 * (16 * predictor.embedding_dim * 4) + std::mem::size_of::<LieGroupPredictor>();

        LieGroupPredictorReport {
            propagation_latency_us: latency_us,
            max_orthogonality_error: max_ortho_err,
            log_exp_roundtrip_error: roundtrip_err,
            helical_pitch_drift: pitch_drift,
            total_lookahead_steps: predictor.lookahead_steps,
            total_memory_footprint_bytes: mem_bytes,
            collision_attention_logit: max_logit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_se3_identity_and_composition() {
        let p_id = Se3Pose::identity();
        assert!((p_id.det_rotation() - 1.0).abs() < 1e-6);
        assert!(p_id.orthogonality_error() < 1e-6);

        let p1 = Se3Pose::new(
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], // 90 deg yaw
            [5.0, 2.0, -1.0],
        );
        let p1_inv = p1.inverse();
        let p_comp = p1.compose(&p1_inv);

        assert!(p_comp.orthogonality_error() < 1e-5);
        for i in 0..3 {
            assert!(p_comp.translation[i].abs() < 1e-4);
        }
    }

    #[test]
    fn test_pure_translation_reduction() {
        let twist = Se3Twist::new([10.0, -5.0, 3.0], [0.0, 0.0, 0.0]);
        let pose = exp_se3(&twist);

        assert!(pose.orthogonality_error() < 1e-6);
        assert_eq!(pose.translation, [10.0, -5.0, 3.0]);

        let recovered = log_se3(&pose);
        assert_eq!(recovered.linear, [10.0, -5.0, 3.0]);
        assert_eq!(recovered.angular, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_lie_predictor_evaluator_metrics() {
        let report = LieGroupPredictorEvaluator::evaluate();
        assert!(report.propagation_latency_us < 5.0, "Latency must be < 5 us, got {}", report.propagation_latency_us);
        assert!(report.max_orthogonality_error < 1e-4, "Orthogonality error must be < 1e-4");
        assert!(report.log_exp_roundtrip_error < 1e-3, "Roundtrip error must be < 1e-3");
        assert!(report.helical_pitch_drift < 1e-3, "Helical pitch drift must be < 1e-3");
        assert_eq!(report.total_lookahead_steps, 8);
    }
}
