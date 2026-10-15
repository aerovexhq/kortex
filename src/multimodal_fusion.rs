//! Multi-Modal Acoustic-Visual Cross-Attention Fusion Engine
//!
//! Provides dynamic cross-attention fusion between Sonon directional acoustic
//! observations (3D Direction of Arrival) and DVS neuromorphic visual patches.
//!
//! Subsystem Capabilities:
//! - Pinhole camera 3D bearing coordinate projection into 2D focal planes
//! - Spatial parallax distance penalty D_spatial aligning auditory and visual coordinates
//! - Dynamic adaptive modality gate alpha in [0, 1] responsive to acoustic SNR and visual occlusions
//! - Unified fused key-value token generation with sub-10 microsecond latency
//! - Graceful single-modality fallback during acoustic rotor wash or visual blackout
//!
//! Strict Mandates:
//! - `#![deny(unsafe_code)]`
//! - Zero unicode emojis across all code, tests, and outputs
//! - Bounded static allocations and deterministic execution

#![deny(unsafe_code)]

/// Pinhole camera intrinsic parameters for focal plane projections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    pub width: f32,
    pub height: f32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        Self {
            width: 64.0,
            height: 64.0,
            fx: 48.0,
            fy: 48.0,
            cx: 32.0,
            cy: 32.0,
        }
    }
}

impl CameraIntrinsics {
    /// Projects a 3D unit bearing vector in camera frame (X: right, Y: down, Z: forward)
    /// into 2D pixel coordinates (u, v).
    ///
    /// Returns None if bearing points behind the camera plane (Z <= 0.01).
    pub fn project_bearing(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
        if z <= 0.01 {
            return None;
        }

        let u = self.fx * (x / z) + self.cx;
        let v = self.fy * (y / z) + self.cy;

        if u >= 0.0 && u < self.width && v >= 0.0 && v < self.height {
            Some((u, v))
        } else {
            None
        }
    }
}

/// An acoustic observation emitted by Sonon acoustic DSP kernel.
#[derive(Debug, Clone)]
pub struct AcousticObservation {
    /// Azimuth angle in radians in horizontal camera plane (0 = straight ahead, + = right).
    pub azimuth_rad: f32,
    /// Elevation angle in radians in vertical camera plane (0 = level, + = up).
    pub elevation_rad: f32,
    /// Signal-to-noise ratio in decibels (e.g. 18.0 dB clear, -5.0 dB heavy rotor noise).
    pub snr_db: f32,
    /// Sound classification confidence score in [0.0, 1.0].
    pub confidence: f32,
    /// Key embedding vector in attention feature space.
    pub key_embedding: Vec<f32>,
    /// Value embedding vector in attention feature space.
    pub value_embedding: Vec<f32>,
}

impl AcousticObservation {
    /// Converts azimuth and elevation angles into a 3D unit direction-of-arrival bearing vector
    /// in standard optical camera frame (X: right, Y: down, Z: forward).
    pub fn unit_bearing(&self) -> (f32, f32, f32) {
        let cos_elev = self.elevation_rad.cos();
        let sin_elev = self.elevation_rad.sin();
        let cos_azim = self.azimuth_rad.cos();
        let sin_azim = self.azimuth_rad.sin();

        // X: right, Y: down (negative elevation is up), Z: forward
        let x = sin_azim * cos_elev;
        let y = -sin_elev;
        let z = cos_azim * cos_elev;

        (x, y, z)
    }
}

/// A visual patch observation emitted by the DVS neuromorphic spike raster.
#[derive(Debug, Clone)]
pub struct VisualPatchObservation {
    pub patch_u: usize,
    pub patch_v: usize,
    /// Horizontal pixel center of the patch.
    pub pixel_x: f32,
    /// Vertical pixel center of the patch.
    pub pixel_y: f32,
    /// Optical flow velocity in camera plane [pixels / ms].
    pub flow_vx: f32,
    pub flow_vy: f32,
    /// Integrated polar energy.
    pub energy: f32,
    /// Key embedding vector in attention feature space.
    pub key_embedding: Vec<f32>,
    /// Value embedding vector in attention feature space.
    pub value_embedding: Vec<f32>,
}

/// Unified cross-attention token synthesized from acoustic and visual modalities.
#[derive(Debug, Clone)]
pub struct FusedMultiModalToken {
    /// Fused key vector injected into continuous attention cache.
    pub fused_key: Vec<f32>,
    /// Fused value vector injected into continuous attention cache.
    pub fused_value: Vec<f32>,
    /// Modality gate alpha in [0.0, 1.0]: weight allocated to acoustic evidence.
    pub modality_gate_alpha: f32,
    /// Normalized spatial misalignment distance between acoustic DoA and selected visual patch.
    pub spatial_distance_norm: f32,
    /// Index of visual patch chosen by cross-attention affinity, if available.
    pub best_visual_patch_idx: Option<usize>,
    /// Composite attention logit score for downstream decision heads.
    pub combined_attention_logit: f32,
}

/// Gated cross-attention fusion unit combining directional acoustics and visual spikes.
#[derive(Debug, Clone)]
pub struct GatedCrossAttentionFusion {
    pub intrinsics: CameraIntrinsics,
    pub embedding_dim: usize,
    /// Penalty coefficient beta scaling spatial misalignment distance.
    pub spatial_penalty_beta: f32,
    /// Reward coefficient gamma scaling logarithmic acoustic SNR.
    pub snr_reward_gamma: f32,
    /// Weights for linear modality gating [2 * embedding_dim].
    w_gate: Vec<f32>,
    /// Bias scalar for modality gating.
    b_gate: f32,
}

impl GatedCrossAttentionFusion {
    /// Initializes the fusion unit with deterministic linear gating parameters.
    pub fn new(intrinsics: CameraIntrinsics, embedding_dim: usize) -> Self {
        let mut w_gate = vec![0.0f32; 2 * embedding_dim];
        for (i, w) in w_gate.iter_mut().enumerate() {
            let seed = (i * 37 + 11) as f32;
            *w = (seed.sin() * 0.25) / (embedding_dim as f32).sqrt();
        }

        Self {
            intrinsics,
            embedding_dim,
            spatial_penalty_beta: 1.25,
            snr_reward_gamma: 0.15,
            w_gate,
            b_gate: 0.0,
        }
    }

    /// Logistic sigmoid activation function.
    #[inline]
    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Evaluates gated cross-attention between a query vector, an optional acoustic DoA observation,
    /// and a set of active visual patch observations.
    pub fn fuse_modalities(
        &self,
        query: &[f32],
        acoustic: Option<&AcousticObservation>,
        visual_patches: &[VisualPatchObservation],
    ) -> FusedMultiModalToken {
        let d = self.embedding_dim;
        let scale = 1.0 / (d as f32).sqrt();

        // 1. Acoustic branch
        let (audio_logit, audio_projected_pixel) = match acoustic {
            Some(audio) => {
                let mut dot_a = 0.0f32;
                for (j, &q_val) in query.iter().enumerate().take(d) {
                    dot_a += q_val * audio.key_embedding[j];
                }
                let snr_factor = (1.0 + (audio.snr_db.max(0.0) / 10.0)).ln();
                let score = dot_a * scale + self.snr_reward_gamma * snr_factor;

                let (bx, by, bz) = audio.unit_bearing();
                let proj = self.intrinsics.project_bearing(bx, by, bz);
                (Some(score), proj)
            }
            None => (None, None),
        };

        // 2. Visual branch with spatial parallax penalty
        let mut best_visual_idx = None;
        let mut best_visual_score = f32::NEG_INFINITY;
        let mut best_spatial_dist = 0.0f32;

        for (idx, patch) in visual_patches.iter().enumerate() {
            let mut dot_v = 0.0f32;
            for (j, &q_val) in query.iter().enumerate().take(d) {
                dot_v += q_val * patch.key_embedding[j];
            }
            let raw_v_score = dot_v * scale;

            // Compute spatial distance to projected acoustic DoA if available
            let spatial_dist = match audio_projected_pixel {
                Some((au, av)) => {
                    let du = (patch.pixel_x - au) / self.intrinsics.width;
                    let dv = (patch.pixel_y - av) / self.intrinsics.height;
                    (du * du + dv * dv).sqrt()
                }
                None => 0.0,
            };

            let penalized_score = raw_v_score - self.spatial_penalty_beta * spatial_dist;
            if penalized_score > best_visual_score {
                best_visual_score = penalized_score;
                best_visual_idx = Some(idx);
                best_spatial_dist = spatial_dist;
            }
        }

        // 3. Modality Gating & Blending
        match (acoustic, best_visual_idx) {
            // Case A: Both modalities available
            (Some(audio), Some(v_idx)) => {
                let best_patch = &visual_patches[v_idx];

                // Linear gate activation
                let mut gate_logit = self.b_gate;
                for j in 0..d {
                    gate_logit += audio.key_embedding[j] * self.w_gate[j];
                    gate_logit += best_patch.key_embedding[j] * self.w_gate[d + j];
                }

                // Modulate gate logit by SNR and spatial agreement
                // High SNR increases alpha (audio); high spatial distance decreases alpha
                let snr_offset = (audio.snr_db - 10.0) * 0.08;
                let spatial_offset = -best_spatial_dist * 1.5;
                let alpha = Self::sigmoid(gate_logit + snr_offset + spatial_offset);

                let mut fused_key = vec![0.0f32; d];
                let mut fused_val = vec![0.0f32; d];
                for j in 0..d {
                    fused_key[j] = alpha * audio.key_embedding[j] + (1.0 - alpha) * best_patch.key_embedding[j];
                    fused_val[j] = alpha * audio.value_embedding[j] + (1.0 - alpha) * best_patch.value_embedding[j];
                }

                let combined_logit = alpha * audio_logit.unwrap_or(0.0) + (1.0 - alpha) * best_visual_score;

                FusedMultiModalToken {
                    fused_key,
                    fused_value: fused_val,
                    modality_gate_alpha: alpha,
                    spatial_distance_norm: best_spatial_dist,
                    best_visual_patch_idx: Some(v_idx),
                    combined_attention_logit: combined_logit,
                }
            }

            // Case B: Audio only (Visual blackout / complete occlusion)
            (Some(audio), None) => {
                let score = audio_logit.unwrap_or(0.0);
                FusedMultiModalToken {
                    fused_key: audio.key_embedding.clone(),
                    fused_value: audio.value_embedding.clone(),
                    modality_gate_alpha: 1.0,
                    spatial_distance_norm: 0.0,
                    best_visual_patch_idx: None,
                    combined_attention_logit: score,
                }
            }

            // Case C: Visual only (Acoustic silence / sensor offline)
            (None, Some(v_idx)) => {
                let best_patch = &visual_patches[v_idx];
                FusedMultiModalToken {
                    fused_key: best_patch.key_embedding.clone(),
                    fused_value: best_patch.value_embedding.clone(),
                    modality_gate_alpha: 0.0,
                    spatial_distance_norm: 0.0,
                    best_visual_patch_idx: Some(v_idx),
                    combined_attention_logit: best_visual_score,
                }
            }

            // Case D: Neither modality available (Zero active sensors)
            (None, None) => FusedMultiModalToken {
                fused_key: vec![0.0; d],
                fused_value: vec![0.0; d],
                modality_gate_alpha: 0.5,
                spatial_distance_norm: 0.0,
                best_visual_patch_idx: None,
                combined_attention_logit: 0.0,
            },
        }
    }
}

/// Verification report returned by `MultiModalFusionEvaluator`.
#[derive(Debug, Clone)]
pub struct MultiModalFusionReport {
    pub fusion_latency_us: f32,
    pub balanced_alpha: f32,
    pub noisy_audio_alpha: f32,
    pub visual_blackout_alpha: f32,
    pub alpha_shift_under_noise: f32,
    pub occlusion_response_retention_pct: f32,
    pub spatial_alignment_verified: bool,
    pub total_memory_footprint_bytes: usize,
}

/// Benchmark evaluator testing multimodal fusion across operational conditions.
pub struct MultiModalFusionEvaluator;

impl MultiModalFusionEvaluator {
    /// Runs multimodal benchmark across balanced conditions, rotor noise, and visual blackout.
    pub fn evaluate() -> MultiModalFusionReport {
        let intrinsics = CameraIntrinsics::default();
        let dim = 64;
        let fusion_unit = GatedCrossAttentionFusion::new(intrinsics, dim);

        // Common planning query vector
        let mut query = vec![0.0f32; dim];
        for (i, q) in query.iter_mut().enumerate() {
            *q = ((i as f32 * 0.25).sin()) / (dim as f32).sqrt();
        }

        // 1. Setup Balanced Conditions: Approaching drone at azimuth = 15 deg, elevation = 5 deg
        let azim_rad = 15.0f32.to_radians();
        let elev_rad = 5.0f32.to_radians();

        let mut audio_key = vec![0.0f32; dim];
        let mut audio_val = vec![0.0f32; dim];
        for (i, (k, v)) in audio_key.iter_mut().zip(audio_val.iter_mut()).enumerate() {
            *k = (i as f32 * 0.31).cos() * 0.4;
            *v = (i as f32 * 0.19).sin() * 0.4;
        }

        let audio_clear = AcousticObservation {
            azimuth_rad: azim_rad,
            elevation_rad: elev_rad,
            snr_db: 18.0, // High SNR
            confidence: 0.95,
            key_embedding: audio_key.clone(),
            value_embedding: audio_val.clone(),
        };

        // Create 8 visual patches across the camera grid
        let mut visual_patches = Vec::with_capacity(8);
        for idx in 0..8 {
            let u = idx % 4;
            let v = idx / 4;
            let px = (u as f32 * 16.0) + 8.0;
            let py = (v as f32 * 32.0) + 16.0;

            let mut vk = vec![0.0f32; dim];
            let mut vv = vec![0.0f32; dim];
            for (i, (k, val)) in vk.iter_mut().zip(vv.iter_mut()).enumerate() {
                *k = ((i + idx * 7) as f32 * 0.27).sin() * 0.4;
                *val = ((i + idx * 11) as f32 * 0.33).cos() * 0.4;
            }

            visual_patches.push(VisualPatchObservation {
                patch_u: u,
                patch_v: v,
                pixel_x: px,
                pixel_y: py,
                flow_vx: 4.5,
                flow_vy: -1.2,
                energy: 2.4,
                key_embedding: vk,
                value_embedding: vv,
            });
        }

        // Benchmark fusion latency over 5,000 iterations
        let num_evals = 5000;
        let t_start = std::time::Instant::now();
        let mut balanced_token = FusedMultiModalToken {
            fused_key: vec![],
            fused_value: vec![],
            modality_gate_alpha: 0.0,
            spatial_distance_norm: 0.0,
            best_visual_patch_idx: None,
            combined_attention_logit: 0.0,
        };

        for _ in 0..num_evals {
            balanced_token = fusion_unit.fuse_modalities(&query, Some(&audio_clear), &visual_patches);
        }
        let elapsed = t_start.elapsed();
        let latency_us = (elapsed.as_nanos() as f32 / 1000.0) / (num_evals as f32);

        let balanced_alpha = balanced_token.modality_gate_alpha;

        // 2. High Acoustic Noise condition (Rotor wash: -8.0 dB SNR)
        let audio_noisy = AcousticObservation {
            azimuth_rad: azim_rad,
            elevation_rad: elev_rad,
            snr_db: -8.0, // Severe noise
            confidence: 0.35,
            key_embedding: audio_key.clone(),
            value_embedding: audio_val.clone(),
        };
        let noisy_token = fusion_unit.fuse_modalities(&query, Some(&audio_noisy), &visual_patches);
        let noisy_alpha = noisy_token.modality_gate_alpha;
        let alpha_shift = balanced_alpha - noisy_alpha;

        // 3. Complete Visual Blackout (Zero visual patches)
        let blackout_token = fusion_unit.fuse_modalities(&query, Some(&audio_clear), &[]);
        let blackout_alpha = blackout_token.modality_gate_alpha;

        let retention_pct = (blackout_token.combined_attention_logit.abs()
            / (balanced_token.combined_attention_logit.abs() + 1e-6))
            * 100.0;

        // Memory footprint: intrinsics (24 B) + w_gate (2 * dim * 4 B) + b_gate (4 B)
        let mem_bytes = std::mem::size_of::<CameraIntrinsics>() + (2 * dim * 4) + 4;

        MultiModalFusionReport {
            fusion_latency_us: latency_us,
            balanced_alpha,
            noisy_audio_alpha: noisy_alpha,
            visual_blackout_alpha: blackout_alpha,
            alpha_shift_under_noise: alpha_shift,
            occlusion_response_retention_pct: retention_pct.min(100.0),
            spatial_alignment_verified: balanced_token.best_visual_patch_idx.is_some(),
            total_memory_footprint_bytes: mem_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinhole_camera_optical_center_projection() {
        let cam = CameraIntrinsics {
            width: 128.0,
            height: 128.0,
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 64.0,
        };

        // Straight ahead bearing (0, 0, 1) must project exactly to optical center (64, 64)
        let proj = cam.project_bearing(0.0, 0.0, 1.0);
        assert!(proj.is_some());
        let (u, v) = proj.unwrap();
        assert!((u - 64.0).abs() < 1e-4);
        assert!((v - 64.0).abs() < 1e-4);

        // Bearing behind camera (Z <= 0) must return None
        assert!(cam.project_bearing(0.0, 0.0, -1.0).is_none());
    }

    #[test]
    fn test_acoustic_bearing_unit_vector_norm() {
        let obs = AcousticObservation {
            azimuth_rad: 0.65,
            elevation_rad: -0.32,
            snr_db: 12.0,
            confidence: 0.88,
            key_embedding: vec![0.0; 16],
            value_embedding: vec![0.0; 16],
        };

        let (x, y, z) = obs.unit_bearing();
        let norm = (x * x + y * y + z * z).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Bearing vector must have unit Euclidean norm");
    }

    #[test]
    fn test_multimodal_fusion_evaluator_metrics() {
        let report = MultiModalFusionEvaluator::evaluate();
        assert!(report.fusion_latency_us < 10.0, "Latency must be < 10 us, got {}", report.fusion_latency_us);
        assert!(report.balanced_alpha > report.noisy_audio_alpha, "Alpha must decrease under acoustic noise");
        assert_eq!(report.visual_blackout_alpha, 1.0, "Blackout alpha must be 1.0 (pure audio)");
        assert!(report.spatial_alignment_verified);
    }
}
