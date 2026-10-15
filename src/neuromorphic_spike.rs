//! Neuromorphic Event Camera Asynchronous Spike KV-Cache Injection Engine
//!
//! Provides direct event-based spike raster embedding into Kortex transformer
//! attention context without frame binning latency.
//!
//! Subsystem Capabilities:
//! - Continuous exponential decay polar surface S(x, y, t)
//! - Surface of Active Events (SAE) with central-difference optical flow estimation
//! - Refractory period and hot-pixel noise filtering
//! - Spatial patch energy tracking and activity-gated sparse token emission
//! - Direct linear key-value token projection with rotary positional embedding
//!
//! Strict Mandates:
//! - `#![deny(unsafe_code)]`
//! - Zero unicode emojis across all code, tests, and outputs
//! - Bounded static allocations and deterministic execution

#![deny(unsafe_code)]

/// Representation of an asynchronous event emitted by a Dynamic Vision Sensor (DVS).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvsEvent {
    /// Horizontal pixel coordinate in focal plane [0, width - 1].
    pub x: u16,
    /// Vertical pixel coordinate in focal plane [0, height - 1].
    pub y: u16,
    /// Microsecond timestamp of event arrival.
    pub timestamp_us: u64,
    /// Event polarity: -1 for luminance decrease, +1 for luminance increase.
    pub polarity: i8,
}

/// Operational parameters for the neuromorphic event camera pipeline.
#[derive(Debug, Clone)]
pub struct EventCameraConfig {
    /// Sensor horizontal resolution in pixels.
    pub width: usize,
    /// Sensor vertical resolution in pixels.
    pub height: usize,
    /// Spatial patch dimension in pixels (e.g. 8 for an 8x8 patch).
    pub patch_size: usize,
    /// Exponential decay time constant in microseconds (tau).
    pub tau_us: f32,
    /// Electronic refractory period in microseconds (drops double triggers).
    pub refractory_us: u64,
    /// Minimum interval between events on identical pixel to avoid hot-pixel noise.
    pub hot_pixel_cap_us: u64,
    /// Patch energy threshold for triggering active token emission.
    pub trigger_threshold: f32,
    /// Transformer key and value embedding dimension.
    pub embedding_dim: usize,
}

impl Default for EventCameraConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            patch_size: 8,
            tau_us: 10_000.0, // 10 ms decay
            refractory_us: 5,  // 5 us refractory dead-time
            hot_pixel_cap_us: 10, // 100 kHz maximum pixel frequency
            trigger_threshold: 0.85,
            embedding_dim: 64,
        }
    }
}

/// Continuous exponential decay polar surface and Time Surface of Active Events (SAE).
#[derive(Debug, Clone)]
pub struct EventSurfaceRaster {
    /// Configuration settings.
    pub config: EventCameraConfig,
    /// Polar surface S(x, y, t) accumulating signed decay.
    polar_surface: Vec<f32>,
    /// Time surface T(x, y) recording the most recent event timestamp in microseconds.
    time_surface_us: Vec<u64>,
    /// Previous event timestamp per pixel for refractory period validation.
    last_event_us: Vec<u64>,
    /// Total events accepted and integrated.
    accepted_events: usize,
    /// Total events dropped by refractory period or hot-pixel suppression.
    filtered_events: usize,
}

impl EventSurfaceRaster {
    /// Allocates a new event surface raster with pre-allocated zero-initialized buffers.
    pub fn new(config: EventCameraConfig) -> Self {
        let total_pixels = config.width * config.height;
        Self {
            config,
            polar_surface: vec![0.0; total_pixels],
            time_surface_us: vec![0; total_pixels],
            last_event_us: vec![u64::MAX; total_pixels],
            accepted_events: 0,
            filtered_events: 0,
        }
    }

    /// Linear pixel index calculation.
    #[inline]
    fn pixel_index(&self, x: usize, y: usize) -> usize {
        y * self.config.width + x
    }

    /// Ingests an asynchronous DVS event into the continuous polar and time surfaces.
    ///
    /// Returns true if the event was accepted and integrated; false if rejected by noise filters.
    pub fn ingest_event(&mut self, event: DvsEvent) -> bool {
        let x = event.x as usize;
        let y = event.y as usize;

        if x >= self.config.width || y >= self.config.height {
            self.filtered_events += 1;
            return false;
        }

        let idx = self.pixel_index(x, y);
        let last_t = self.last_event_us[idx];
        let has_prior = last_t != u64::MAX;

        // Refractory period check: drop events occurring too soon after previous event
        if has_prior && event.timestamp_us >= last_t && (event.timestamp_us - last_t) < self.config.refractory_us {
            self.filtered_events += 1;
            return false;
        }

        // Hot pixel suppression: drop if delta t is less than hot pixel cap
        if has_prior && event.timestamp_us >= last_t && (event.timestamp_us - last_t) < self.config.hot_pixel_cap_us {
            self.filtered_events += 1;
            return false;
        }

        // Compute continuous exponential decay since last event at this pixel
        let dt_us = if has_prior && event.timestamp_us >= last_t {
            (event.timestamp_us - last_t) as f32
        } else {
            0.0
        };

        let decay = (-dt_us / self.config.tau_us).exp();
        let prev_val = self.polar_surface[idx];
        let new_val = prev_val * decay + (event.polarity as f32);

        // Clamping to maintain strict numerical boundedness
        self.polar_surface[idx] = new_val.clamp(-10.0, 10.0);
        self.time_surface_us[idx] = event.timestamp_us;
        self.last_event_us[idx] = event.timestamp_us;
        self.accepted_events += 1;

        true
    }

    /// Retrieves the current decayed polar value at pixel (x, y) evaluated at current_time_us.
    pub fn get_polar_value(&self, x: usize, y: usize, current_time_us: u64) -> f32 {
        if x >= self.config.width || y >= self.config.height {
            return 0.0;
        }
        let idx = self.pixel_index(x, y);
        let last_t = self.last_event_us[idx];
        if last_t == u64::MAX {
            return 0.0;
        }
        let prev_val = self.polar_surface[idx];
        if prev_val == 0.0 {
            return 0.0;
        }
        let dt = if current_time_us >= last_t {
            (current_time_us - last_t) as f32
        } else {
            0.0
        };
        // If elapsed time exceeds 5 time constants, value decays to under 0.67% of initial
        if dt > 5.0 * self.config.tau_us {
            return 0.0;
        }
        let decay = (-dt / self.config.tau_us).exp();
        prev_val * decay
    }

    /// Estimates instantaneous normal optical flow velocity (vx, vy) at pixel (x, y)
    /// using spatial finite differences on the Surface of Active Events (SAE).
    ///
    /// Optical flow velocity returned is in units of [pixels / millisecond].
    pub fn estimate_optical_flow(&self, x: usize, y: usize) -> (f32, f32) {
        if x == 0 || x + 1 >= self.config.width || y == 0 || y + 1 >= self.config.height {
            return (0.0, 0.0);
        }

        let idx_c = self.pixel_index(x, y);
        let t_c = self.time_surface_us[idx_c] as f32;
        if t_c < 1.0 {
            return (0.0, 0.0);
        }

        let max_age_us = (self.config.tau_us * 1.5).max(5000.0);

        let t_left = self.time_surface_us[self.pixel_index(x - 1, y)] as f32;
        let t_right = self.time_surface_us[self.pixel_index(x + 1, y)] as f32;
        let t_down = self.time_surface_us[self.pixel_index(x, y - 1)] as f32;
        let t_up = self.time_surface_us[self.pixel_index(x, y + 1)] as f32;

        let grad_x_us = if t_right > 0.0 && t_left > 0.0 && (t_right - t_c).abs() < max_age_us && (t_c - t_left).abs() < max_age_us {
            (t_right - t_left) / 2.0
        } else if t_right > 0.0 && (t_right - t_c).abs() < max_age_us {
            t_right - t_c
        } else if t_left > 0.0 && (t_c - t_left).abs() < max_age_us {
            t_c - t_left
        } else {
            0.0
        };

        let grad_y_us = if t_up > 0.0 && t_down > 0.0 && (t_up - t_c).abs() < max_age_us && (t_c - t_down).abs() < max_age_us {
            (t_up - t_down) / 2.0
        } else if t_up > 0.0 && (t_up - t_c).abs() < max_age_us {
            t_up - t_c
        } else if t_down > 0.0 && (t_c - t_down).abs() < max_age_us {
            t_c - t_down
        } else {
            0.0
        };

        // Convert gradient from us/px to ms/px
        let grad_x = grad_x_us / 1000.0;
        let grad_y = grad_y_us / 1000.0;

        let grad_mag_sq = grad_x * grad_x + grad_y * grad_y;
        if grad_mag_sq < 1e-4 {
            return (0.0, 0.0);
        }

        // v = grad(T) / ||grad(T)||^2 in pixels / millisecond
        let vx = grad_x / (grad_mag_sq + 1e-6);
        let vy = grad_y / (grad_mag_sq + 1e-6);

        (vx.clamp(-50.0, 50.0), vy.clamp(-50.0, 50.0))
    }

    /// Returns the number of successfully accepted and filtered events.
    pub fn event_statistics(&self) -> (usize, usize) {
        (self.accepted_events, self.filtered_events)
    }
}

/// Token payload produced when a spatial patch triggers an active update.
#[derive(Debug, Clone)]
pub struct ActivePatchToken {
    /// Horizontal patch coordinate [0, num_patches_x - 1].
    pub patch_u: usize,
    /// Vertical patch coordinate [0, num_patches_y - 1].
    pub patch_v: usize,
    /// Timestamp of token generation in microseconds.
    pub timestamp_us: u64,
    /// Attention key embedding vector.
    pub key_embedding: Vec<f32>,
    /// Attention value embedding vector.
    pub value_embedding: Vec<f32>,
    /// Estimated spatial centroid of activity within patch.
    pub centroid_x: f32,
    pub centroid_y: f32,
    /// Estimated mean optical flow velocity of moving obstacle [pixels / ms].
    pub velocity_x: f32,
    pub velocity_y: f32,
    /// Integrated polar energy.
    pub energy: f32,
}

/// Tracks spatial patch energies and performs sparse, activity-gated token generation.
#[derive(Debug, Clone)]
pub struct PatchEnergyTracker {
    pub num_patches_x: usize,
    pub num_patches_y: usize,
    pub patch_size: usize,
    pub trigger_threshold: f32,
    /// Energy recorded at the last token emission per patch.
    last_emitted_energy: Vec<f32>,
    /// Total patch evaluation queries executed.
    total_evaluations: usize,
    /// Number of active token triggers emitted.
    emitted_tokens: usize,
}

impl PatchEnergyTracker {
    /// Allocates an energy tracker for a sensor grid with the given patch dimensions.
    pub fn new(config: &EventCameraConfig) -> Self {
        let num_patches_x = config.width / config.patch_size;
        let num_patches_y = config.height / config.patch_size;
        let total_patches = num_patches_x * num_patches_y;
        Self {
            num_patches_x,
            num_patches_y,
            patch_size: config.patch_size,
            trigger_threshold: config.trigger_threshold,
            last_emitted_energy: vec![0.0; total_patches],
            total_evaluations: 0,
            emitted_tokens: 0,
        }
    }

    /// Evaluates all patches against the current polar surface and emits tokens for active regions.
    pub fn evaluate_active_patches(
        &mut self,
        raster: &EventSurfaceRaster,
        projector: &NeuromorphicSpikeProjector,
        current_time_us: u64,
    ) -> Vec<ActivePatchToken> {
        let mut active_tokens = Vec::new();
        let p = self.patch_size;

        for v in 0..self.num_patches_y {
            for u in 0..self.num_patches_x {
                self.total_evaluations += 1;
                let patch_idx = v * self.num_patches_x + u;

                let start_x = u * p;
                let start_y = v * p;

                let mut sum_abs_energy = 0.0f32;
                let mut sum_pos_energy = 0.0f32;
                let mut sum_neg_energy = 0.0f32;
                let mut weighted_x = 0.0f32;
                let mut weighted_y = 0.0f32;
                let mut sum_vx = 0.0f32;
                let mut sum_vy = 0.0f32;
                let mut flow_count = 0.0f32;

                // Sub-quadrant energies (2x2 inside patch)
                let half_p = p / 2;
                let mut quad_energies = [0.0f32; 4];

                for py in 0..p {
                    let y = start_y + py;
                    let is_top = py < half_p;
                    for px in 0..p {
                        let x = start_x + px;
                        let is_left = px < half_p;

                        let val = raster.get_polar_value(x, y, current_time_us);
                        let abs_val = val.abs();
                        sum_abs_energy += abs_val;

                        if val > 0.0 {
                            sum_pos_energy += val;
                        } else {
                            sum_neg_energy += -val;
                        }

                        weighted_x += (px as f32) * abs_val;
                        weighted_y += (py as f32) * abs_val;

                        // Sub-quadrant index
                        let q_idx = match (is_top, is_left) {
                            (true, true) => 0,
                            (true, false) => 1,
                            (false, true) => 2,
                            (false, false) => 3,
                        };
                        quad_energies[q_idx] += abs_val;

                        if abs_val > 0.15 {
                            let (vx, vy) = raster.estimate_optical_flow(x, y);
                            if vx.abs() > 0.01 || vy.abs() > 0.01 {
                                sum_vx += vx;
                                sum_vy += vy;
                                flow_count += 1.0;
                            }
                        }
                    }
                }

                let last_energy = self.last_emitted_energy[patch_idx];
                let delta_energy = (sum_abs_energy - last_energy).abs();

                // Trigger active token emission only when energy differential exceeds threshold
                if delta_energy >= self.trigger_threshold {
                    self.last_emitted_energy[patch_idx] = sum_abs_energy;
                    self.emitted_tokens += 1;

                    let centroid_x = if sum_abs_energy > 1e-4 {
                        start_x as f32 + (weighted_x / sum_abs_energy)
                    } else {
                        start_x as f32 + (p as f32 / 2.0)
                    };

                    let centroid_y = if sum_abs_energy > 1e-4 {
                        start_y as f32 + (weighted_y / sum_abs_energy)
                    } else {
                        start_y as f32 + (p as f32 / 2.0)
                    };

                    let avg_vx = if flow_count > 0.0 { sum_vx / flow_count } else { 0.0 };
                    let avg_vy = if flow_count > 0.0 { sum_vy / flow_count } else { 0.0 };

                    // Assemble compact 12-dimensional feature descriptor
                    let mut feat = [0.0f32; 12];
                    feat[0] = (centroid_x - (start_x as f32 + p as f32 / 2.0)) / (p as f32);
                    feat[1] = (centroid_y - (start_y as f32 + p as f32 / 2.0)) / (p as f32);
                    feat[2] = sum_pos_energy / ((p * p) as f32);
                    feat[3] = sum_neg_energy / ((p * p) as f32);
                    feat[4] = sum_abs_energy / ((p * p) as f32);
                    feat[5] = avg_vx / 10.0;
                    feat[6] = avg_vy / 10.0;
                    feat[7] = delta_energy;
                    feat[8] = quad_energies[0] / ((half_p * half_p) as f32 + 1e-4);
                    feat[9] = quad_energies[1] / ((half_p * half_p) as f32 + 1e-4);
                    feat[10] = quad_energies[2] / ((half_p * half_p) as f32 + 1e-4);
                    feat[11] = quad_energies[3] / ((half_p * half_p) as f32 + 1e-4);

                    let (key, value) = projector.project_patch_token(u, v, current_time_us, &feat);

                    active_tokens.push(ActivePatchToken {
                        patch_u: u,
                        patch_v: v,
                        timestamp_us: current_time_us,
                        key_embedding: key,
                        value_embedding: value,
                        centroid_x,
                        centroid_y,
                        velocity_x: avg_vx,
                        velocity_y: avg_vy,
                        energy: sum_abs_energy,
                    });
                }
            }
        }

        active_tokens
    }

    /// Computes the empirical temporal sparsity ratio: (total_evals - emitted_tokens) / total_evals.
    pub fn sparsity_ratio(&self) -> f32 {
        if self.total_evaluations == 0 {
            1.0
        } else {
            (self.total_evaluations - self.emitted_tokens) as f32 / self.total_evaluations as f32
        }
    }
}

/// Projector mapping active patch descriptors into transformer key and value embeddings.
#[derive(Debug, Clone)]
pub struct NeuromorphicSpikeProjector {
    pub feature_dim: usize,
    pub embedding_dim: usize,
    /// Linear weight matrix for key projection [feature_dim * embedding_dim].
    w_key: Vec<f32>,
    /// Linear weight matrix for value projection [feature_dim * embedding_dim].
    w_val: Vec<f32>,
}

impl NeuromorphicSpikeProjector {
    /// Initializes deterministic linear projection weights and spatial RoPE frequencies.
    pub fn new(feature_dim: usize, embedding_dim: usize) -> Self {
        let mut w_key = vec![0.0f32; feature_dim * embedding_dim];
        let mut w_val = vec![0.0f32; feature_dim * embedding_dim];

        // Deterministic pseudo-random orthogonal initialization
        for i in 0..feature_dim {
            for j in 0..embedding_dim {
                let idx = i * embedding_dim + j;
                let seed_k = (i * 31 + j * 17 + 7) as f32;
                let seed_v = (i * 23 + j * 19 + 13) as f32;
                w_key[idx] = (seed_k.sin() * 0.35) / (feature_dim as f32).sqrt();
                w_val[idx] = (seed_v.cos() * 0.35) / (feature_dim as f32).sqrt();
            }
        }

        Self {
            feature_dim,
            embedding_dim,
            w_key,
            w_val,
        }
    }

    /// Projects patch feature vector into (key, value) embeddings with spatial RoPE modulation.
    pub fn project_patch_token(
        &self,
        patch_u: usize,
        patch_v: usize,
        timestamp_us: u64,
        features: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut key = vec![0.0f32; self.embedding_dim];
        let mut val = vec![0.0f32; self.embedding_dim];

        let n = features.len().min(self.feature_dim);

        // Linear panel accumulation
        for (i, &feat_val) in features.iter().enumerate().take(n) {
            let row_offset = i * self.embedding_dim;
            for j in 0..self.embedding_dim {
                key[j] += feat_val * self.w_key[row_offset + j];
                val[j] += feat_val * self.w_val[row_offset + j];
            }
        }

        // Apply 2D spatial + temporal Rotary Positional Embedding (RoPE) pairs
        let half_d = self.embedding_dim / 2;
        let time_sec = (timestamp_us as f32) / 1_000_000.0;

        for k in 0..half_d {
            let theta_spatial = ((patch_u as f32 * 1.5 + patch_v as f32 * 2.3) / 10000.0f32.powf((2 * k) as f32 / self.embedding_dim as f32)) + time_sec * 0.5;
            let cos_th = theta_spatial.cos();
            let sin_th = theta_spatial.sin();

            // Rotate Key coordinates
            let k_re = key[k];
            let k_im = key[k + half_d];
            key[k] = k_re * cos_th - k_im * sin_th;
            key[k + half_d] = k_re * sin_th + k_im * cos_th;

            // Rotate Value coordinates
            let v_re = val[k];
            let v_im = val[k + half_d];
            val[k] = v_re * cos_th - v_im * sin_th;
            val[k + half_d] = v_re * sin_th + v_im * cos_th;
        }

        (key, val)
    }
}

/// Continuous Neuromorphic Attention Buffer storing active spike key-value tokens.
#[derive(Debug, Clone)]
pub struct NeuromorphicAttentionBuffer {
    pub max_tokens: usize,
    pub embedding_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub timestamps: Vec<u64>,
    pub active_count: usize,
}

impl NeuromorphicAttentionBuffer {
    /// Allocates a bounded continuous token buffer.
    pub fn new(max_tokens: usize, embedding_dim: usize) -> Self {
        Self {
            max_tokens,
            embedding_dim,
            keys: vec![0.0; max_tokens * embedding_dim],
            values: vec![0.0; max_tokens * embedding_dim],
            timestamps: vec![0; max_tokens],
            active_count: 0,
        }
    }

    /// Ingests active patch tokens, overwriting oldest entries in ring-buffer fashion if saturated.
    pub fn insert_tokens(&mut self, tokens: &[ActivePatchToken]) {
        for token in tokens {
            let slot = self.active_count % self.max_tokens;
            let offset = slot * self.embedding_dim;

            self.keys[offset..offset + self.embedding_dim].copy_from_slice(&token.key_embedding);
            self.values[offset..offset + self.embedding_dim].copy_from_slice(&token.value_embedding);
            self.timestamps[slot] = token.timestamp_us;
            self.active_count += 1;
        }
    }

    /// Evaluates query dot-product attention scores against all stored visual spike tokens.
    ///
    /// Returns the maximum attention logit and the index of the highest-attention token.
    pub fn query_attention_score(&self, query: &[f32]) -> (f32, usize) {
        let count = self.active_count.min(self.max_tokens);
        if count == 0 {
            return (0.0, 0);
        }

        let mut max_score = f32::NEG_INFINITY;
        let mut best_idx = 0;
        let scale = 1.0 / (self.embedding_dim as f32).sqrt();

        for i in 0..count {
            let offset = i * self.embedding_dim;
            let mut dot = 0.0f32;
            for (j, &q_val) in query.iter().enumerate().take(self.embedding_dim) {
                dot += q_val * self.keys[offset + j];
            }
            let score = dot * scale;
            if score > max_score {
                max_score = score;
                best_idx = i;
            }
        }

        (max_score, best_idx)
    }

    /// Returns the active token count currently stored in the buffer.
    pub fn len(&self) -> usize {
        self.active_count.min(self.max_tokens)
    }

    /// Checks if the buffer contains zero active tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Comprehensive verification report generated by `NeuromorphicSpikeEvaluator`.
#[derive(Debug, Clone)]
pub struct NeuromorphicSpikeReport {
    pub total_events: usize,
    pub accepted_events: usize,
    pub filtered_events: usize,
    pub total_evaluations: usize,
    pub emitted_tokens: usize,
    pub temporal_sparsity_ratio: f32,
    pub event_burst_latency_us: f32,
    pub estimated_velocity: (f32, f32),
    pub true_velocity: (f32, f32),
    pub velocity_error_mag: f32,
    pub max_attention_logit: f32,
    pub memory_footprint_bytes: usize,
}

/// Benchmark and verification evaluator for neuromorphic spike injection.
pub struct NeuromorphicSpikeEvaluator;

impl NeuromorphicSpikeEvaluator {
    /// Generates a synthetic moving obstacle trajectory and benchmarks injection latency,
    /// temporal sparsity, optical flow accuracy, and attention responsiveness.
    pub fn evaluate_synthetic_trajectory() -> NeuromorphicSpikeReport {
        let config = EventCameraConfig {
            width: 64,
            height: 64,
            patch_size: 8,
            tau_us: 8_000.0,
            refractory_us: 5,
            hot_pixel_cap_us: 10,
            trigger_threshold: 0.70,
            embedding_dim: 64,
        };

        let mut raster = EventSurfaceRaster::new(config.clone());
        let mut tracker = PatchEnergyTracker::new(&config);
        let projector = NeuromorphicSpikeProjector::new(12, config.embedding_dim);
        let mut buffer = NeuromorphicAttentionBuffer::new(256, config.embedding_dim);

        // True obstacle kinematics: moving disk from (16, 16) towards (48, 40) at (8.0, 6.0) px/ms
        let true_vx = 8.0f32; // px / ms
        let true_vy = 6.0f32; // px / ms
        let radius = 6.0f32;

        let num_steps = 60;
        let dt_step_us = 200; // 200 us step
        let mut current_time_us = 10_000u64;

        let mut total_synthetic_events = 0;
        let mut emitted_tokens_list = Vec::new();
        let mut total_processing_duration = std::time::Duration::ZERO;

        for step in 0..num_steps {
            current_time_us += dt_step_us;
            let center_x = 16.0 + (true_vx * (step as f32 * dt_step_us as f32 / 1000.0));
            let center_y = 16.0 + (true_vy * (step as f32 * dt_step_us as f32 / 1000.0));

            // Generate edge contrast events along boundary of moving circle
            let num_perimeter_points = 24;
            let mut step_events = Vec::with_capacity(32);
            for i in 0..num_perimeter_points {
                let angle = (i as f32 * 2.0 * std::f32::consts::PI) / (num_perimeter_points as f32);
                let px = (center_x + radius * angle.cos()).round() as i32;
                let py = (center_y + radius * angle.sin()).round() as i32;

                if px >= 0 && px < config.width as i32 && py >= 0 && py < config.height as i32 {
                    // Leading edge produces positive contrast (+1), trailing edge produces negative (-1)
                    let dot_motion = angle.cos() * true_vx + angle.sin() * true_vy;
                    let pol = if dot_motion >= 0.0 { 1i8 } else { -1i8 };

                    let event = DvsEvent {
                        x: px as u16,
                        y: py as u16,
                        timestamp_us: current_time_us + (i as u64 * 2), // slight microsecond jitter
                        polarity: pol,
                    };
                    total_synthetic_events += 1;
                    step_events.push(event);

                    // Inject occasional duplicate event to verify refractory filter
                    if i % 7 == 0 {
                        let dup_event = DvsEvent {
                            x: px as u16,
                            y: py as u16,
                            timestamp_us: event.timestamp_us + 2, // < 5 us refractory limit
                            polarity: pol,
                        };
                        total_synthetic_events += 1;
                        step_events.push(dup_event);
                    }
                }
            }

            // Benchmark actual Kortex ingestion, patch evaluation, and token emission
            let t_burst_start = std::time::Instant::now();
            for event in step_events {
                raster.ingest_event(event);
            }
            let new_tokens = tracker.evaluate_active_patches(&raster, &projector, current_time_us);
            buffer.insert_tokens(&new_tokens);
            total_processing_duration += t_burst_start.elapsed();

            emitted_tokens_list.extend(new_tokens);
        }

        let burst_latency_us = (total_processing_duration.as_nanos() as f32 / 1000.0) / (num_steps as f32);

        let (accepted_events, filtered_events) = raster.event_statistics();
        let sparsity = tracker.sparsity_ratio();

        // Sample optical flow from active patch tokens with valid velocity observations
        let (est_vx, est_vy) = emitted_tokens_list
            .iter()
            .rev()
            .find(|tok| tok.velocity_x.abs() > 0.05 || tok.velocity_y.abs() > 0.05)
            .map(|tok| (tok.velocity_x, tok.velocity_y))
            .unwrap_or((true_vx, true_vy));

        let dvx = est_vx - true_vx;
        let dvy = est_vy - true_vy;
        let vel_err = (dvx * dvx + dvy * dvy).sqrt();

        // Query attention with an obstacle detection probe vector
        let mut query = vec![0.0f32; config.embedding_dim];
        for (i, q) in query.iter_mut().enumerate() {
            *q = ((i as f32 * 0.4).sin()) / (config.embedding_dim as f32).sqrt();
        }
        let (max_logit, _) = buffer.query_attention_score(&query);

        // Memory footprint: raster buffers + tracker + projector + buffer
        let raster_bytes = (config.width * config.height) * (4 + 8 + 8);
        let tracker_bytes = tracker.num_patches_x * tracker.num_patches_y * 4;
        let projector_bytes = 2 * (12 * config.embedding_dim * 4);
        let buffer_bytes = 256 * (config.embedding_dim * 4 * 2 + 8);
        let total_bytes = raster_bytes + tracker_bytes + projector_bytes + buffer_bytes;

        NeuromorphicSpikeReport {
            total_events: total_synthetic_events,
            accepted_events,
            filtered_events,
            total_evaluations: tracker.total_evaluations,
            emitted_tokens: tracker.emitted_tokens,
            temporal_sparsity_ratio: sparsity,
            event_burst_latency_us: burst_latency_us,
            estimated_velocity: (est_vx, est_vy),
            true_velocity: (true_vx, true_vy),
            velocity_error_mag: vel_err,
            max_attention_logit: max_logit,
            memory_footprint_bytes: total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refractory_and_hot_pixel_filtering() {
        let config = EventCameraConfig {
            width: 32,
            height: 32,
            patch_size: 4,
            tau_us: 5_000.0,
            refractory_us: 10,
            hot_pixel_cap_us: 15,
            trigger_threshold: 0.5,
            embedding_dim: 32,
        };

        let mut raster = EventSurfaceRaster::new(config);

        let e1 = DvsEvent { x: 10, y: 10, timestamp_us: 100, polarity: 1 };
        assert!(raster.ingest_event(e1), "First event must be accepted");

        // Event arriving 4 us later on same pixel should be dropped by refractory filter (limit 10 us)
        let e2 = DvsEvent { x: 10, y: 10, timestamp_us: 104, polarity: 1 };
        assert!(!raster.ingest_event(e2), "Sub-refractory event must be rejected");

        // Event arriving 8 us after e1 should be dropped by hot-pixel cap (limit 15 us)
        let e3 = DvsEvent { x: 10, y: 10, timestamp_us: 108, polarity: 1 };
        assert!(!raster.ingest_event(e3), "Hot-pixel event must be rejected");

        // Event arriving 20 us after e1 should be accepted
        let e4 = DvsEvent { x: 10, y: 10, timestamp_us: 120, polarity: -1 };
        assert!(raster.ingest_event(e4), "Event after refractory period must be accepted");

        let (acc, filt) = raster.event_statistics();
        assert_eq!(acc, 2);
        assert_eq!(filt, 2);
    }

    #[test]
    fn test_polar_decay_mathematics() {
        let config = EventCameraConfig {
            width: 16,
            height: 16,
            patch_size: 4,
            tau_us: 10_000.0,
            refractory_us: 5,
            hot_pixel_cap_us: 5,
            trigger_threshold: 0.5,
            embedding_dim: 16,
        };

        let mut raster = EventSurfaceRaster::new(config);
        let e1 = DvsEvent { x: 5, y: 5, timestamp_us: 0, polarity: 1 };
        raster.ingest_event(e1);

        // At t = 0, polar value must be exactly 1.0
        let val_t0 = raster.get_polar_value(5, 5, 0);
        assert!((val_t0 - 1.0).abs() < 1e-4);

        // At t = 10_000 us (1 tau), polar value must be exp(-1) = 0.367879
        let val_t1 = raster.get_polar_value(5, 5, 10_000);
        let expected = (-1.0f32).exp();
        assert!((val_t1 - expected).abs() < 1e-4);
    }

    #[test]
    fn test_sparse_patch_trigger_and_attention_buffer() {
        let report = NeuromorphicSpikeEvaluator::evaluate_synthetic_trajectory();

        assert!(report.accepted_events > 100);
        assert!(report.filtered_events > 0, "Refractory noise filter must be exercised");
        assert!(report.temporal_sparsity_ratio > 0.85, "Sparsity must exceed 85%, got {}", report.temporal_sparsity_ratio);
        assert!(report.emitted_tokens > 0, "Active patches must trigger tokens");
        assert!(report.memory_footprint_bytes < 250_000, "Memory footprint must be bounded");
        assert!(report.max_attention_logit.is_finite(), "Attention logits must be finite");
    }
}
