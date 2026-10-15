//! Information-Theoretic Active Perception & Sensor Saccade Planner Engine
//!
//! Subsystem Capabilities:
//! - Construction of a discrete spherical directional information entropy field H(theta, phi)
//!   fusing Sonon acoustic Direction-of-Arrival (DoA) and neuromorphic DVS event patch energy.
//! - Directional Mutual Information maximization Delta I(theta, phi) balancing expected entropy
//!   reduction against mechanical actuator effort and angular displacement.
//! - Analytical 5th-order minimum-jerk saccade polynomial trajectory generation
//!   satisfying zero boundary velocity and acceleration conditions to suppress motion blur.
//! - Attention token saliency modulation prioritizing high-information gaze sectors in KV-cache.
//!
//! Strict Mandates:
//! - `#![deny(unsafe_code)]`
//! - Zero unicode emojis across all code, tests, and outputs
//! - Bounded static allocations and deterministic sub-10 microsecond execution

#![deny(unsafe_code)]

use std::f32::consts::PI;

/// Number of discrete azimuth bins spanning [-pi, pi).
pub const AZIMUTH_BINS: usize = 16;
/// Number of discrete elevation bins spanning [-pi/2, pi/2).
pub const ELEVATION_BINS: usize = 8;
/// Total number of discrete directional cells on the sphere S^2.
pub const TOTAL_DIRECTIONAL_BINS: usize = AZIMUTH_BINS * ELEVATION_BINS; // 128

/// Converts spherical angles (azimuth, elevation) in radians to a 3D unit direction vector.
#[inline]
pub fn spherical_to_cartesian(azimuth_rad: f32, elevation_rad: f32) -> [f32; 3] {
    let cos_el = elevation_rad.cos();
    [
        cos_el * azimuth_rad.cos(),
        cos_el * azimuth_rad.sin(),
        elevation_rad.sin(),
    ]
}

/// Computes the great-circle geodesic angular distance between two spherical points.
#[inline]
pub fn great_circle_distance(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    let u1 = spherical_to_cartesian(az1, el1);
    let u2 = spherical_to_cartesian(az2, el2);
    let dot = (u1[0] * u2[0] + u1[1] * u2[1] + u1[2] * u2[2]).clamp(-1.0, 1.0);
    dot.acos()
}

/// Maps a directional bin index to its center spherical angles (azimuth, elevation).
#[inline]
pub fn bin_to_angles(bin_idx: usize) -> (f32, f32) {
    let az_idx = bin_idx % AZIMUTH_BINS;
    let el_idx = bin_idx / AZIMUTH_BINS;

    let az = -PI + (2.0 * PI / AZIMUTH_BINS as f32) * (az_idx as f32 + 0.5);
    let el = -0.5 * PI + (PI / ELEVATION_BINS as f32) * (el_idx as f32 + 0.5);
    (az, el)
}

/// Precomputed geometry for a directional cell on S^2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalCell {
    pub azimuth_rad: f32,
    pub elevation_rad: f32,
    pub unit_vector: [f32; 3],
}

/// Returns precomputed directional cells for fast inner-product operations without trigonometric overhead.
pub fn get_directional_cells() -> &'static [DirectionalCell; TOTAL_DIRECTIONAL_BINS] {
    static CELLS: std::sync::OnceLock<[DirectionalCell; TOTAL_DIRECTIONAL_BINS]> = std::sync::OnceLock::new();
    CELLS.get_or_init(|| {
        let mut arr = [DirectionalCell {
            azimuth_rad: 0.0,
            elevation_rad: 0.0,
            unit_vector: [1.0, 0.0, 0.0],
        }; TOTAL_DIRECTIONAL_BINS];
        for (i, cell) in arr.iter_mut().enumerate() {
            let (az, el) = bin_to_angles(i);
            let u = spherical_to_cartesian(az, el);
            *cell = DirectionalCell {
                azimuth_rad: az,
                elevation_rad: el,
                unit_vector: u,
            };
        }
        arr
    })
}

/// Maps spherical angles (azimuth, elevation) to the nearest directional bin index.
#[inline]
pub fn angles_to_bin(azimuth_rad: f32, elevation_rad: f32) -> usize {
    let mut az_norm = (azimuth_rad + PI) / (2.0 * PI);
    az_norm = az_norm - az_norm.floor();
    let az_idx = ((az_norm * AZIMUTH_BINS as f32) as usize).min(AZIMUTH_BINS - 1);

    let el_norm = ((elevation_rad + 0.5 * PI) / PI).clamp(0.0, 0.9999);
    let el_idx = ((el_norm * ELEVATION_BINS as f32) as usize).min(ELEVATION_BINS - 1);

    el_idx * AZIMUTH_BINS + az_idx
}

/// Gimbal / camera gaze orientation in spherical coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GimbalOrientation {
    /// Azimuth angle in radians in [-pi, pi].
    pub azimuth_rad: f32,
    /// Elevation angle in radians in [-pi/2, pi/2].
    pub elevation_rad: f32,
}

impl Default for GimbalOrientation {
    fn default() -> Self {
        Self {
            azimuth_rad: 0.0,
            elevation_rad: 0.0,
        }
    }
}

impl GimbalOrientation {
    /// Creates a new gimbal orientation.
    pub fn new(azimuth_rad: f32, elevation_rad: f32) -> Self {
        Self {
            azimuth_rad: azimuth_rad.clamp(-PI, PI),
            elevation_rad: elevation_rad.clamp(-0.5 * PI, 0.5 * PI),
        }
    }

    /// Converts this orientation to a 3D unit gaze vector.
    pub fn to_unit_vector(&self) -> [f32; 3] {
        spherical_to_cartesian(self.azimuth_rad, self.elevation_rad)
    }

    /// Computes the great-circle angular distance to another orientation.
    pub fn angular_distance_to(&self, other: &GimbalOrientation) -> f32 {
        great_circle_distance(
            self.azimuth_rad,
            self.elevation_rad,
            other.azimuth_rad,
            other.elevation_rad,
        )
    }
}

/// Discrete spherical directional information entropy field.
#[derive(Debug, Clone)]
pub struct DirectionalEntropyField {
    /// Normalized probability distribution across all 128 directional cells.
    pub probabilities: [f32; TOTAL_DIRECTIONAL_BINS],
    /// Total Shannon prior entropy of the directional belief in bits.
    pub prior_entropy_bits: f32,
}

impl DirectionalEntropyField {
    /// Computes the Shannon entropy in bits given the probability array.
    pub fn compute_entropy(probs: &[f32; TOTAL_DIRECTIONAL_BINS]) -> f32 {
        let mut h = 0.0f32;
        for &p in probs.iter() {
            if p > 1e-12 {
                h -= p * p.log2();
            }
        }
        h
    }
}

/// High-level saccade command selected by mutual information optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaccadeCommand {
    /// Target orientation for the gimbal / sensor gaze.
    pub target: GimbalOrientation,
    /// Geodesic angular distance in radians from start to target.
    pub angular_distance_rad: f32,
    /// Expected information gain (entropy reduction minus mechanical effort cost).
    pub expected_info_gain: f32,
    /// Planned saccade duration in seconds.
    pub saccade_duration_sec: f32,
    /// Optimal directional bin index.
    pub optimal_bin_index: usize,
}

/// Kinematic sample along a trajectory including position, velocity, acceleration, and jerk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    pub position: GimbalOrientation,
    pub velocity: (f32, f32),
    pub acceleration: (f32, f32),
    pub jerk: (f32, f32),
}

/// Fifth-order minimum-jerk trajectory representation for camera saccades.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumJerkTrajectory {
    pub start: GimbalOrientation,
    pub target: GimbalOrientation,
    pub duration_sec: f32,
}

impl MinimumJerkTrajectory {
    /// Samples position, angular velocity, angular acceleration, and angular jerk at time t.
    ///
    /// Analytical formulas for normalized time tau = t / T:
    /// s(tau)   = 10*tau^3 - 15*tau^4 + 6*tau^5
    /// s'(tau)  = 30*tau^2 - 60*tau^3 + 30*tau^4
    /// s''(tau) = 60*tau - 180*tau^2 + 120*tau^3
    /// s'''(tau)= 60 - 360*tau + 360*tau^2
    pub fn sample(&self, t_sec: f32) -> TrajectorySample {
        let t = t_sec.clamp(0.0, self.duration_sec);
        let tau = if self.duration_sec > 1e-6 {
            t / self.duration_sec
        } else {
            1.0
        };

        let tau2 = tau * tau;
        let tau3 = tau2 * tau;
        let tau4 = tau3 * tau;
        let tau5 = tau4 * tau;

        // Position polynomial: s(tau)
        let s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5;

        // First derivative: s'(tau) / T
        let inv_t = if self.duration_sec > 1e-6 { 1.0 / self.duration_sec } else { 0.0 };
        let ds = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) * inv_t;

        // Second derivative: s''(tau) / T^2
        let inv_t2 = inv_t * inv_t;
        let dds = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) * inv_t2;

        // Third derivative (jerk): s'''(tau) / T^3
        let inv_t3 = inv_t2 * inv_t;
        let ddds = (60.0 - 360.0 * tau + 360.0 * tau2) * inv_t3;

        let delta_az = self.target.azimuth_rad - self.start.azimuth_rad;
        let delta_el = self.target.elevation_rad - self.start.elevation_rad;

        let az = self.start.azimuth_rad + delta_az * s;
        let el = self.start.elevation_rad + delta_el * s;

        let vel = (delta_az * ds, delta_el * ds);
        let acc = (delta_az * dds, delta_el * dds);
        let jerk = (delta_az * ddds, delta_el * ddds);

        TrajectorySample {
            position: GimbalOrientation::new(az, el),
            velocity: vel,
            acceleration: acc,
            jerk,
        }
    }

    /// Evaluates the maximum total angular jerk magnitude along the trajectory.
    pub fn max_jerk_magnitude(&self) -> f32 {
        let delta_az = self.target.azimuth_rad - self.start.azimuth_rad;
        let delta_el = self.target.elevation_rad - self.start.elevation_rad;
        let angular_distance = (delta_az * delta_az + delta_el * delta_el).sqrt();

        if self.duration_sec <= 1e-6 {
            return 0.0;
        }

        // Maximum of |60 - 360*tau + 360*tau^2| occurs at tau=0 (value 60) or tau=0.5 (value |-30| = 30)
        let max_poly = 60.0f32;
        let inv_t3 = 1.0 / (self.duration_sec * self.duration_sec * self.duration_sec);
        angular_distance * max_poly * inv_t3
    }
}

/// Active perception and saccade planning engine.
#[derive(Debug, Clone)]
pub struct ActivePerceptionPlanner {
    /// Current physical gaze orientation.
    pub current_gaze: GimbalOrientation,
    /// Penalty weight for mechanical saccade angular displacement (lambda_effort).
    pub effort_weight: f32,
    /// Expected information reduction factor for foveal visual fixation.
    pub fovea_gain: f32,
    /// Minimum mechanical saccade duration in seconds.
    pub min_duration_sec: f32,
    /// Duration scaling factor per radian of angular displacement.
    pub duration_scale: f32,
    /// Attention key modulation saliency scaling factor.
    pub saliency_weight: f32,
}

impl Default for ActivePerceptionPlanner {
    fn default() -> Self {
        Self {
            current_gaze: GimbalOrientation::default(),
            effort_weight: 0.15,
            fovea_gain: 2.0,
            min_duration_sec: 0.05,
            duration_scale: 0.10,
            saliency_weight: 0.50,
        }
    }
}

impl ActivePerceptionPlanner {
    /// Creates a new active perception planner with specific parameters.
    pub fn new(current_gaze: GimbalOrientation) -> Self {
        Self {
            current_gaze,
            ..Default::default()
        }
    }

    /// Constructs the discrete spherical directional entropy field fusing multi-modal inputs.
    pub fn update_entropy_field(
        &self,
        acoustic_bearing: &[f32; 3],
        acoustic_energy: f32,
        visual_patches: &[(f32, f32, f32)], // (azimuth_rad, elevation_rad, patch_energy)
    ) -> DirectionalEntropyField {
        let cells = get_directional_cells();
        let mut raw_scores = [0.0f32; TOTAL_DIRECTIONAL_BINS];
        let epsilon_prior = 0.01f32;

        let acoustic_norm = (acoustic_bearing[0] * acoustic_bearing[0]
            + acoustic_bearing[1] * acoustic_bearing[1]
            + acoustic_bearing[2] * acoustic_bearing[2])
            .sqrt()
            .max(1e-6);
        let b_a = [
            acoustic_bearing[0] / acoustic_norm,
            acoustic_bearing[1] / acoustic_norm,
            acoustic_bearing[2] / acoustic_norm,
        ];

        // Precompute Cartesian unit vectors for the visual patches (up to 8)
        let mut visual_vectors = [( [0.0f32; 3], 0.0f32 ); 8];
        let n_patches = visual_patches.len().min(8);
        for (idx, patch) in visual_patches.iter().take(n_patches).enumerate() {
            visual_vectors[idx] = (spherical_to_cartesian(patch.0, patch.1), patch.2);
        }

        let mut sum_scores = 0.0f32;

        for (bin_idx, cell) in cells.iter().enumerate() {
            let u = cell.unit_vector;

            // 1. Acoustic directional component (von Mises-Fisher cosine kernel)
            let cos_theta = (b_a[0] * u[0] + b_a[1] * u[1] + b_a[2] * u[2]).max(0.0);
            let s_acoustic = acoustic_energy * cos_theta.powi(4);

            // 2. Visual patch directional component (Gaussian angular distance kernel)
            let mut s_visual = 0.0f32;
            for &(v_u, v_energy) in visual_vectors[..n_patches].iter() {
                let dot_v = (u[0] * v_u[0] + u[1] * v_u[1] + u[2] * v_u[2]).clamp(-1.0, 1.0);
                let dist = dot_v.acos();
                let kernel = (-0.5 * (dist / 0.25).powi(2)).exp();
                s_visual += v_energy * kernel;
            }

            let score = 0.40 * s_acoustic + 0.60 * s_visual + epsilon_prior;
            raw_scores[bin_idx] = score;
            sum_scores += score;
        }

        let inv_sum = 1.0 / sum_scores.max(1e-6);
        let mut probs = [0.0f32; TOTAL_DIRECTIONAL_BINS];
        for i in 0..TOTAL_DIRECTIONAL_BINS {
            probs[i] = raw_scores[i] * inv_sum;
        }

        let prior_entropy = DirectionalEntropyField::compute_entropy(&probs);

        DirectionalEntropyField {
            probabilities: probs,
            prior_entropy_bits: prior_entropy,
        }
    }

    /// Plans the optimal sensor saccade target maximizing directional mutual information.
    pub fn plan_saccade(&self, field: &DirectionalEntropyField) -> (SaccadeCommand, MinimumJerkTrajectory) {
        let cells = get_directional_cells();
        let u_gaze = self.current_gaze.to_unit_vector();

        let mut best_bin = 0;
        let mut best_info_gain = -1e9f32;
        let mut best_target = self.current_gaze;
        let mut best_dist = 0.0f32;

        for (bin_idx, cell) in cells.iter().enumerate() {
            let u = cell.unit_vector;
            let dot_gaze = (u_gaze[0] * u[0] + u_gaze[1] * u[1] + u_gaze[2] * u[2]).clamp(-1.0, 1.0);
            let dist = dot_gaze.acos();

            // Expected entropy reduction: foveal inspection resolves uncertainty in neighboring cells
            let p_target = field.probabilities[bin_idx];
            let delta_h = p_target * (1.0 + self.fovea_gain * p_target).log2();

            // Mutual information objective: information gain penalized by actuator travel effort
            let info_gain = delta_h - self.effort_weight * dist;

            if info_gain > best_info_gain {
                best_info_gain = info_gain;
                best_bin = bin_idx;
                best_target = GimbalOrientation::new(cell.azimuth_rad, cell.elevation_rad);
                best_dist = dist;
            }
        }

        let duration = self.min_duration_sec + self.duration_scale * best_dist;

        let cmd = SaccadeCommand {
            target: best_target,
            angular_distance_rad: best_dist,
            expected_info_gain: best_info_gain,
            saccade_duration_sec: duration,
            optimal_bin_index: best_bin,
        };

        let traj = MinimumJerkTrajectory {
            start: self.current_gaze,
            target: best_target,
            duration_sec: duration,
        };

        (cmd, traj)
    }

    /// Modulates transformer key activations by spatial information entropy saliency.
    pub fn modulate_key(
        &self,
        key: &mut [f32],
        token_bearing: (f32, f32),
        field: &DirectionalEntropyField,
    ) {
        let bin_idx = angles_to_bin(token_bearing.0, token_bearing.1);
        let p = field.probabilities[bin_idx];
        let max_prob = 1.0 / (TOTAL_DIRECTIONAL_BINS as f32);
        let saliency = (p / max_prob).min(4.0);

        let scale = 1.0 + self.saliency_weight * (saliency - 1.0);
        for x in key.iter_mut() {
            *x *= scale;
        }
    }
}

/// Comprehensive benchmark report for the Active Perception & Saccade Planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivePerceptionReport {
    /// Wall-clock latency for entropy evaluation and saccade planning in microseconds.
    pub planning_latency_us: f64,
    /// Initial directional entropy before saccade planning in bits.
    pub prior_entropy_bits: f32,
    /// Expected posterior directional entropy after foveal gaze fixation in bits.
    pub post_saccade_entropy_bits: f32,
    /// Percentage reduction in directional uncertainty.
    pub entropy_reduction_pct: f32,
    /// Planned saccade angular distance in degrees.
    pub saccade_angular_distance_deg: f32,
    /// Maximum jerk magnitude along the minimum-jerk trajectory in rad/s^3.
    pub max_jerk_rad_s3: f32,
    /// Maximum velocity error at boundary points t=0 and t=T.
    pub boundary_velocity_error: f32,
    /// Maximum acceleration error at boundary points t=0 and t=T.
    pub boundary_acceleration_error: f32,
    /// Total static memory footprint in bytes.
    pub total_memory_footprint_bytes: usize,
}

/// Verification harness and empirical benchmark evaluator.
pub struct ActivePerceptionEvaluator;

impl ActivePerceptionEvaluator {
    /// Runs empirical benchmarks validating the active perception engine.
    pub fn evaluate() -> ActivePerceptionReport {
        let planner = ActivePerceptionPlanner::new(GimbalOrientation::new(0.0, 0.0));

        // Simulated directional inputs: acoustic detection at azimuth +45 deg, elevation +20 deg
        let acoustic_bearing = spherical_to_cartesian(45.0 * PI / 180.0, 20.0 * PI / 180.0);
        let acoustic_energy = 0.85f32;

        // Visual patch at azimuth +50 deg, elevation +18 deg
        let visual_patches = [
            (50.0 * PI / 180.0, 18.0 * PI / 180.0, 0.90f32),
            (-30.0 * PI / 180.0, -10.0 * PI / 180.0, 0.15f32),
        ];

        // Benchmark timing over 1,000 full perception-action cycles
        let start = std::time::Instant::now();
        let iterations = 1000;
        let mut last_cmd = None;
        let mut last_traj = None;
        let mut last_field = None;

        for _ in 0..iterations {
            let field = planner.update_entropy_field(&acoustic_bearing, acoustic_energy, &visual_patches);
            let (cmd, traj) = planner.plan_saccade(&field);
            last_cmd = Some(cmd);
            last_traj = Some(traj);
            last_field = Some(field);
        }

        let elapsed = start.elapsed();
        let latency_us = (elapsed.as_nanos() as f64 / iterations as f64) / 1000.0;

        let cmd = last_cmd.unwrap();
        let traj = last_traj.unwrap();
        let field = last_field.unwrap();

        // Verify boundary conditions on the trajectory
        let sample_0 = traj.sample(0.0);
        let sample_t = traj.sample(traj.duration_sec);

        let vel_err = sample_0.velocity.0.abs()
            .max(sample_0.velocity.1.abs())
            .max(sample_t.velocity.0.abs())
            .max(sample_t.velocity.1.abs());
        let acc_err = sample_0.acceleration.0.abs()
            .max(sample_0.acceleration.1.abs())
            .max(sample_t.acceleration.0.abs())
            .max(sample_t.acceleration.1.abs());

        let max_jerk = traj.max_jerk_magnitude();

        // Simulate posterior entropy reduction after gaze fixation on best_bin
        let mut posterior_probs = field.probabilities;
        let best_bin = cmd.optimal_bin_index;
        // Fixation reduces uncertainty in the foveated bin by concentrating mass
        posterior_probs[best_bin] *= 1.0 + planner.fovea_gain;
        let sum_p: f32 = posterior_probs.iter().sum();
        for p in posterior_probs.iter_mut() {
            *p /= sum_p;
        }
        let post_entropy = DirectionalEntropyField::compute_entropy(&posterior_probs);
        let entropy_reduction_pct = ((field.prior_entropy_bits - post_entropy) / field.prior_entropy_bits) * 100.0;

        let mem_bytes = std::mem::size_of::<ActivePerceptionPlanner>()
            + std::mem::size_of::<DirectionalEntropyField>()
            + std::mem::size_of::<MinimumJerkTrajectory>()
            + std::mem::size_of::<SaccadeCommand>();

        ActivePerceptionReport {
            planning_latency_us: latency_us,
            prior_entropy_bits: field.prior_entropy_bits,
            post_saccade_entropy_bits: post_entropy,
            entropy_reduction_pct,
            saccade_angular_distance_deg: cmd.angular_distance_rad * 180.0 / PI,
            max_jerk_rad_s3: max_jerk,
            boundary_velocity_error: vel_err,
            boundary_acceleration_error: acc_err,
            total_memory_footprint_bytes: mem_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_conversions_and_distances() {
        let u_fwd = spherical_to_cartesian(0.0, 0.0);
        assert!((u_fwd[0] - 1.0).abs() < 1e-6);
        assert!(u_fwd[1].abs() < 1e-6);
        assert!(u_fwd[2].abs() < 1e-6);

        let dist_90 = great_circle_distance(0.0, 0.0, 0.5 * PI, 0.0);
        assert!((dist_90 - 0.5 * PI).abs() < 1e-5);

        for bin in 0..TOTAL_DIRECTIONAL_BINS {
            let (az, el) = bin_to_angles(bin);
            let recovered_bin = angles_to_bin(az, el);
            assert_eq!(recovered_bin, bin);
        }
    }

    #[test]
    fn test_minimum_jerk_boundary_conditions() {
        let start = GimbalOrientation::new(0.0, 0.0);
        let target = GimbalOrientation::new(0.8, -0.4);
        let traj = MinimumJerkTrajectory {
            start,
            target,
            duration_sec: 0.15,
        };

        let sample_0 = traj.sample(0.0);
        assert!((sample_0.position.azimuth_rad - start.azimuth_rad).abs() < 1e-6);
        assert!((sample_0.position.elevation_rad - start.elevation_rad).abs() < 1e-6);
        assert!(sample_0.velocity.0.abs() < 1e-5);
        assert!(sample_0.velocity.1.abs() < 1e-5);
        assert!(sample_0.acceleration.0.abs() < 1e-5);
        assert!(sample_0.acceleration.1.abs() < 1e-5);

        let sample_t = traj.sample(0.15);
        assert!((sample_t.position.azimuth_rad - target.azimuth_rad).abs() < 1e-6);
        assert!((sample_t.position.elevation_rad - target.elevation_rad).abs() < 1e-6);
        assert!(sample_t.velocity.0.abs() < 1e-5);
        assert!(sample_t.velocity.1.abs() < 1e-5);
        assert!(sample_t.acceleration.0.abs() < 1e-5);
        assert!(sample_t.acceleration.1.abs() < 1e-5);

        assert!(traj.max_jerk_magnitude() > 0.0);
    }

    #[test]
    fn test_active_perception_evaluator_metrics() {
        let report = ActivePerceptionEvaluator::evaluate();
        assert!(report.planning_latency_us < 10.0, "Latency must be < 10 us, got {}", report.planning_latency_us);
        assert!(report.boundary_velocity_error < 1e-5, "Boundary velocity error must be < 1e-5");
        assert!(report.boundary_acceleration_error < 1e-5, "Boundary acceleration error must be < 1e-5");
        assert!(report.post_saccade_entropy_bits < report.prior_entropy_bits, "Entropy must decrease");
        assert!(report.entropy_reduction_pct > 0.0, "Entropy reduction must be positive");
        assert!(report.saccade_angular_distance_deg > 0.0);
        assert!(report.total_memory_footprint_bytes < 16_000);
    }
}
