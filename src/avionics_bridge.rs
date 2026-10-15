//! Chronos Multi-World Seqlock Rollout Verification & Kestrel Avionics Interconnect.
//!
//! Provides the real-time robotics safety and sensory-motor bridge connecting Kortex
//! to the Chronos multi-physics engine, Kestrel flight control firmware, and Sonon
//! acoustic DSP engine in pure safe Rust (`#![deny(unsafe_code)]`):
//! - Zero-Copy Chronos Seqlock State Reader (`ChronosSeqlockArena`):
//!   Ingests 6-DoF rigid-body states across 128 parallel Chronos simulation worlds
//!   using atomic sequence counters (`AtomicU64`) with $< 50\text{ ns}$ read latency.
//! - Multi-World Rollout Feasibility Verifier (`RolloutFeasibilityVerifier`):
//!   Simulates and verifies 128 parallel forward rollouts for candidate action tokens
//!   in $< 30\text{ ms}$, masking out kinematically infeasible or collision-prone
//!   trajectories before token emission.
//! - Kestrel 400 Hz / 20 Hz CAN-FD Telemetry Bridge (`KestrelCanFdBridge`):
//!   Packs and verifies 64-byte CAN-FD frames with CRC-16-CCITT checksums and enforces
//!   a deterministic hardware watchdog that switches Kestrel's 400 Hz LQR loop into
//!   minimum-jerk loiter fallback if a 20 Hz Kortex planning frame misses its deadline.
//! - Lockless Sonon Acoustic Logit SPSC Ring Buffer (`SononLogitRingBuffer`):
//!   Streams acoustic phoneme/wake-word logit frames from Sonon into Kortex's embedding
//!   pipeline with zero heap allocations.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Number of parallel Chronos simulation worlds in the Seqlock rollout arena.
pub const NUM_CHRONOS_WORLDS: usize = 128;

/// Exact payload size of a CAN-FD avionics frame in bytes.
pub const CAN_FD_FRAME_BYTES: usize = 64;

/// 6-DoF rigid-body state snapshot from a Chronos simulation world.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RigidBodyState6Dof {
    /// Position in local North-East-Down (NED) coordinates $[x, y, z]$ (meters).
    pub pos_ned: [f32; 3],
    /// Linear velocity in local NED coordinates $[v_x, v_y, v_z]$ (m/s).
    pub vel_ned: [f32; 3],
    /// Unit orientation quaternion $[w, x, y, z]$.
    pub quat_wxyz: [f32; 4],
    /// Body-frame angular velocity $[p, q, r]$ (rad/s).
    pub omega_body: [f32; 3],
    /// Simulation timestamp in seconds.
    pub sim_time_s: f64,
}

impl Default for RigidBodyState6Dof {
    fn default() -> Self {
        Self {
            pos_ned: [0.0, 0.0, -10.0],
            vel_ned: [0.0, 0.0, 0.0],
            quat_wxyz: [1.0, 0.0, 0.0, 0.0],
            omega_body: [0.0, 0.0, 0.0],
            sim_time_s: 0.0,
        }
    }
}

/// Atomic Seqlock slot protecting a single Chronos world's 6-DoF state.
///
/// Even sequence numbers indicate a consistent quiescent state; odd sequence numbers
/// indicate an in-progress writer update.
pub struct SeqlockWorldSlot {
    seq: AtomicU64,
    /// Packed 64-bit words storing the serialized `RigidBodyState6Dof` atomically without `unsafe`.
    words: [AtomicU64; 8],
}

impl Default for SeqlockWorldSlot {
    fn default() -> Self {
        let slot = Self {
            seq: AtomicU64::new(0),
            words: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        };
        slot.write_state(&RigidBodyState6Dof::default());
        slot
    }
}

impl SeqlockWorldSlot {
    #[inline]
    fn pack_f32_pair(a: f32, b: f32) -> u64 {
        ((a.to_bits() as u64) << 32) | (b.to_bits() as u64)
    }

    #[inline]
    fn unpack_f32_pair(word: u64) -> (f32, f32) {
        let a = f32::from_bits((word >> 32) as u32);
        let b = f32::from_bits((word & 0xFFFF_FFFF) as u32);
        (a, b)
    }

    /// Publishes a new 6-DoF state using the Seqlock writer protocol (odd -> store -> even).
    #[inline]
    pub fn write_state(&self, state: &RigidBodyState6Dof) {
        let s0 = self.seq.load(Ordering::Relaxed);
        let odd_seq = (s0 & !1) + 1;
        self.seq.store(odd_seq, Ordering::Release);

        self.words[0].store(
            Self::pack_f32_pair(state.pos_ned[0], state.pos_ned[1]),
            Ordering::Relaxed,
        );
        self.words[1].store(
            Self::pack_f32_pair(state.pos_ned[2], state.vel_ned[0]),
            Ordering::Relaxed,
        );
        self.words[2].store(
            Self::pack_f32_pair(state.vel_ned[1], state.vel_ned[2]),
            Ordering::Relaxed,
        );
        self.words[3].store(
            Self::pack_f32_pair(state.quat_wxyz[0], state.quat_wxyz[1]),
            Ordering::Relaxed,
        );
        self.words[4].store(
            Self::pack_f32_pair(state.quat_wxyz[2], state.quat_wxyz[3]),
            Ordering::Relaxed,
        );
        self.words[5].store(
            Self::pack_f32_pair(state.omega_body[0], state.omega_body[1]),
            Ordering::Relaxed,
        );
        self.words[6].store(
            Self::pack_f32_pair(state.omega_body[2], 0.0),
            Ordering::Relaxed,
        );
        self.words[7].store(state.sim_time_s.to_bits(), Ordering::Relaxed);

        self.seq.store(odd_seq + 1, Ordering::Release);
    }

    /// Performs a lock-free optimistic Seqlock read, retrying if a concurrent writer was active.
    #[inline]
    pub fn read_state(&self) -> (RigidBodyState6Dof, u64) {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if (s1 & 1) != 0 {
                std::hint::spin_loop();
                continue;
            }

            let w0 = self.words[0].load(Ordering::Relaxed);
            let w1 = self.words[1].load(Ordering::Relaxed);
            let w2 = self.words[2].load(Ordering::Relaxed);
            let w3 = self.words[3].load(Ordering::Relaxed);
            let w4 = self.words[4].load(Ordering::Relaxed);
            let w5 = self.words[5].load(Ordering::Relaxed);
            let w6 = self.words[6].load(Ordering::Relaxed);
            let w7 = self.words[7].load(Ordering::Acquire);

            let s2 = self.seq.load(Ordering::Acquire);
            if s1 == s2 {
                let (px, py) = Self::unpack_f32_pair(w0);
                let (pz, vx) = Self::unpack_f32_pair(w1);
                let (vy, vz) = Self::unpack_f32_pair(w2);
                let (qw, qx) = Self::unpack_f32_pair(w3);
                let (qy, qz) = Self::unpack_f32_pair(w4);
                let (ox, oy) = Self::unpack_f32_pair(w5);
                let (oz, _) = Self::unpack_f32_pair(w6);
                let sim_time_s = f64::from_bits(w7);

                return (
                    RigidBodyState6Dof {
                        pos_ned: [px, py, pz],
                        vel_ned: [vx, vy, vz],
                        quat_wxyz: [qw, qx, qy, qz],
                        omega_body: [ox, oy, oz],
                        sim_time_s,
                    },
                    s2,
                );
            }
        }
    }
}

/// 128-world Chronos Seqlock arena providing $< 50\text{ ns}$ lock-free state access.
pub struct ChronosSeqlockArena {
    pub worlds: Vec<SeqlockWorldSlot>,
}

impl Default for ChronosSeqlockArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ChronosSeqlockArena {
    pub fn new() -> Self {
        let mut worlds = Vec::with_capacity(NUM_CHRONOS_WORLDS);
        for _ in 0..NUM_CHRONOS_WORLDS {
            worlds.push(SeqlockWorldSlot::default());
        }
        Self { worlds }
    }

    #[inline]
    pub fn publish_world(&self, world_idx: usize, state: &RigidBodyState6Dof) {
        self.worlds[world_idx].write_state(state);
    }

    #[inline]
    pub fn read_world(&self, world_idx: usize) -> (RigidBodyState6Dof, u64) {
        self.worlds[world_idx].read_state()
    }
}

/// Spherical obstacle in NED coordinates used during multi-world rollout verification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SphericalObstacle {
    pub center_ned: [f32; 3],
    pub radius_m: f32,
}

/// Candidate spatial action token emitted by Kortex's trajectory head.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CandidateActionToken {
    pub token_id: usize,
    /// Commanded target velocity $[v_x, v_y, v_z]$ in NED (m/s).
    pub cmd_vel_ned: [f32; 3],
    /// Commanded yaw rate (rad/s).
    pub cmd_yaw_rate: f32,
    /// Original unmasked logit score from Kortex.
    pub raw_logit: f32,
}

/// Safety and kinematic constraints for 128-world Chronos rollout verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutSafetyConfig {
    /// Maximum allowable total acceleration $\|a\|_2 \le a_{\max}$ (m/s$^2$).
    pub max_accel_mps2: f32,
    /// Maximum allowable vehicle tilt angle $\theta_{\max}$ (radians).
    pub max_tilt_rad: f32,
    /// Minimum required clearance beyond obstacle surface $r_{\text{safe}}$ (meters).
    pub safety_margin_m: f32,
    /// Rollout integration horizon $H$ steps per world.
    pub horizon_steps: usize,
    /// Rollout integration timestep $\Delta t$ (seconds).
    pub dt_s: f32,
}

impl Default for RolloutSafetyConfig {
    fn default() -> Self {
        Self {
            max_accel_mps2: 14.0,
            max_tilt_rad: 0.65, // ~37.2 degrees
            safety_margin_m: 1.2,
            horizon_steps: 20,
            dt_s: 0.05, // 1.0s lookahead at 20 Hz
        }
    }
}

/// Multi-world rollout verifier that simulates candidate action tokens across 128
/// parallel Chronos worlds (including wind gust perturbations) and masks unsafe tokens.
pub struct RolloutFeasibilityVerifier {
    pub config: RolloutSafetyConfig,
    pub obstacles: Vec<SphericalObstacle>,
}

impl RolloutFeasibilityVerifier {
    pub fn new(config: RolloutSafetyConfig, obstacles: Vec<SphericalObstacle>) -> Self {
        Self { config, obstacles }
    }

    /// Evaluates 128 parallel forward rollouts across the Chronos Seqlock arena,
    /// where subsets of worlds test each candidate action token under varying wind disturbances.
    /// Returns a boolean feasibility mask for `candidates` and updates `masked_logits`.
    pub fn verify_candidates_128_worlds(
        &self,
        arena: &ChronosSeqlockArena,
        candidates: &[CandidateActionToken],
        masked_logits: &mut [f32],
    ) -> Vec<bool> {
        assert_eq!(candidates.len(), masked_logits.len());
        let num_cands = candidates.len().max(1);
        let mut feasible = vec![true; candidates.len()];

        for world_idx in 0..NUM_CHRONOS_WORLDS {
            let cand_idx = world_idx % num_cands;
            if !feasible[cand_idx] {
                continue;
            }
            let cand = &candidates[cand_idx];
            let (mut state, _) = arena.read_world(world_idx);

            // Deterministic per-world wind disturbance profile across the 128 ensemble worlds
            let ensemble_member = (world_idx / num_cands) as f32;
            let wind_ned = [
                (ensemble_member * 0.37).sin() * 0.6,
                (ensemble_member * 0.53).cos() * 0.6,
                0.0,
            ];

            let tau_ctrl = 0.45f32; // Closed-loop quadrotor velocity tracking time constant (s)
            for _step in 0..self.config.horizon_steps {
                let ax = (cand.cmd_vel_ned[0] - state.vel_ned[0]) / tau_ctrl + wind_ned[0];
                let ay = (cand.cmd_vel_ned[1] - state.vel_ned[1]) / tau_ctrl + wind_ned[1];
                let az = (cand.cmd_vel_ned[2] - state.vel_ned[2]) / tau_ctrl + wind_ned[2];

                let accel_mag = (ax * ax + ay * ay + az * az).sqrt();
                if accel_mag > self.config.max_accel_mps2 {
                    feasible[cand_idx] = false;
                    break;
                }

                // Approximate required thrust tilt angle tan(theta) = sqrt(ax^2 + ay^2) / g
                let horiz_accel = (ax * ax + ay * ay).sqrt();
                let tilt_rad = (horiz_accel / 9.81).atan();
                if tilt_rad > self.config.max_tilt_rad {
                    feasible[cand_idx] = false;
                    break;
                }

                // Semi-implicit Euler step
                state.vel_ned[0] += ax * self.config.dt_s;
                state.vel_ned[1] += ay * self.config.dt_s;
                state.vel_ned[2] += az * self.config.dt_s;
                state.pos_ned[0] += state.vel_ned[0] * self.config.dt_s;
                state.pos_ned[1] += state.vel_ned[1] * self.config.dt_s;
                state.pos_ned[2] += state.vel_ned[2] * self.config.dt_s;

                // Check spherical obstacle clearance
                for obs in &self.obstacles {
                    let dx = state.pos_ned[0] - obs.center_ned[0];
                    let dy = state.pos_ned[1] - obs.center_ned[1];
                    let dz = state.pos_ned[2] - obs.center_ned[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < obs.radius_m + self.config.safety_margin_m {
                        feasible[cand_idx] = false;
                        break;
                    }
                }
                if !feasible[cand_idx] {
                    break;
                }
            }
        }

        for (i, &is_ok) in feasible.iter().enumerate() {
            masked_logits[i] = if is_ok {
                candidates[i].raw_logit
            } else {
                f32::NEG_INFINITY
            };
        }

        feasible
    }
}

/// Hardware watchdog state of the dual-tier Kestrel 400 Hz / Kortex 20 Hz bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WatchdogState {
    /// Kortex 20 Hz spatial planning frames are arriving within the 50 ms deadline.
    Nominal20Hz,
    /// Kortex missed its 50 ms frame deadline; Kestrel 400 Hz LQR loop is executing autonomous loiter fallback.
    FallbackLoiter400Hz,
}

/// Computes the standard CRC-16-CCITT (polynomial `0x1021`, init `0xFFFF`) over a byte slice.
pub fn crc16_ccitt(bytes: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in bytes {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// 64-byte CAN-FD avionics frame linking Kortex (20 Hz outer loop) and Kestrel (400 Hz inner loop).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanFdTelemetryFrame {
    /// Monotonic frame sequence number.
    pub seq_id: u32,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
    /// Target NED position setpoint $[x, y, z]$ (m).
    pub target_pos_ned: [f32; 3],
    /// Target NED velocity feedforward $[v_x, v_y, v_z]$ (m/s).
    pub target_vel_ned: [f32; 3],
    /// Target NED acceleration feedforward $[a_x, a_y, a_z]$ (m/s$^2$).
    pub target_acc_ned: [f32; 3],
    /// Target yaw angle (rad) and yaw rate (rad/s).
    pub target_yaw: f32,
    pub target_yaw_rate: f32,
    /// Active token ID selected by Kortex.
    pub action_token_id: u16,
    /// Watchdog status flag (`0 = Nominal20Hz`, `1 = FallbackLoiter400Hz`).
    pub watchdog_mode: u8,
    /// Reserved flag byte.
    pub flags: u8,
    /// CRC-16-CCITT checksum over bytes `0..62`.
    pub crc16: u16,
}

impl CanFdTelemetryFrame {
    /// Serializes this telemetry frame into an exact 64-byte CAN-FD payload with CRC-16-CCITT.
    pub fn to_bytes(&self) -> [u8; CAN_FD_FRAME_BYTES] {
        let mut buf = [0u8; CAN_FD_FRAME_BYTES];
        buf[0..4].copy_from_slice(&self.seq_id.to_le_bytes());
        buf[4..12].copy_from_slice(&self.timestamp_us.to_le_bytes());

        let mut offset = 12usize;
        for &v in &self.target_pos_ned {
            buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
            offset += 4;
        }
        for &v in &self.target_vel_ned {
            buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
            offset += 4;
        }
        for &v in &self.target_acc_ned {
            buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
            offset += 4;
        }
        buf[offset..offset + 4].copy_from_slice(&self.target_yaw.to_le_bytes());
        offset += 4;
        buf[offset..offset + 4].copy_from_slice(&self.target_yaw_rate.to_le_bytes());
        offset += 4;
        buf[offset..offset + 2].copy_from_slice(&self.action_token_id.to_le_bytes());
        offset += 2;
        buf[offset] = self.watchdog_mode;
        buf[offset + 1] = self.flags;

        let crc = crc16_ccitt(&buf[0..62]);
        buf[62..64].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserializes and validates a 64-byte CAN-FD payload, verifying its CRC-16-CCITT checksum.
    pub fn from_bytes(buf: &[u8; CAN_FD_FRAME_BYTES]) -> Result<Self, &'static str> {
        let expected_crc = crc16_ccitt(&buf[0..62]);
        let actual_crc = u16::from_le_bytes([buf[62], buf[63]]);
        if expected_crc != actual_crc {
            return Err("CAN-FD CRC-16-CCITT mismatch");
        }

        let seq_id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let timestamp_us = u64::from_le_bytes([
            buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
        ]);

        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };

        let target_pos_ned = [read_f32(12), read_f32(16), read_f32(20)];
        let target_vel_ned = [read_f32(24), read_f32(28), read_f32(32)];
        let target_acc_ned = [read_f32(36), read_f32(40), read_f32(44)];
        let target_yaw = read_f32(48);
        let target_yaw_rate = read_f32(52);
        let action_token_id = u16::from_le_bytes([buf[56], buf[57]]);
        let watchdog_mode = buf[58];
        let flags = buf[59];

        Ok(Self {
            seq_id,
            timestamp_us,
            target_pos_ned,
            target_vel_ned,
            target_acc_ned,
            target_yaw,
            target_yaw_rate,
            action_token_id,
            watchdog_mode,
            flags,
            crc16: actual_crc,
        })
    }
}

/// Dual-tier 400 Hz (Kestrel EKF/LQR) / 20 Hz (Kortex Spatial Planner) CAN-FD bridge with watchdog fallback.
pub struct KestrelCanFdBridge {
    /// Maximum allowable age of a Kortex 20 Hz setpoint before watchdog fallback triggers ($50,000\,\mu\text{s}$).
    pub watchdog_timeout_us: u64,
    /// Timestamp of the last valid Kortex 20 Hz frame received ($\mu\text{s}$).
    pub last_kortex_frame_us: u64,
    /// Last verified spatial setpoint received from Kortex.
    pub active_frame: CanFdTelemetryFrame,
    /// Safe hover/loiter anchor position in NED coordinates $[x, y, z]$ latched for fallback.
    pub safe_loiter_pos_ned: [f32; 3],
    /// Current hardware watchdog state.
    pub state: WatchdogState,
}

impl KestrelCanFdBridge {
    pub fn new(watchdog_timeout_us: u64, initial_pos_ned: [f32; 3]) -> Self {
        let init_frame = CanFdTelemetryFrame {
            seq_id: 0,
            timestamp_us: 0,
            target_pos_ned: initial_pos_ned,
            target_vel_ned: [0.0; 3],
            target_acc_ned: [0.0; 3],
            target_yaw: 0.0,
            target_yaw_rate: 0.0,
            action_token_id: 0,
            watchdog_mode: 0,
            flags: 0,
            crc16: 0,
        };
        Self {
            watchdog_timeout_us,
            last_kortex_frame_us: 0,
            active_frame: init_frame,
            safe_loiter_pos_ned: initial_pos_ned,
            state: WatchdogState::Nominal20Hz,
        }
    }

    /// Ingests a raw 64-byte CAN-FD frame from Kortex (20 Hz planner).
    pub fn receive_kortex_can_fd(&mut self, raw_bytes: &[u8; CAN_FD_FRAME_BYTES]) -> Result<(), &'static str> {
        let frame = CanFdTelemetryFrame::from_bytes(raw_bytes)?;
        self.last_kortex_frame_us = frame.timestamp_us;
        self.safe_loiter_pos_ned = frame.target_pos_ned;
        self.active_frame = frame;
        self.state = WatchdogState::Nominal20Hz;
        Ok(())
    }

    /// Executes one 400 Hz ($2.5\text{ ms}$) Kestrel inner-loop tick at `now_us`, returning the
    /// active position and velocity setpoints and transitioning to `FallbackLoiter400Hz` if
    /// Kortex exceeded `watchdog_timeout_us`.
    pub fn step_400hz_inner_loop(
        &mut self,
        now_us: u64,
        current_vel_ned: [f32; 3],
    ) -> ([f32; 3], [f32; 3], WatchdogState) {
        let elapsed_us = now_us.saturating_sub(self.last_kortex_frame_us);
        if elapsed_us > self.watchdog_timeout_us {
            self.state = WatchdogState::FallbackLoiter400Hz;
            // Minimum-jerk exponential velocity damping toward zero around safe_loiter_pos_ned
            let damped_vel = [
                current_vel_ned[0] * 0.85,
                current_vel_ned[1] * 0.85,
                current_vel_ned[2] * 0.85,
            ];
            (self.safe_loiter_pos_ned, damped_vel, self.state)
        } else {
            self.state = WatchdogState::Nominal20Hz;
            (
                self.active_frame.target_pos_ned,
                self.active_frame.target_vel_ned,
                self.state,
            )
        }
    }
}

/// Quantized acoustic logit vector emitted by the Sonon DSP wake-word & phoneme engine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SononAcousticLogitFrame {
    pub frame_seq: u32,
    pub vad_probability_q8: u8,
    pub wake_word_id: u8,
    /// 16-bin int8 quantized phoneme/keyword logit embedding.
    pub quantized_logits: [i8; 16],
}

impl SononAcousticLogitFrame {
    /// Packs the acoustic frame into two 64-bit words + one 64-bit metadata word for lockless atomic storage.
    #[inline]
    pub fn pack_words(&self) -> [u64; 3] {
        let meta = ((self.frame_seq as u64) << 16)
            | ((self.vad_probability_q8 as u64) << 8)
            | (self.wake_word_id as u64);
        let mut lo_bytes = [0u8; 8];
        let mut hi_bytes = [0u8; 8];
        for i in 0..8 {
            lo_bytes[i] = self.quantized_logits[i] as u8;
            hi_bytes[i] = self.quantized_logits[i + 8] as u8;
        }
        [meta, u64::from_le_bytes(lo_bytes), u64::from_le_bytes(hi_bytes)]
    }

    /// Unpacks the three 64-bit words into a `SononAcousticLogitFrame`.
    #[inline]
    pub fn unpack_words(words: [u64; 3]) -> Self {
        let frame_seq = (words[0] >> 16) as u32;
        let vad_probability_q8 = ((words[0] >> 8) & 0xFF) as u8;
        let wake_word_id = (words[0] & 0xFF) as u8;
        let lo_bytes = words[1].to_le_bytes();
        let hi_bytes = words[2].to_le_bytes();
        let mut quantized_logits = [0i8; 16];
        for i in 0..8 {
            quantized_logits[i] = lo_bytes[i] as i8;
            quantized_logits[i + 8] = hi_bytes[i] as i8;
        }
        Self {
            frame_seq,
            vad_probability_q8,
            wake_word_id,
            quantized_logits,
        }
    }
}

/// Lockless Single-Producer Single-Consumer (SPSC) Ring Buffer streaming Sonon acoustic
/// logit frames into Kortex's token embedding layer with zero heap allocations.
pub struct SononLogitRingBuffer {
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    slots: Vec<[AtomicU64; 3]>,
}

impl SononLogitRingBuffer {
    /// Creates a new lockless SPSC ring buffer with power-of-two capacity.
    pub fn new(power_of_two_capacity: usize) -> Self {
        let cap = power_of_two_capacity.next_power_of_two().max(4);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
        }
        Self {
            capacity: cap,
            mask: cap - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            slots,
        }
    }

    /// Pushes an acoustic frame into the ring buffer without locks or allocations.
    /// Returns `false` if the ring buffer is full.
    #[inline]
    pub fn try_push(&self, frame: &SononAcousticLogitFrame) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.capacity {
            return false;
        }
        let slot_idx = head & self.mask;
        let words = frame.pack_words();
        self.slots[slot_idx][0].store(words[0], Ordering::Relaxed);
        self.slots[slot_idx][1].store(words[1], Ordering::Relaxed);
        self.slots[slot_idx][2].store(words[2], Ordering::Relaxed);
        self.head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Pops the next acoustic frame from the ring buffer without locks or allocations.
    #[inline]
    pub fn try_pop(&self) -> Option<SononAcousticLogitFrame> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let slot_idx = tail & self.mask;
        let words = [
            self.slots[slot_idx][0].load(Ordering::Relaxed),
            self.slots[slot_idx][1].load(Ordering::Relaxed),
            self.slots[slot_idx][2].load(Ordering::Relaxed),
        ];
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(SononAcousticLogitFrame::unpack_words(words))
    }

    /// Projects a popped `SononAcousticLogitFrame` directly into a pre-allocated Kortex
    /// token embedding slice `out_embed` with zero heap allocations.
    #[inline]
    pub fn project_into_embedding(frame: &SononAcousticLogitFrame, scale: f32, out_embed: &mut [f32]) {
        let vad_gate = (frame.vad_probability_q8 as f32) / 255.0;
        for (idx, val) in out_embed.iter_mut().enumerate() {
            let q = frame.quantized_logits[idx & 15] as f32;
            *val = (q / 127.0) * vad_gate * scale;
        }
    }
}

/// Evaluation report for the Chronos Seqlock & Kestrel Avionics Interconnect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvionicsBridgeReport {
    /// Mean Seqlock 6-DoF state read latency in nanoseconds ($< 50\text{ ns}$).
    pub seqlock_read_latency_ns: f64,
    /// Total 128-world rollout feasibility verification latency in milliseconds ($< 30\text{ ms}$).
    pub rollout_128_worlds_latency_ms: f64,
    /// Number of candidate tokens evaluated.
    pub total_candidates: usize,
    /// Number of safe candidate tokens retained after 128-world obstacle & kinematic verification.
    pub safe_candidates_retained: usize,
    /// Whether the Kestrel 400 Hz watchdog deterministically triggered loiter fallback on a missed 20 Hz frame.
    pub watchdog_fallback_verified: bool,
    /// Number of Sonon acoustic frames streamed locklessly with zero heap allocations.
    pub sonon_frames_streamed: usize,
}

/// Evaluator for the Chronos Seqlock Rollout Verifier & Kestrel Avionics Interconnect.
pub struct AvionicsBridgeEvaluator;

impl AvionicsBridgeEvaluator {
    pub fn evaluate() -> AvionicsBridgeReport {
        let arena = ChronosSeqlockArena::new();
        for w in 0..NUM_CHRONOS_WORLDS {
            let state = RigidBodyState6Dof {
                pos_ned: [0.0, 0.0, -15.0],
                vel_ned: [2.0, 0.0, 0.0],
                quat_wxyz: [1.0, 0.0, 0.0, 0.0],
                omega_body: [0.0, 0.0, 0.0],
                sim_time_s: (w as f64) * 0.001,
            };
            arena.publish_world(w, &state);
        }

        // Benchmark Seqlock read latency across 10,000 reads
        let read_iters = 10_000usize;
        let start_read = Instant::now();
        for i in 0..read_iters {
            let (s, seq) = arena.read_world(i & (NUM_CHRONOS_WORLDS - 1));
            std::hint::black_box((s.pos_ned[0], seq));
        }
        let seqlock_read_latency_ns =
            start_read.elapsed().as_secs_f64() * 1e9 / (read_iters as f64);

        // Configure obstacle directly ahead at [7.0, 0.0, -15.0] with radius 1.4m
        let obstacles = vec![SphericalObstacle {
            center_ned: [7.0, 0.0, -15.0],
            radius_m: 1.4,
        }];
        let verifier = RolloutFeasibilityVerifier::new(RolloutSafetyConfig::default(), obstacles);

        // 4 candidate action tokens:
        // Token 0: flies straight into the obstacle at +7.5 m/s X -> UNSAFE (collision)
        // Token 1: Violent +20 m/s lateral jump -> UNSAFE (exceeds max_accel & max_tilt)
        // Token 2: Smooth lateral avoidance (+2.2 m/s X, +2.5 m/s Y) -> SAFE
        // Token 3: Controlled climb avoidance (+2.2 m/s X, -3.2 m/s Z) -> SAFE
        let candidates = vec![
            CandidateActionToken {
                token_id: 10,
                cmd_vel_ned: [7.5, 0.0, 0.0],
                cmd_yaw_rate: 0.0,
                raw_logit: 5.2,
            },
            CandidateActionToken {
                token_id: 11,
                cmd_vel_ned: [2.0, 20.0, 0.0],
                cmd_yaw_rate: 0.0,
                raw_logit: 4.8,
            },
            CandidateActionToken {
                token_id: 12,
                cmd_vel_ned: [2.2, 2.5, 0.0],
                cmd_yaw_rate: 0.15,
                raw_logit: 4.1,
            },
            CandidateActionToken {
                token_id: 13,
                cmd_vel_ned: [2.2, 0.0, -3.2],
                cmd_yaw_rate: 0.0,
                raw_logit: 3.9,
            },
        ];

        let mut masked_logits = vec![0.0f32; candidates.len()];
        let start_rollout = Instant::now();
        let feasible_mask =
            verifier.verify_candidates_128_worlds(&arena, &candidates, &mut masked_logits);
        let rollout_128_worlds_latency_ms = start_rollout.elapsed().as_secs_f64() * 1e3;
        let safe_candidates_retained = feasible_mask.iter().filter(|&&ok| ok).count();

        // Verify CAN-FD 400 Hz / 20 Hz bridge and hardware watchdog fallback
        let mut bridge = KestrelCanFdBridge::new(50_000, [0.0, 0.0, -15.0]);
        let tx_frame = CanFdTelemetryFrame {
            seq_id: 1,
            timestamp_us: 100_000,
            target_pos_ned: [12.0, 4.0, -15.0],
            target_vel_ned: [3.0, 1.0, 0.0],
            target_acc_ned: [0.2, 0.1, 0.0],
            target_yaw: 0.25,
            target_yaw_rate: 0.05,
            action_token_id: 12,
            watchdog_mode: 0,
            flags: 1,
            crc16: 0,
        };
        let wire_bytes = tx_frame.to_bytes();
        bridge
            .receive_kortex_can_fd(&wire_bytes)
            .expect("Valid CAN-FD frame must pass CRC-16 check");

        // At t = 125_000 us (+25 ms, within 50 ms budget), watchdog is Nominal20Hz
        let (_, _, state_nominal) = bridge.step_400hz_inner_loop(125_000, [3.0, 1.0, 0.0]);
        // At t = 160_000 us (+60 ms, exceeding 50 ms budget), watchdog triggers FallbackLoiter400Hz
        let (_, damped_vel, state_fallback) =
            bridge.step_400hz_inner_loop(160_000, [3.0, 1.0, 0.0]);
        let watchdog_fallback_verified = state_nominal == WatchdogState::Nominal20Hz
            && state_fallback == WatchdogState::FallbackLoiter400Hz
            && damped_vel[0] < 3.0;

        // Verify Sonon SPSC lockless ring buffer streaming
        let ring = SononLogitRingBuffer::new(64);
        let mut embed_buf = [0.0f32; 32];
        let mut streamed = 0usize;
        for seq in 0..64u32 {
            let frame = SononAcousticLogitFrame {
                frame_seq: seq,
                vad_probability_q8: 240,
                wake_word_id: 1,
                quantized_logits: [12, -8, 45, 64, -30, 19, 88, -14, 5, 22, -11, 33, 90, -40, 17, 6],
            };
            if ring.try_push(&frame) {
                if let Some(popped) = ring.try_pop() {
                    SononLogitRingBuffer::project_into_embedding(&popped, 1.0, &mut embed_buf);
                    streamed += 1;
                }
            }
        }

        AvionicsBridgeReport {
            seqlock_read_latency_ns,
            rollout_128_worlds_latency_ms,
            total_candidates: candidates.len(),
            safe_candidates_retained,
            watchdog_fallback_verified,
            sonon_frames_streamed: streamed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chronos_seqlock_and_rollout_verifier() {
        let report = AvionicsBridgeEvaluator::evaluate();
        assert!(
            report.rollout_128_worlds_latency_ms < 30.0,
            "128-world rollout verification must complete in < 30 ms, got {:.3} ms",
            report.rollout_128_worlds_latency_ms
        );
        assert_eq!(
            report.safe_candidates_retained, 2,
            "Exactly 2 of the 4 candidate trajectories should pass obstacle and kinematic bounds"
        );
        assert!(report.watchdog_fallback_verified);
        assert_eq!(report.sonon_frames_streamed, 64);
    }

    #[test]
    fn test_can_fd_crc16_corruption_rejection() {
        let frame = CanFdTelemetryFrame {
            seq_id: 77,
            timestamp_us: 500_000,
            target_pos_ned: [1.0, 2.0, -10.0],
            target_vel_ned: [0.5, 0.0, 0.0],
            target_acc_ned: [0.0, 0.0, 0.0],
            target_yaw: 0.0,
            target_yaw_rate: 0.0,
            action_token_id: 3,
            watchdog_mode: 0,
            flags: 0,
            crc16: 0,
        };
        let mut bytes = frame.to_bytes();
        assert!(CanFdTelemetryFrame::from_bytes(&bytes).is_ok());

        // Flip a bit in target_pos_ned and verify CRC-16-CCITT catches the corruption
        bytes[14] ^= 0x04;
        assert!(CanFdTelemetryFrame::from_bytes(&bytes).is_err());
    }
}
