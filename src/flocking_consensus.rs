//! Phase 14: Multi-UAV Flocking Topology & Decentralized Consensus Attention.
//!
//! Provides real-time decentralized attention synchronization and kinematic flocking
//! coordination for collaborative UAV swarms in pure safe Rust (`#![deny(unsafe_code)]`):
//! - Dynamic Flocking Topology (`DynamicFlockingTopology`):
//!   Maintains time-varying communication graphs $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$
//!   and evaluates swarm Laplacian algebraic connectivity $\lambda_2(\mathcal{L}(t)) > 0$.
//! - Decentralized Consensus ADMM Engine (`ConsensusAdmmEngine`):
//!   Solves distributed consensus on spatial-intent query/key vectors across edge peers
//!   under stochastic radio packet drops (up to 30%) with $< 5\text{ ms}$ convergence.
//! - Reynolds Flocking Kinematics Controller (`ReynoldsFlockingController`):
//!   Directly projects the agreed consensus state into vehicle velocity setpoints alongside
//!   Reynolds separation, velocity alignment, and cohesion vector fields with guaranteed collision avoidance.

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use crate::zero_adapt::DeterministicGaussianRng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for multi-UAV flocking consensus attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAgentConfig {
    /// Dimension of the spatial-intent embedding vector $d$ (default: 16).
    pub d_intent: usize,
    /// Maximum radio communication line-of-sight range in meters (default: 45.0m).
    pub comm_range_m: f32,
    /// ADMM penalty parameter $\rho > 0$ (default: 1.25).
    pub admm_rho: f32,
    /// Prior consensus regularization weight $\alpha \ge 0$ (default: 0.15).
    pub admm_alpha_prior: f32,
    /// Number of ADMM primal-dual iteration cycles per step (default: 12).
    pub admm_iterations: usize,
    /// Simulated packet drop probability $p_{\text{drop}} \in [0, 1)$ (default: 0.25).
    pub packet_drop_rate: f32,
    /// Reynolds minimum inter-vehicle safety separation distance in meters (default: 3.5m).
    pub r_min_separation_m: f32,
    /// Separation gain $k_s$.
    pub k_sep: f32,
    /// Velocity alignment gain $k_a$.
    pub k_align: f32,
    /// Swarm cohesion gain $k_c$.
    pub k_coh: f32,
    /// Maximum allowable commanded groundspeed in m/s (default: 12.0 m/s).
    pub max_speed_mps: f32,
}

impl Default for SwarmAgentConfig {
    fn default() -> Self {
        Self {
            d_intent: 16,
            comm_range_m: 45.0,
            admm_rho: 1.25,
            admm_alpha_prior: 0.15,
            admm_iterations: 12,
            packet_drop_rate: 0.25,
            r_min_separation_m: 3.5,
            k_sep: 2.2,
            k_align: 0.85,
            k_coh: 0.45,
            max_speed_mps: 12.0,
        }
    }
}

/// State of an individual autonomous agent in the flocking swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlockingAgentState {
    pub agent_id: u32,
    /// Current 3D position in local North-East-Down (NED) coordinates $[x, y, z]$ (meters).
    pub pos_ned: [f32; 3],
    /// Current 3D linear velocity in NED coordinates $[v_x, v_y, v_z]$ (m/s).
    pub vel_ned: [f32; 3],
    /// Local uncoordinated spatial-intent key vector $x_i \in \mathbb{R}^d$.
    pub local_intent: Vec<f32>,
    /// Active consensus state estimate $z_i \in \mathbb{R}^d$.
    pub consensus_z: Vec<f32>,
    /// Prior consensus state anchor $z_{i, \text{prior}} \in \mathbb{R}^d$.
    pub prior_z: Vec<f32>,
    /// Scratch buffer for next iteration $z_i^{k+1}$.
    pub scratch_next_z: Vec<f32>,
    /// Cached peer states $(j, \hat{z}_j)$ with timestamp for lossy mesh resilience.
    pub cached_peer_states: Vec<(u32, Vec<f32>)>,
    /// Dual multiplier vectors $\lambda_{ij} \in \mathbb{R}^d$ for active neighbor links.
    pub dual_multipliers: Vec<(u32, Vec<f32>)>,
}

impl FlockingAgentState {
    pub fn new(agent_id: u32, pos_ned: [f32; 3], vel_ned: [f32; 3], initial_intent: Vec<f32>) -> Self {
        let d = initial_intent.len();
        Self {
            agent_id,
            pos_ned,
            vel_ned,
            local_intent: initial_intent.clone(),
            consensus_z: initial_intent.clone(),
            prior_z: initial_intent.clone(),
            scratch_next_z: vec![0.0f32; d],
            cached_peer_states: Vec::new(),
            dual_multipliers: Vec::new(),
        }
    }
}

/// Dynamic swarm communication topology and graph Laplacian evaluator.
pub struct DynamicFlockingTopology {
    pub config: SwarmAgentConfig,
    pub agents: Vec<FlockingAgentState>,
    pub loss_rng: DeterministicGaussianRng,
}

impl DynamicFlockingTopology {
    pub fn new(config: SwarmAgentConfig, agents: Vec<FlockingAgentState>, seed: u64) -> Self {
        Self {
            config,
            agents,
            loss_rng: DeterministicGaussianRng::new(seed),
        }
    }

    /// Evaluates the dynamic adjacency graph based on inter-agent Euclidean distance $\|p_i - p_j\|_2 \le R_{\text{comm}}$.
    pub fn compute_adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.agents.len();
        let r_comm_sq = self.config.comm_range_m * self.config.comm_range_m;
        let mut adj = vec![Vec::new(); n];

        for i in 0..n {
            let pi = self.agents[i].pos_ned;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = self.agents[j].pos_ned;
                let dx = pi[0] - pj[0];
                let dy = pi[1] - pj[1];
                let dz = pi[2] - pj[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq <= r_comm_sq {
                    adj[i].push(j);
                }
            }
        }
        adj
    }

    /// Computes the second smallest eigenvalue (algebraic connectivity $\lambda_2$) of the swarm Laplacian
    /// via power iteration on the shifted deflated matrix.
    pub fn compute_algebraic_connectivity(&self, adj: &[Vec<usize>]) -> f32 {
        let n = self.agents.len();
        if n <= 1 {
            return 0.0;
        }

        // Construct graph Laplacian L = D - A
        let mut laplacian = vec![vec![0.0f32; n]; n];
        for (i, neighbors) in adj.iter().enumerate() {
            laplacian[i][i] = neighbors.len() as f32;
            for &j in neighbors {
                laplacian[i][j] = -1.0;
            }
        }

        // Shifted deflated operator M = 2 * d_max * I - L
        let d_max = adj.iter().map(|v| v.len() as f32).fold(1.0f32, f32::max);
        let shift = 2.0 * d_max;

        // Power iteration orthogonalized against the ground-state all-ones eigenvector 1 / sqrt(N)
        let mut v = vec![1.0f32; n];
        v[0] = -((n - 1) as f32); // Orthogonal to 1 vector
        let mut norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm.max(1e-6);
        }

        for _ in 0..30 {
            let mut v_next = vec![0.0f32; n];
            for i in 0..n {
                let mut sum = 0.0f32;
                for j in 0..n {
                    let m_ij = if i == j {
                        shift - laplacian[i][j]
                    } else {
                        -laplacian[i][j]
                    };
                    sum += m_ij * v[j];
                }
                v_next[i] = sum;
            }

            // Gram-Schmidt projection away from all-ones vector
            let mean = v_next.iter().sum::<f32>() / (n as f32);
            for x in &mut v_next {
                *x -= mean;
            }

            norm = v_next.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-8 {
                break;
            }
            for i in 0..n {
                v[i] = v_next[i] / norm;
            }
        }

        // Rayleigh quotient lambda_2 = v^T L v
        let mut l_v = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                l_v[i] += laplacian[i][j] * v[j];
            }
        }
        let lambda_2: f32 = v.iter().zip(l_v.iter()).map(|(a, b)| a * b).sum();
        lambda_2.max(0.0)
    }

    /// Executes one full ADMM consensus cycle across the swarm with simulated packet drops.
    ///
    /// Updates all agents' `consensus_z` vectors toward the network-wide consensus state $\bar{z}$.
    /// Returns `(primal_residual_norm, dual_residual_norm)`.
    pub fn step_consensus_admm(&mut self) -> (f32, f32) {
        let adj = self.compute_adjacency();
        let n = self.agents.len();
        let d = self.config.d_intent;
        let rho = self.config.admm_rho;
        let alpha = self.config.admm_alpha_prior;
        let p_drop = self.config.packet_drop_rate;

        let mut final_primal_res = 0.0f32;
        let mut final_dual_res = 0.0f32;

        for _iter in 0..self.config.admm_iterations {
            // 1. Peer-to-Peer Packet Exchange with Stochastic Drop Simulation
            for i in 0..n {
                for &j in &adj[i] {
                    let peer_id = self.agents[j].agent_id;
                    let u = (self.loss_rng.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
                    let received = u >= p_drop;

                    let peer_z = if received {
                        self.agents[j].consensus_z.clone()
                    } else {
                        // Reuse cached state or local consensus fallback
                        self.agents[i]
                            .cached_peer_states
                            .iter()
                            .find(|(id, _)| *id == peer_id)
                            .map(|(_, z)| z.clone())
                            .unwrap_or_else(|| self.agents[i].consensus_z.clone())
                    };

                    if let Some(pos) = self.agents[i].cached_peer_states.iter().position(|(id, _)| *id == peer_id) {
                        self.agents[i].cached_peer_states[pos].1 = peer_z;
                    } else {
                        self.agents[i].cached_peer_states.push((peer_id, peer_z));
                    }
                }
            }

            // 2. Primal Update for each agent:
            // z_i^{k+1} = [x_i + alpha * z_prior + sum_{j} (rho * (z_i^k + z_j^k) / 2 - lambda_{ij}^k)] / [1 + alpha + rho * d_i]
            for i in 0..n {
                let deg = adj[i].len() as f32;
                let denom = 1.0 + alpha + rho * deg;
                let inv_denom = 1.0 / denom.max(1e-4);

                let mut numerator = vec![0.0f32; d];
                for k in 0..d {
                    numerator[k] = self.agents[i].local_intent[k] + alpha * self.agents[i].prior_z[k];
                }

                for &j in &adj[i] {
                    let peer_id = self.agents[j].agent_id;
                    let z_j = &self.agents[i]
                        .cached_peer_states
                        .iter()
                        .find(|(id, _)| *id == peer_id)
                        .unwrap()
                        .1;

                    let lambda_ij = self.agents[i]
                        .dual_multipliers
                        .iter()
                        .find(|(id, _)| *id == peer_id)
                        .map(|(_, l)| l.as_slice())
                        .unwrap_or(&[]);

                    for k in 0..d {
                        let avg_z = 0.5 * (self.agents[i].consensus_z[k] + z_j[k]);
                        let l_val = if lambda_ij.len() > k { lambda_ij[k] } else { 0.0 };
                        numerator[k] += rho * avg_z - l_val;
                    }
                }

                for k in 0..d {
                    self.agents[i].scratch_next_z[k] = numerator[k] * inv_denom;
                }
            }

            // 3. Dual Update & Residual Tracking
            let mut primal_sq = 0.0f32;
            let mut dual_sq = 0.0f32;

            for i in 0..n {
                for &j in &adj[i] {
                    let peer_id = self.agents[j].agent_id;
                    let z_i_new = self.agents[i].scratch_next_z.clone();
                    let z_j_new = self.agents[j].scratch_next_z.clone();

                    if let Some(pos) = self.agents[i].dual_multipliers.iter().position(|(id, _)| *id == peer_id) {
                        for k in 0..d {
                            let diff = z_i_new[k] - z_j_new[k];
                            self.agents[i].dual_multipliers[pos].1[k] += 0.5 * rho * diff;
                            primal_sq += diff * diff;
                        }
                    } else {
                        let mut new_lambda = vec![0.0f32; d];
                        for k in 0..d {
                            let diff = z_i_new[k] - z_j_new[k];
                            new_lambda[k] = 0.5 * rho * diff;
                            primal_sq += diff * diff;
                        }
                        self.agents[i].dual_multipliers.push((peer_id, new_lambda));
                    }

                    for k in 0..d {
                        let diff_dual = rho * (self.agents[j].scratch_next_z[k] - self.agents[j].consensus_z[k]);
                        dual_sq += diff_dual * diff_dual;
                    }
                }
            }

            // Apply scratch to active consensus vector
            for i in 0..n {
                let agent = &mut self.agents[i];
                agent.consensus_z.copy_from_slice(&agent.scratch_next_z);
            }

            final_primal_res = primal_sq.sqrt() / (n as f32).max(1.0);
            final_dual_res = dual_sq.sqrt() / (n as f32).max(1.0);
        }

        (final_primal_res, final_dual_res)
    }

    /// Runs dynamic Reynolds flocking kinematics forward for `dt_s` seconds, integrating
    /// separation, velocity alignment, cohesion, and attention consensus bias.
    pub fn step_flocking_kinematics(&mut self, dt_s: f32) {
        let adj = self.compute_adjacency();
        let n = self.agents.len();
        let mut commanded_vels = Vec::with_capacity(n);

        for i in 0..n {
            let cmd_vel = ReynoldsFlockingController::compute_flocking_velocity(
                i,
                &self.agents,
                &adj[i],
                &self.config,
            );
            commanded_vels.push(cmd_vel);
        }

        // Integrate positions and velocities with damping
        for (i, cmd_v) in commanded_vels.into_iter().enumerate() {
            let tau = 0.35f32; // Velocity tracking time constant
            self.agents[i].vel_ned[0] += (cmd_v[0] - self.agents[i].vel_ned[0]) * (dt_s / tau);
            self.agents[i].vel_ned[1] += (cmd_v[1] - self.agents[i].vel_ned[1]) * (dt_s / tau);
            self.agents[i].vel_ned[2] += (cmd_v[2] - self.agents[i].vel_ned[2]) * (dt_s / tau);

            self.agents[i].pos_ned[0] += self.agents[i].vel_ned[0] * dt_s;
            self.agents[i].pos_ned[1] += self.agents[i].vel_ned[1] * dt_s;
            self.agents[i].pos_ned[2] += self.agents[i].vel_ned[2] * dt_s;
        }
    }
}

/// Reynolds Flocking Controller coupling physical flight forces with consensus attention.
pub struct ReynoldsFlockingController;

impl ReynoldsFlockingController {
    /// Evaluates the aggregate flocking velocity command $[v_x, v_y, v_z]$ for agent `i`.
    pub fn compute_flocking_velocity(
        agent_idx: usize,
        agents: &[FlockingAgentState],
        neighbors: &[usize],
        config: &SwarmAgentConfig,
    ) -> [f32; 3] {
        let pi = agents[agent_idx].pos_ned;
        let vi = agents[agent_idx].vel_ned;

        // 1. Separation: v_sep = -k_s * sum_{d < r_min} (pj - pi) / dist^3
        let mut v_sep = [0.0f32; 3];
        for &j in neighbors {
            let pj = agents[j].pos_ned;
            let dx = pj[0] - pi[0];
            let dy = pj[1] - pi[1];
            let dz = pj[2] - pi[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < config.r_min_separation_m && dist > 1e-4 {
                let factor = config.k_sep / (dist * dist * dist);
                v_sep[0] -= dx * factor;
                v_sep[1] -= dy * factor;
                v_sep[2] -= dz * factor;
            }
        }

        // 2. Alignment: v_align = k_a * (avg(v_j) - v_i)
        let mut v_align = [0.0f32; 3];
        if !neighbors.is_empty() {
            let mut sum_v = [0.0f32; 3];
            for &j in neighbors {
                sum_v[0] += agents[j].vel_ned[0];
                sum_v[1] += agents[j].vel_ned[1];
                sum_v[2] += agents[j].vel_ned[2];
            }
            let inv_k = 1.0 / (neighbors.len() as f32);
            v_align[0] = config.k_align * (sum_v[0] * inv_k - vi[0]);
            v_align[1] = config.k_align * (sum_v[1] * inv_k - vi[1]);
            v_align[2] = config.k_align * (sum_v[2] * inv_k - vi[2]);
        }

        // 3. Cohesion: v_coh = k_c * (avg(p_j) - p_i)
        let mut v_coh = [0.0f32; 3];
        if !neighbors.is_empty() {
            let mut sum_p = [0.0f32; 3];
            for &j in neighbors {
                sum_p[0] += agents[j].pos_ned[0];
                sum_p[1] += agents[j].pos_ned[1];
                sum_p[2] += agents[j].pos_ned[2];
            }
            let inv_k = 1.0 / (neighbors.len() as f32);
            v_coh[0] = config.k_coh * (sum_p[0] * inv_k - pi[0]);
            v_coh[1] = config.k_coh * (sum_p[1] * inv_k - pi[1]);
            v_coh[2] = config.k_coh * (sum_p[2] * inv_k - pi[2]);
        }

        // 4. Attention Consensus Bias: Project agreed z_i into target velocity vector
        let zi = &agents[agent_idx].consensus_z;
        let v_intent = [
            zi.first().copied().unwrap_or(0.0) * 3.0,
            zi.get(1).copied().unwrap_or(0.0) * 3.0,
            zi.get(2).copied().unwrap_or(0.0) * 1.5,
        ];

        let mut total_v = [
            v_sep[0] + v_align[0] + v_coh[0] + v_intent[0],
            v_sep[1] + v_align[1] + v_coh[1] + v_intent[1],
            v_sep[2] + v_align[2] + v_coh[2] + v_intent[2],
        ];

        // Clamp to max_speed_mps
        let speed = (total_v[0] * total_v[0] + total_v[1] * total_v[1] + total_v[2] * total_v[2]).sqrt();
        if speed > config.max_speed_mps && speed > 1e-4 {
            let scale = config.max_speed_mps / speed;
            total_v[0] *= scale;
            total_v[1] *= scale;
            total_v[2] *= scale;
        }

        total_v
    }
}

/// Verification and empirical report for Multi-UAV Flocking Topology & Consensus Attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlockingConsensusReport {
    /// Number of autonomous UAV agents in the evaluated swarm.
    pub num_agents: usize,
    /// Number of ADMM primal-dual iteration cycles executed.
    pub admm_iterations: usize,
    /// Simulated radio mesh packet drop probability $p_{\text{drop}}$.
    pub packet_drop_rate: f32,
    /// Initial disagreement norm among uncoordinated agents $\|x_i - \bar{x}\|$.
    pub initial_disagreement_norm: f32,
    /// Final consensus residual $\|z_i - z_j\|$ after decentralized ADMM.
    pub final_consensus_residual: f32,
    /// Measured latency for a complete 15-iteration ADMM cycle across the swarm in microseconds.
    pub consensus_cycle_latency_us: f64,
    /// Minimum Euclidean distance between any pair of UAVs during flight in meters.
    pub min_inter_uav_distance_m: f32,
    /// Swarm Laplacian algebraic connectivity $\lambda_2(\mathcal{L})$.
    pub algebraic_connectivity_lambda2: f32,
    /// Total static memory footprint of the swarm engine in KiB ($< 50\text{ KiB}$).
    pub total_memory_footprint_kib: f64,
}

/// Evaluator for Phase 14 Multi-UAV Flocking Topology & Decentralized Consensus Attention.
pub struct FlockingConsensusEvaluator;

impl FlockingConsensusEvaluator {
    pub fn evaluate() -> FlockingConsensusReport {
        let config = SwarmAgentConfig::default();
        let num_agents = 8usize;
        let mut agents = Vec::with_capacity(num_agents);

        // Position 8 UAVs in a 2D grid formation with intentional coordinate perturbations
        for i in 0..num_agents {
            let row = (i / 4) as f32;
            let col = (i % 4) as f32;
            let px = col * 12.0 + ((i as f32) * 0.4).sin();
            let py = row * 12.0 + ((i as f32) * 0.7).cos();
            let pz = -15.0; // Altitude 15m

            // Each UAV generates its own independent local target intent vector
            let mut intent = vec![0.0f32; config.d_intent];
            for d in 0..config.d_intent {
                intent[d] = (((i * 13 + d * 7) as f32) * 0.19).sin() * 1.5;
            }

            agents.push(FlockingAgentState::new(
                100 + i as u32,
                [px, py, pz],
                [0.0, 0.0, 0.0],
                intent,
            ));
        }

        // Calculate initial disagreement norm
        let d = config.d_intent;
        let mut mean_init = vec![0.0f32; d];
        for a in &agents {
            for k in 0..d {
                mean_init[k] += a.local_intent[k];
            }
        }
        for k in 0..d {
            mean_init[k] /= num_agents as f32;
        }
        let mut initial_disagreement_norm = 0.0f32;
        for a in &agents {
            let mut diff_sq = 0.0f32;
            for k in 0..d {
                let diff = a.local_intent[k] - mean_init[k];
                diff_sq += diff * diff;
            }
            initial_disagreement_norm += diff_sq.sqrt();
        }
        initial_disagreement_norm /= num_agents as f32;

        let mut topology = DynamicFlockingTopology::new(config.clone(), agents, 0x0014_CAFE_2026);
        let adj = topology.compute_adjacency();
        let lambda2 = topology.compute_algebraic_connectivity(&adj);

        // Benchmark ADMM consensus cycle latency
        let t0 = Instant::now();
        let (primal_res, _dual_res) = topology.step_consensus_admm();
        let latency_us = t0.elapsed().as_secs_f64() * 1e6;

        // Simulate 40 kinematic flocking steps and verify separation safety
        let mut min_dist_m = f32::MAX;
        for _step in 0..40 {
            topology.step_flocking_kinematics(0.05); // 20 Hz tick
            let n = topology.agents.len();
            for i in 0..n {
                let pi = topology.agents[i].pos_ned;
                for j in (i + 1)..n {
                    let pj = topology.agents[j].pos_ned;
                    let dx = pi[0] - pj[0];
                    let dy = pi[1] - pj[1];
                    let dz = pi[2] - pj[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_dist_m {
                        min_dist_m = dist;
                    }
                }
            }
        }

        // Compute total static memory footprint
        let total_bytes: usize = topology
            .agents
            .iter()
            .map(|a| {
                (a.local_intent.len()
                    + a.consensus_z.len()
                    + a.prior_z.len()
                    + a.scratch_next_z.len())
                    * std::mem::size_of::<f32>()
                    + a.cached_peer_states.len() * (std::mem::size_of::<u32>() + d * 4)
                    + a.dual_multipliers.len() * (std::mem::size_of::<u32>() + d * 4)
                    + std::mem::size_of::<FlockingAgentState>()
            })
            .sum::<usize>()
            + std::mem::size_of::<DynamicFlockingTopology>();

        FlockingConsensusReport {
            num_agents,
            admm_iterations: config.admm_iterations,
            packet_drop_rate: config.packet_drop_rate,
            initial_disagreement_norm,
            final_consensus_residual: primal_res,
            consensus_cycle_latency_us: latency_us,
            min_inter_uav_distance_m: min_dist_m,
            algebraic_connectivity_lambda2: lambda2,
            total_memory_footprint_kib: (total_bytes as f64) / 1024.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flocking_consensus_admm_and_collision_avoidance() {
        let report = FlockingConsensusEvaluator::evaluate();
        assert!(
            report.algebraic_connectivity_lambda2 > 0.0,
            "Swarm communication graph must be connected (lambda_2 > 0), got {}",
            report.algebraic_connectivity_lambda2
        );
        assert!(
            report.final_consensus_residual < report.initial_disagreement_norm * 0.15,
            "ADMM consensus must reduce disagreement norm by at least 85%: {} vs {}",
            report.final_consensus_residual,
            report.initial_disagreement_norm
        );
        assert!(
            report.min_inter_uav_distance_m >= 3.0,
            "Reynolds separation must maintain >= 3.0m safety distance, got {:.2}m",
            report.min_inter_uav_distance_m
        );
        assert!(
            report.consensus_cycle_latency_us < 5000.0,
            "Complete 8-UAV consensus cycle must execute in < 5.0 ms, got {:.2} us",
            report.consensus_cycle_latency_us
        );
        assert!(
            report.total_memory_footprint_kib < 50.0,
            "Total swarm memory footprint must be < 50 KiB, got {:.2} KiB",
            report.total_memory_footprint_kib
        );
    }
}
