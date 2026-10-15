//! Symplectic Hamiltonian Neural Dynamics for Long-Horizon Key Evolution Engine
//!
//! Subsystem Capabilities:
//! - Cotangent bundle phase-space representation M = T* Q ~= R^(2d) mapping keys k to
//!   generalized coordinates q and values v to canonical conjugate momenta p.
//! - Separable Hamiltonian formulation H(q, p) = (1/2) p^T M^(-1) p + V(q) governing
//!   energy-conserving attention state trajectories.
//! - Symplectic Störmer-Verlet leapfrog (2nd order) and Yoshida composition (4th order)
//!   integrators strictly preserving canonical 2-form omega = dq ^ dp and Liouville volume.
//! - Exact time-reversibility S(-dt)^N o S(dt)^N(z_0) == z_0 eliminating numerical dissipation.
//! - Shadow Hamiltonian tracking proving bounded energy evolution across long horizons.
//!
//! Strict Mandates:
//! - `#![deny(unsafe_code)]`
//! - Zero unicode emojis across all code, tests, and outputs
//! - Bounded static allocations and deterministic execution (< 5.0 us/step)

#![deny(unsafe_code)]

/// Default phase-space coordinate dimension (half of 32-dim head).
pub const DEFAULT_PHASE_DIM: usize = 16;

/// Phase-space state vector z = [q^T, p^T]^T in R^(2d).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSpaceState {
    /// Generalized coordinates (keys) in R^d.
    pub q: Vec<f32>,
    /// Canonical conjugate momenta (values) in R^d.
    pub p: Vec<f32>,
}

impl PhaseSpaceState {
    /// Creates a new phase-space state from key and value slices.
    pub fn new(q: &[f32], p: &[f32]) -> Self {
        assert_eq!(q.len(), p.len(), "q and p dimensions must match");
        Self {
            q: q.to_vec(),
            p: p.to_vec(),
        }
    }

    /// Dimensionality of the configuration manifold d.
    pub fn dim(&self) -> usize {
        self.q.len()
    }

    /// L2 norm of coordinates (keys).
    pub fn norm_q(&self) -> f32 {
        self.q.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// L2 norm of momenta (values).
    pub fn norm_p(&self) -> f32 {
        self.p.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Smooth separable Hamiltonian energy function H(q, p) = T(p) + V(q).
#[derive(Debug, Clone)]
pub struct SeparableHamiltonian {
    /// Dimensionality d of coordinates and momenta.
    pub dim: usize,
    /// Diagonal inverse mass matrix elements m_i^(-1).
    pub inv_mass: Vec<f32>,
    /// Diagonal harmonic stiffness parameters k_i for quadratic base potential.
    pub potential_stiffness: Vec<f32>,
    /// Hidden projection weights for non-linear potential layer (hidden_dim x dim).
    pub hidden_weights: Vec<f32>,
    /// Hidden layer biases in R^hidden_dim.
    pub hidden_biases: Vec<f32>,
    /// Scaling coefficients lambda_j for non-linear potential components.
    pub hidden_scales: Vec<f32>,
    /// Hidden dimension for non-linear potential.
    pub hidden_dim: usize,
}

impl SeparableHamiltonian {
    /// Constructs a deterministic separable Hamiltonian system.
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        let mut inv_mass = Vec::with_capacity(dim);
        let mut potential_stiffness = Vec::with_capacity(dim);
        for i in 0..dim {
            // Positive definite mass: m_i in [0.8, 1.2]
            let m_inv = 1.0 + 0.15 * ((i as f32 * 0.4).sin());
            inv_mass.push(m_inv.max(0.2));
            // Stiffness: k_i in [0.4, 0.8]
            let k = 0.6 + 0.2 * ((i as f32 * 0.7).cos());
            potential_stiffness.push(k.max(0.1));
        }

        let total_weights = hidden_dim * dim;
        let mut hidden_weights = Vec::with_capacity(total_weights);
        for idx in 0..total_weights {
            let val = 0.05 * ((idx as f32 * 0.13).sin());
            hidden_weights.push(val);
        }

        let mut hidden_biases = Vec::with_capacity(hidden_dim);
        let mut hidden_scales = Vec::with_capacity(hidden_dim);
        for j in 0..hidden_dim {
            hidden_biases.push(0.01 * ((j as f32 * 0.31).cos()));
            hidden_scales.push(0.10 * (1.0 + 0.2 * (j as f32 * 0.19).sin()));
        }

        Self {
            dim,
            inv_mass,
            potential_stiffness,
            hidden_weights,
            hidden_biases,
            hidden_scales,
            hidden_dim,
        }
    }

    /// Evaluates the kinetic energy T(p) = (1/2) p^T M^(-1) p.
    #[inline]
    pub fn kinetic_energy(&self, p: &[f32]) -> f32 {
        let mut t = 0.0f32;
        for (i, &p_val) in p.iter().take(self.dim).enumerate() {
            t += 0.5 * self.inv_mass[i] * p_val * p_val;
        }
        t
    }

    /// Evaluates smooth sigmoid function sigma(x) = 1 / (1 + exp(-x)).
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Evaluates smooth GELU approximation and its first derivative:
    /// GELU(x) ~= x * sigma(1.702 * x)
    /// GELU'(x) ~= sigma(1.702 * x) + 1.702 * x * sigma(1.702 * x) * (1 - sigma(1.702 * x))
    #[inline]
    fn gelu_and_deriv(x: f32) -> (f32, f32) {
        let s = Self::sigmoid(1.702 * x);
        let val = x * s;
        let dval = s + 1.702 * x * s * (1.0 - s);
        (val, dval)
    }

    /// Evaluates potential energy V(q) = (1/2) q^T K q + sum_j lambda_j GELU(w_j^T q + b_j).
    pub fn potential_energy(&self, q: &[f32]) -> f32 {
        let mut v = 0.0f32;
        // 1. Quadratic harmonic term
        for (i, &q_val) in q.iter().take(self.dim).enumerate() {
            v += 0.5 * self.potential_stiffness[i] * q_val * q_val;
        }

        // 2. Non-linear contextual potential term
        for j in 0..self.hidden_dim {
            let row_start = j * self.dim;
            let mut dot = self.hidden_biases[j];
            for (i, &q_val) in q.iter().take(self.dim).enumerate() {
                dot += self.hidden_weights[row_start + i] * q_val;
            }
            let (gelu_val, _) = Self::gelu_and_deriv(dot);
            v += self.hidden_scales[j] * gelu_val;
        }
        v
    }

    /// Computes analytical potential gradient nabla_q V(q).
    pub fn potential_gradient(&self, q: &[f32], grad: &mut [f32]) {
        assert_eq!(q.len(), self.dim);
        assert_eq!(grad.len(), self.dim);

        // 1. Quadratic harmonic gradient: K q
        for (i, &q_val) in q.iter().take(self.dim).enumerate() {
            grad[i] = self.potential_stiffness[i] * q_val;
        }

        // 2. Non-linear gradient: sum_j lambda_j w_j GELU'(w_j^T q + b_j)
        for j in 0..self.hidden_dim {
            let row_start = j * self.dim;
            let mut dot = self.hidden_biases[j];
            for (i, &q_val) in q.iter().take(self.dim).enumerate() {
                dot += self.hidden_weights[row_start + i] * q_val;
            }
            let (_, gelu_deriv) = Self::gelu_and_deriv(dot);
            let factor = self.hidden_scales[j] * gelu_deriv;

            for (i, g_val) in grad.iter_mut().take(self.dim).enumerate() {
                *g_val += factor * self.hidden_weights[row_start + i];
            }
        }
    }

    /// Computes total Hamiltonian energy H(q, p) = T(p) + V(q).
    #[inline]
    pub fn total_energy(&self, state: &PhaseSpaceState) -> f32 {
        self.kinetic_energy(&state.p) + self.potential_energy(&state.q)
    }

    /// Computes the leading-order shadow Hamiltonian H_shadow = H + (dt^2 / 24) * {V, {V, T}}.
    pub fn shadow_energy(&self, state: &PhaseSpaceState, dt: f32) -> f32 {
        let h0 = self.total_energy(state);
        let mut grad_v = vec![0.0f32; self.dim];
        self.potential_gradient(&state.q, &mut grad_v);

        // Term: -||nabla V||_{M^-1}^2
        let mut norm_grad_sq = 0.0f32;
        for (i, &g_val) in grad_v.iter().take(self.dim).enumerate() {
            norm_grad_sq += self.inv_mass[i] * g_val * g_val;
        }

        // Leading-order perturbation: (dt^2 / 24) * norm_grad_sq
        h0 - (dt * dt / 24.0) * norm_grad_sq
    }
}

/// Second-order symplectic Störmer-Verlet leapfrog integrator.
#[derive(Debug, Default, Clone, Copy)]
pub struct StormerVerletIntegrator;

impl StormerVerletIntegrator {
    /// In-place forward symplectic leapfrog step:
    /// p_(n+1/2) = p_n - (dt/2) * nabla V(q_n)
    /// q_(n+1)   = q_n + dt * M^(-1) * p_(n+1/2)
    /// p_(n+1)   = p_(n+1/2) - (dt/2) * nabla V(q_(n+1))
    pub fn step_inplace(
        &self,
        ham: &SeparableHamiltonian,
        q: &mut [f32],
        p: &mut [f32],
        dt: f32,
        grad_buf: &mut [f32],
    ) {
        let half_dt = 0.5 * dt;

        // 1. Half-step momentum kick: p <- p - (dt/2) * nabla V(q)
        ham.potential_gradient(q, grad_buf);
        for i in 0..ham.dim {
            p[i] -= half_dt * grad_buf[i];
        }

        // 2. Full-step coordinate drift: q <- q + dt * M^(-1) * p
        for i in 0..ham.dim {
            q[i] += dt * ham.inv_mass[i] * p[i];
        }

        // 3. Half-step momentum kick: p <- p - (dt/2) * nabla V(q_new)
        ham.potential_gradient(q, grad_buf);
        for i in 0..ham.dim {
            p[i] -= half_dt * grad_buf[i];
        }
    }

    /// Functional forward step returning a new PhaseSpaceState.
    pub fn step(
        &self,
        ham: &SeparableHamiltonian,
        state: &PhaseSpaceState,
        dt: f32,
    ) -> PhaseSpaceState {
        let mut q = state.q.clone();
        let mut p = state.p.clone();
        let mut grad_buf = vec![0.0f32; ham.dim];
        self.step_inplace(ham, &mut q, &mut p, dt, &mut grad_buf);
        PhaseSpaceState { q, p }
    }

    /// Functional backward step integrating with -dt (exact time reversibility).
    pub fn step_backward(
        &self,
        ham: &SeparableHamiltonian,
        state: &PhaseSpaceState,
        dt: f32,
    ) -> PhaseSpaceState {
        self.step(ham, state, -dt)
    }
}

/// Fourth-order symplectic Yoshida composition integrator.
#[derive(Debug, Default, Clone, Copy)]
pub struct Yoshida4thIntegrator;

impl Yoshida4thIntegrator {
    /// In-place 4th-order symplectic step using Yoshida triple composition.
    pub fn step_inplace(
        &self,
        ham: &SeparableHamiltonian,
        q: &mut [f32],
        p: &mut [f32],
        dt: f32,
        grad_buf: &mut [f32],
    ) {
        let sv = StormerVerletIntegrator;
        let cbrt2 = 2.0f32.powf(1.0 / 3.0);
        let w1 = 1.0 / (2.0 - cbrt2);
        let w0 = -cbrt2 / (2.0 - cbrt2);

        let dt1 = w1 * dt;
        let dt0 = w0 * dt;

        sv.step_inplace(ham, q, p, dt1, grad_buf);
        sv.step_inplace(ham, q, p, dt0, grad_buf);
        sv.step_inplace(ham, q, p, dt1, grad_buf);
    }

    /// Functional 4th-order step returning a new PhaseSpaceState.
    pub fn step(
        &self,
        ham: &SeparableHamiltonian,
        state: &PhaseSpaceState,
        dt: f32,
    ) -> PhaseSpaceState {
        let mut q = state.q.clone();
        let mut p = state.p.clone();
        let mut grad_buf = vec![0.0f32; ham.dim];
        self.step_inplace(ham, &mut q, &mut p, dt, &mut grad_buf);
        PhaseSpaceState { q, p }
    }
}

/// Shadow Hamiltonian diagnostic tracker for bounding energy invariants.
#[derive(Debug, Clone)]
pub struct ShadowHamiltonianTracker {
    pub initial_shadow_energy: f32,
    pub tolerance: f32,
}

impl ShadowHamiltonianTracker {
    pub fn new(ham: &SeparableHamiltonian, state: &PhaseSpaceState, dt: f32, tolerance: f32) -> Self {
        let e0 = ham.shadow_energy(state, dt);
        Self {
            initial_shadow_energy: e0,
            tolerance,
        }
    }

    pub fn verify_invariance(
        &self,
        ham: &SeparableHamiltonian,
        state: &PhaseSpaceState,
        dt: f32,
    ) -> (bool, f32) {
        let cur = ham.shadow_energy(state, dt);
        let drift = (cur - self.initial_shadow_energy).abs() / self.initial_shadow_energy.abs().max(1e-6);
        (drift <= self.tolerance, drift)
    }
}

/// Benchmark metrics report for Symplectic Hamiltonian Dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymplecticDynamicsReport {
    /// Single-step integration latency in microseconds.
    pub step_latency_us: f64,
    /// Initial Hamiltonian energy H(0).
    pub initial_energy: f32,
    /// Final Hamiltonian energy H(T).
    pub final_energy: f32,
    /// Relative energy drift |H(T) - H(0)| / |H(0)|.
    pub energy_drift_rel: f32,
    /// Relative shadow Hamiltonian drift |H_shadow(T) - H_shadow(0)| / |H_shadow(0)|.
    pub shadow_energy_drift_rel: f32,
    /// Volume conservation error |det(J) - 1.0|.
    pub volume_conservation_det_err: f32,
    /// Relative Frobenius error of time-reversed reconstruction S(-dt)^N o S(dt)^N(z_0).
    pub time_reversibility_rel_err: f32,
    /// Total static memory footprint in bytes.
    pub total_memory_footprint_bytes: usize,
}

/// Verification harness and empirical evaluator for Symplectic Dynamics.
pub struct SymplecticDynamicsEvaluator;

impl SymplecticDynamicsEvaluator {
    /// Evaluates numerical Jacobian determinant via 2x2 coordinate-momentum block perturbation.
    pub fn evaluate_volume_conservation(
        ham: &SeparableHamiltonian,
        integrator: &StormerVerletIntegrator,
        dt: f32,
    ) -> f32 {
        let dim = ham.dim;
        let mut state = PhaseSpaceState {
            q: vec![0.5f32; dim],
            p: vec![0.2f32; dim],
        };

        // For a symplectic shear map, det(J) is identically 1.0.
        // We verify numerically along coordinate 0 and momentum 0:
        // J_block = [ dq_next/dq, dq_next/dp ]
        //           [ dp_next/dq, dp_next/dp ]
        let eps = 1e-4f32;

        let z_base = integrator.step(ham, &state, dt);

        // Perturb q[0]
        state.q[0] += eps;
        let z_dq = integrator.step(ham, &state, dt);
        state.q[0] -= eps;

        // Perturb p[0]
        state.p[0] += eps;
        let z_dp = integrator.step(ham, &state, dt);
        state.p[0] -= eps;

        let j11 = (z_dq.q[0] - z_base.q[0]) / eps;
        let j12 = (z_dp.q[0] - z_base.q[0]) / eps;
        let j21 = (z_dq.p[0] - z_base.p[0]) / eps;
        let j22 = (z_dp.p[0] - z_base.p[0]) / eps;

        let det = j11 * j22 - j12 * j21;
        (det - 1.0).abs()
    }

    /// Runs full empirical benchmarks over 100 long-horizon integration steps.
    pub fn evaluate() -> SymplecticDynamicsReport {
        let dim = DEFAULT_PHASE_DIM;
        let hidden_dim = 16;
        let ham = SeparableHamiltonian::new(dim, hidden_dim);
        let integrator = StormerVerletIntegrator;

        let mut initial_q = vec![0.0f32; dim];
        let mut initial_p = vec![0.0f32; dim];
        for i in 0..dim {
            initial_q[i] = (i as f32 * 0.25).sin();
            initial_p[i] = (i as f32 * 0.35).cos();
        }

        let state_0 = PhaseSpaceState::new(&initial_q, &initial_p);
        let dt = 0.01f32;
        let n_steps = 100;

        let e0 = ham.total_energy(&state_0);
        let tracker = ShadowHamiltonianTracker::new(&ham, &state_0, dt, 1e-3);

        // Benchmark in-place latency over 1,000 steps
        let mut q_bench = state_0.q.clone();
        let mut p_bench = state_0.p.clone();
        let mut grad_buf = vec![0.0f32; dim];

        let start = std::time::Instant::now();
        let bench_iters = 1000;
        for _ in 0..bench_iters {
            integrator.step_inplace(&ham, &mut q_bench, &mut p_bench, dt, &mut grad_buf);
        }
        let elapsed = start.elapsed();
        let step_latency_us = (elapsed.as_nanos() as f64 / bench_iters as f64) / 1000.0;

        // Roll out 100 steps
        let mut state_t = state_0.clone();
        for _ in 0..n_steps {
            state_t = integrator.step(&ham, &state_t, dt);
        }

        let e_final = ham.total_energy(&state_t);
        let e_drift_rel = (e_final - e0).abs() / e0.abs().max(1e-6);

        let (_, shadow_drift) = tracker.verify_invariance(&ham, &state_t, dt);

        // Test exact time-reversibility: reverse 100 steps
        let mut state_rev = state_t.clone();
        for _ in 0..n_steps {
            state_rev = integrator.step_backward(&ham, &state_rev, dt);
        }

        let mut diff_sq = 0.0f32;
        let mut orig_sq = 0.0f32;
        for i in 0..dim {
            let err_q = state_rev.q[i] - state_0.q[i];
            let err_p = state_rev.p[i] - state_0.p[i];
            diff_sq += err_q * err_q + err_p * err_p;
            orig_sq += state_0.q[i] * state_0.q[i] + state_0.p[i] * state_0.p[i];
        }
        let rev_err = (diff_sq / orig_sq.max(1e-12)).sqrt();

        let vol_err = Self::evaluate_volume_conservation(&ham, &integrator, dt);

        let mem_bytes = std::mem::size_of::<SeparableHamiltonian>()
            + (dim * 4 * 2) // q, p
            + (dim * 4) // inv_mass
            + (dim * 4) // stiffness
            + (hidden_dim * dim * 4) // hidden_weights
            + (hidden_dim * 4 * 2); // biases, scales

        SymplecticDynamicsReport {
            step_latency_us,
            initial_energy: e0,
            final_energy: e_final,
            energy_drift_rel: e_drift_rel,
            shadow_energy_drift_rel: shadow_drift,
            volume_conservation_det_err: vol_err,
            time_reversibility_rel_err: rev_err,
            total_memory_footprint_bytes: mem_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_space_norms_and_initialization() {
        let q = [1.0, 2.0, 3.0];
        let p = [4.0, 5.0, 6.0];
        let s = PhaseSpaceState::new(&q, &p);
        assert_eq!(s.dim(), 3);
        assert!((s.norm_q() - (14.0f32).sqrt()).abs() < 1e-6);
        assert!((s.norm_p() - (77.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_stormer_verlet_time_reversibility() {
        let dim = 8;
        let hidden_dim = 16;
        let ham = SeparableHamiltonian::new(dim, hidden_dim);
        let sv = StormerVerletIntegrator;

        let state_0 = PhaseSpaceState {
            q: vec![0.3; dim],
            p: vec![-0.4; dim],
        };

        let dt = 0.02;
        let mut forward = state_0.clone();
        for _ in 0..50 {
            forward = sv.step(&ham, &forward, dt);
        }

        let mut backward = forward.clone();
        for _ in 0..50 {
            backward = sv.step_backward(&ham, &backward, dt);
        }

        for i in 0..dim {
            assert!((backward.q[i] - state_0.q[i]).abs() < 1e-5);
            assert!((backward.p[i] - state_0.p[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_yoshida4th_higher_order_energy_stability() {
        let dim = 8;
        let hidden_dim = 16;
        let ham = SeparableHamiltonian::new(dim, hidden_dim);
        let y4 = Yoshida4thIntegrator;

        let state_0 = PhaseSpaceState {
            q: vec![0.5; dim],
            p: vec![0.1; dim],
        };

        let e0 = ham.total_energy(&state_0);
        let mut state_t = state_0;
        let dt = 0.05;
        for _ in 0..50 {
            state_t = y4.step(&ham, &state_t, dt);
        }
        let e_final = ham.total_energy(&state_t);
        let drift = (e_final - e0).abs() / e0.abs();
        assert!(drift < 1e-3, "4th order energy drift must be < 1e-3, got {}", drift);
    }

    #[test]
    fn test_symplectic_evaluator_metrics() {
        let report = SymplecticDynamicsEvaluator::evaluate();
        assert!(report.step_latency_us < 5.0, "Latency must be < 5.0 us, got {}", report.step_latency_us);
        assert!(report.volume_conservation_det_err < 1e-3, "Volume error must be < 1e-3");
        assert!(report.shadow_energy_drift_rel < 1e-3, "Shadow drift must be < 1e-3");
        assert!(report.time_reversibility_rel_err < 1e-4, "Reversibility error must be < 1e-4");
        assert!(report.total_memory_footprint_bytes < 32_000);
    }
}
