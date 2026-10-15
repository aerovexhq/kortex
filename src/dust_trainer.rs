//! Phase 31: Dust Zeroth-Order Full-Cycle Non-Backprop Training & Multi-Layer Pretraining Pipeline.
//!
//! Implements Q Labs' Dust activation-space virtual population search for full multi-layer
//! neural network training without reverse-mode automatic differentiation:
//! 1. Evaluates multi-layer forward architectures forward-only without backward computational graphs.
//! 2. Perturbs activations across all layers simultaneously: y_{t, l} -> y_{t, l} + sigma * a_{t, l}.
//! 3. Evaluates sequence-level loss and computes causal credit assignment: r_{t, l} = sum_{s >= t} gamma^{s-t} c_s.
//! 4. Synthesizes layer-wise outer product gradients: G_{W_l} = sum_t g_hat_{t, l} x_{t, l-1}^T.
//! 5. Updates parameters via zero-grad AdamW with decoupled weight decay in bounded O(1) activation memory.

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use crate::zero_adapt::DeterministicGaussianRng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Supported activation functions for Dust layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Identity linear activation: f(x) = x
    Identity,
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit approximation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    GELU,
}

impl ActivationType {
    /// Applies the activation function element-wise.
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            Self::ReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Self::GELU => {
                let sqrt_2_over_pi = 0.797_884_6;
                let inner = sqrt_2_over_pi * (x + 0.044_715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
        }
    }
}

/// A single trainable layer in a Dust multi-layer neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustLayer {
    /// Input dimension d_in
    pub d_in: usize,
    /// Output dimension d_out
    pub d_out: usize,
    /// Weight matrix stored in row-major layout (d_out * d_in)
    pub weights: Vec<f32>,
    /// Bias vector (d_out)
    pub bias: Vec<f32>,
    /// Accumulated weight gradients (d_out * d_in)
    pub grad_weights: Vec<f32>,
    /// Accumulated bias gradients (d_out)
    pub grad_bias: Vec<f32>,
    /// AdamW first moment for weights
    pub m_weights: Vec<f32>,
    /// AdamW second moment for weights
    pub v_weights: Vec<f32>,
    /// AdamW first moment for bias
    pub m_bias: Vec<f32>,
    /// AdamW second moment for bias
    pub v_bias: Vec<f32>,
    /// Activation function type
    pub activation: ActivationType,
}

impl DustLayer {
    /// Creates a new DustLayer with deterministic Xavier/Glorot normal initialization.
    pub fn new(d_in: usize, d_out: usize, activation: ActivationType, rng: &mut DeterministicGaussianRng) -> Self {
        let std_dev = (2.0 / (d_in + d_out) as f32).sqrt();
        let num_weights = d_out * d_in;
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push(rng.next_standard_normal() * std_dev);
        }
        let bias = vec![0.0f32; d_out];

        Self {
            d_in,
            d_out,
            weights,
            bias,
            grad_weights: vec![0.0f32; num_weights],
            grad_bias: vec![0.0f32; d_out],
            m_weights: vec![0.0f32; num_weights],
            v_weights: vec![0.0f32; num_weights],
            m_bias: vec![0.0f32; d_out],
            v_bias: vec![0.0f32; d_out],
            activation,
        }
    }

    /// Evaluates linear projection + activation: y = phi(W * x + b).
    #[inline]
    pub fn forward_token(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(out.len(), self.d_out);

        for i in 0..self.d_out {
            let row_offset = i * self.d_in;
            let mut sum = self.bias[i];
            for j in 0..self.d_in {
                sum += self.weights[row_offset + j] * x[j];
            }
            out[i] = self.activation.apply(sum);
        }
    }

    /// Evaluates linear projection with activation-space perturbation: y = phi(W * x + b + perturbation).
    #[inline]
    pub fn forward_token_perturbed(&self, x: &[f32], perturbation: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(perturbation.len(), self.d_out);
        debug_assert_eq!(out.len(), self.d_out);

        for i in 0..self.d_out {
            let row_offset = i * self.d_in;
            let mut sum = self.bias[i] + perturbation[i];
            for j in 0..self.d_in {
                sum += self.weights[row_offset + j] * x[j];
            }
            out[i] = self.activation.apply(sum);
        }
    }

    /// Resets accumulated gradients to zero.
    #[inline]
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    }
}

/// Multi-layer neural network trained entirely without backpropagation via Dust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustMultiLayerNetwork {
    /// Sequence of layers
    pub layers: Vec<DustLayer>,
    /// Pre-allocated forward activation buffers: [layer_idx][token_idx * d_out]
    activations: Vec<Vec<f32>>,
    /// Pre-allocated perturbed ping-pong buffer A: [token_idx * max_dim]
    perturbed_buf_a: Vec<f32>,
    /// Pre-allocated perturbed ping-pong buffer B: [token_idx * max_dim]
    perturbed_buf_b: Vec<f32>,
}

impl DustMultiLayerNetwork {
    /// Builds a multi-layer network given layer dimensions and activations.
    pub fn new(
        dims: &[usize],
        activations: &[ActivationType],
        max_seq_len: usize,
        rng: &mut DeterministicGaussianRng,
    ) -> Self {
        assert!(dims.len() >= 2, "Network must have at least 1 layer (2 dimensions)");
        assert_eq!(dims.len() - 1, activations.len(), "Each layer must have an activation type");

        let mut layers = Vec::with_capacity(dims.len() - 1);
        let mut act_buffers = Vec::with_capacity(dims.len());

        // Buffer for layer 0 (input)
        act_buffers.push(vec![0.0f32; max_seq_len * dims[0]]);

        let mut max_dim = dims[0];
        for l in 0..(dims.len() - 1) {
            layers.push(DustLayer::new(dims[l], dims[l + 1], activations[l], rng));
            act_buffers.push(vec![0.0f32; max_seq_len * dims[l + 1]]);
            if dims[l + 1] > max_dim {
                max_dim = dims[l + 1];
            }
        }

        Self {
            layers,
            activations: act_buffers,
            perturbed_buf_a: vec![0.0f32; max_seq_len * max_dim],
            perturbed_buf_b: vec![0.0f32; max_seq_len * max_dim],
        }
    }

    /// Evaluates clean forward pass through all layers for a sequence of T tokens.
    /// Returns reference to output tokens at layer L: slice of size T * d_out.
    pub fn forward_sequence(&mut self, inputs: &[f32], seq_len: usize) -> &[f32] {
        let d_in = self.layers[0].d_in;
        debug_assert_eq!(inputs.len(), seq_len * d_in);

        // Copy inputs into activation buffer 0
        self.activations[0][..seq_len * d_in].copy_from_slice(inputs);

        // Forward through each layer using split_at_mut to avoid simultaneous mutable borrows
        for l in 0..self.layers.len() {
            let in_dim = self.layers[l].d_in;
            let out_dim = self.layers[l].d_out;

            let (left, right) = self.activations.split_at_mut(l + 1);
            let in_slice = &left[l];
            let out_slice = &mut right[0];

            for t in 0..seq_len {
                let in_start = t * in_dim;
                let out_start = t * out_dim;
                let token_in = &in_slice[in_start..in_start + in_dim];
                let token_out = &mut out_slice[out_start..out_start + out_dim];
                self.layers[l].forward_token(token_in, token_out);
            }
        }

        let last_idx = self.layers.len();
        let last_dim = self.layers[last_idx - 1].d_out;
        &self.activations[last_idx][..seq_len * last_dim]
    }

    /// Evaluates forward pass with perturbation injected specifically at layer `perturb_layer`.
    /// Returns slice of output tokens: size T * d_out.
    pub fn forward_sequence_perturbed(
        &mut self,
        seq_len: usize,
        perturb_layer: usize,
        perturbations: &[f32],
    ) -> &[f32] {
        let num_layers = self.layers.len();
        debug_assert!(perturb_layer < num_layers);
        let p_dim = self.layers[perturb_layer].d_out;
        let in_dim = self.layers[perturb_layer].d_in;
        debug_assert_eq!(perturbations.len(), seq_len * p_dim);

        // Perturb the output of perturb_layer into perturbed_buf_a
        let in_slice = &self.activations[perturb_layer];
        for t in 0..seq_len {
            let in_start = t * in_dim;
            let p_start = t * p_dim;
            let token_in = &in_slice[in_start..in_start + in_dim];
            let token_p = &perturbations[p_start..p_start + p_dim];
            let token_out = &mut self.perturbed_buf_a[p_start..p_start + p_dim];
            self.layers[perturb_layer].forward_token_perturbed(token_in, token_p, token_out);
        }

        if perturb_layer + 1 == num_layers {
            return &self.perturbed_buf_a[..seq_len * p_dim];
        }

        // Downstream layers propagate through ping-pong buffers without touching self.activations
        let mut in_is_a = true;
        for l in (perturb_layer + 1)..num_layers {
            let l_in_dim = self.layers[l].d_in;
            let l_out_dim = self.layers[l].d_out;

            for t in 0..seq_len {
                let in_start = t * l_in_dim;
                let out_start = t * l_out_dim;

                if in_is_a {
                    let token_in = &self.perturbed_buf_a[in_start..in_start + l_in_dim];
                    let token_out = &mut self.perturbed_buf_b[out_start..out_start + l_out_dim];
                    self.layers[l].forward_token(token_in, token_out);
                } else {
                    let token_in = &self.perturbed_buf_b[in_start..in_start + l_in_dim];
                    let token_out = &mut self.perturbed_buf_a[out_start..out_start + l_out_dim];
                    self.layers[l].forward_token(token_in, token_out);
                }
            }
            in_is_a = !in_is_a;
        }

        let last_dim = self.layers[num_layers - 1].d_out;
        if in_is_a {
            &self.perturbed_buf_a[..seq_len * last_dim]
        } else {
            &self.perturbed_buf_b[..seq_len * last_dim]
        }
    }

    /// Accumulates outer-product weight and bias gradients for layer `layer_idx`.
    pub fn accumulate_gradients(
        &mut self,
        layer_idx: usize,
        seq_len: usize,
        output_errors: &[f32],
    ) {
        let in_dim = self.layers[layer_idx].d_in;
        let out_dim = self.layers[layer_idx].d_out;
        let in_act = &self.activations[layer_idx];
        let layer = &mut self.layers[layer_idx];

        for t in 0..seq_len {
            let g_t = &output_errors[t * out_dim..(t + 1) * out_dim];
            let x_t = &in_act[t * in_dim..(t + 1) * in_dim];

            for i in 0..out_dim {
                let g_val = g_t[i];
                let row_offset = i * in_dim;
                for j in 0..in_dim {
                    layer.grad_weights[row_offset + j] += g_val * x_t[j];
                }
                layer.grad_bias[i] += g_val;
            }
        }

        let inv_seq = 1.0 / (seq_len as f32);
        for g in layer.grad_weights.iter_mut() {
            *g *= inv_seq;
        }
        for g in layer.grad_bias.iter_mut() {
            *g *= inv_seq;
        }
    }

    /// Accessor for clean cached activations at layer `l`: size T * d_l.
    pub fn layer_activations(&self, layer_idx: usize, seq_len: usize) -> &[f32] {
        let dim = if layer_idx == 0 {
            self.layers[0].d_in
        } else {
            self.layers[layer_idx - 1].d_out
        };
        &self.activations[layer_idx][..seq_len * dim]
    }
}

/// Hyperparameter configuration for the Dust non-backprop training engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustTrainerConfig {
    /// Activation perturbation standard deviation sigma (typically 0.02 - 0.10)
    pub perturbation_sigma: f32,
    /// Number of virtual population draws per step (typically 8 - 32)
    pub num_draws: usize,
    /// Causal credit assignment discount factor gamma in [0.0, 1.0)
    pub causal_gamma: f32,
    /// AdamW learning rate eta
    pub learning_rate: f32,
    /// AdamW decoupled weight decay lambda
    pub weight_decay: f32,
    /// AdamW first moment exponential decay beta1
    pub beta1: f32,
    /// AdamW second moment exponential decay beta2
    pub beta2: f32,
    /// Numerical stability epsilon for AdamW
    pub eps: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for DustTrainerConfig {
    fn default() -> Self {
        Self {
            perturbation_sigma: 0.05,
            num_draws: 16,
            causal_gamma: 0.95,
            learning_rate: 0.008,
            weight_decay: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            seed: 42,
        }
    }
}

/// Full-cycle non-backpropagation trainer powered by Dust activation virtual populations.
pub struct DustTrainer {
    /// Configuration parameters
    pub config: DustTrainerConfig,
    /// Deterministic PRNG
    pub rng: DeterministicGaussianRng,
    /// Current training step counter
    pub step_count: u64,
    /// Pre-allocated scratch buffers
    perturbation_draws: Vec<Vec<f32>>,
    clean_losses: Vec<f32>,
    perturbed_losses: Vec<f32>,
    token_credits: Vec<f32>,
    output_error_estimates: Vec<f32>,
}

impl DustTrainer {
    /// Initializes a new DustTrainer with pre-allocated scratch memory.
    pub fn new(config: DustTrainerConfig, max_seq_len: usize, max_layer_dim: usize) -> Self {
        let rng = DeterministicGaussianRng::new(config.seed);
        let num_draws = config.num_draws;

        let mut perturbation_draws = Vec::with_capacity(num_draws);
        for _ in 0..num_draws {
            perturbation_draws.push(vec![0.0f32; max_seq_len * max_layer_dim]);
        }

        Self {
            rng,
            step_count: 0,
            perturbation_draws,
            clean_losses: vec![0.0f32; max_seq_len],
            perturbed_losses: vec![0.0f32; max_seq_len],
            token_credits: vec![0.0f32; max_seq_len],
            output_error_estimates: vec![0.0f32; max_seq_len * max_layer_dim],
            config,
        }
    }

    /// Executes one training step on a sequence of inputs and targets using Dust virtual population search.
    /// Completely bypasses reverse-mode autograd: calculates outer-product gradients and updates parameters.
    /// Returns the mean clean sequence loss before the update.
    pub fn train_step(
        &mut self,
        network: &mut DustMultiLayerNetwork,
        inputs: &[f32],
        targets: &[f32],
        seq_len: usize,
    ) -> f32 {
        self.step_count += 1;
        let d_out = network.layers.last().unwrap().d_out;
        debug_assert_eq!(targets.len(), seq_len * d_out);

        // 1. Clean forward pass
        let outputs = network.forward_sequence(inputs, seq_len);

        // Compute clean token-level MSE loss
        let mut total_clean_loss = 0.0f32;
        for t in 0..seq_len {
            let mut token_loss = 0.0f32;
            for j in 0..d_out {
                let diff = outputs[t * d_out + j] - targets[t * d_out + j];
                token_loss += diff * diff;
            }
            token_loss /= d_out as f32;
            self.clean_losses[t] = token_loss;
            total_clean_loss += token_loss;
        }
        let mean_clean_loss = total_clean_loss / seq_len as f32;

        // Reset layer gradient accumulators
        for layer in &mut network.layers {
            layer.zero_grad();
        }

        // 2. Iterate through each layer to synthesize zeroth-order gradients
        let num_layers = network.layers.len();
        for l in 0..num_layers {
            let layer_out_dim = network.layers[l].d_out;
            let total_perturb_elements = seq_len * layer_out_dim;

            // Generate K perturbation draws for this layer
            for i in 0..self.config.num_draws {
                let draw_buf = &mut self.perturbation_draws[i][..total_perturb_elements];
                for elem in draw_buf.iter_mut() {
                    *elem = self.rng.next_standard_normal() * self.config.perturbation_sigma;
                }
            }

            // Zero output error accumulator for this layer
            self.output_error_estimates[..total_perturb_elements].fill(0.0);

            // Evaluate each draw
            for i in 0..self.config.num_draws {
                let pert_slice = &self.perturbation_draws[i][..total_perturb_elements];
                let pert_outputs = network.forward_sequence_perturbed(seq_len, l, pert_slice);

                // Compute perturbed token losses and raw loss improvement c_s = L_clean - L_perturbed
                for t in 0..seq_len {
                    let mut token_loss = 0.0f32;
                    for j in 0..d_out {
                        let diff = pert_outputs[t * d_out + j] - targets[t * d_out + j];
                        token_loss += diff * diff;
                    }
                    token_loss /= d_out as f32;
                    self.perturbed_losses[t] = token_loss;
                }

                // Causal backward credit accumulation: r_t = c_t + gamma * r_{t+1}
                let mut future_credit = 0.0f32;
                for t in (0..seq_len).rev() {
                    let c_t = self.clean_losses[t] - self.perturbed_losses[t];
                    let r_t = c_t + self.config.causal_gamma * future_credit;
                    self.token_credits[t] = r_t;
                    future_credit = r_t;
                }

                // Accumulate credit-weighted perturbation:
                // g_hat_t += r_t * a_t
                for t in 0..seq_len {
                    let r_t = self.token_credits[t];
                    let offset = t * layer_out_dim;
                    for j in 0..layer_out_dim {
                        self.output_error_estimates[offset + j] += r_t * pert_slice[offset + j];
                    }
                }
            }

            // Normalize zeroth-order output gradient estimate:
            // g_hat_t = - (1 / (K * sigma^2)) * sum_i r_t^{(i)} * Delta_y^{(i)}
            // Since Delta_y = sigma * a, this is - (1 / (K * sigma)) * sum_i r_t * a
            let norm_factor = -1.0 / (self.config.num_draws as f32 * self.config.perturbation_sigma * self.config.perturbation_sigma);
            for elem in self.output_error_estimates[..total_perturb_elements].iter_mut() {
                *elem *= norm_factor;
            }

            // 3. Outer-product gradient synthesis: G_W_l = sum_t g_hat_{t, l} * x_{t, l-1}^T
            network.accumulate_gradients(l, seq_len, &self.output_error_estimates);
        }

        // 4. Decoupled AdamW parameter update across all layers
        self.apply_adamw_step(network);

        mean_clean_loss
    }

    /// Applies decoupled AdamW parameter updates to all layers using synthesized gradients.
    fn apply_adamw_step(&self, network: &mut DustMultiLayerNetwork) {
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;
        let eps = self.config.eps;
        let t_f32 = self.step_count as f32;

        let bc1 = 1.0 - beta1.powf(t_f32);
        let bc2 = 1.0 - beta2.powf(t_f32);

        for layer in &mut network.layers {
            // Update weights with AdamW + decoupled weight decay
            for idx in 0..layer.weights.len() {
                let g = layer.grad_weights[idx];
                layer.m_weights[idx] = beta1 * layer.m_weights[idx] + (1.0 - beta1) * g;
                layer.v_weights[idx] = beta2 * layer.v_weights[idx] + (1.0 - beta2) * (g * g);

                let m_hat = layer.m_weights[idx] / bc1;
                let v_hat = layer.v_weights[idx] / bc2;

                let update = (m_hat / (v_hat.sqrt() + eps)) + wd * layer.weights[idx];
                layer.weights[idx] -= lr * update;
            }

            // Update bias with AdamW (no weight decay on bias)
            for idx in 0..layer.bias.len() {
                let g = layer.grad_bias[idx];
                layer.m_bias[idx] = beta1 * layer.m_bias[idx] + (1.0 - beta1) * g;
                layer.v_bias[idx] = beta2 * layer.v_bias[idx] + (1.0 - beta2) * (g * g);

                let m_hat = layer.m_bias[idx] / bc1;
                let v_hat = layer.v_bias[idx] / bc2;

                let update = m_hat / (v_hat.sqrt() + eps);
                layer.bias[idx] -= lr * update;
            }
        }
    }
}

/// Empirical verification report for Dust non-backprop training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustTrainingReport {
    /// Number of training layers
    pub num_layers: usize,
    /// Total trainable parameters
    pub total_parameters: usize,
    /// Static working memory footprint in KiB
    pub memory_footprint_kib: f32,
    /// Initial training sequence loss
    pub initial_loss: f32,
    /// Final training sequence loss after N epochs
    pub final_loss: f32,
    /// Percentage loss reduction
    pub loss_reduction_pct: f32,
    /// Mean gradient cosine similarity vs exact analytical gradient
    pub mean_gradient_cosine_similarity: f32,
    /// Mean per-step execution latency in microseconds
    pub step_latency_us: f32,
    /// Training epochs evaluated
    pub epochs_completed: usize,
}

/// Evaluator harness benchmarking Dust non-backpropagation training performance.
pub struct DustTrainingEvaluator {
    /// Dimension topology [d_in, d_hidden, d_out]
    pub dims: Vec<usize>,
    /// Number of tokens in training sequence
    pub seq_len: usize,
    /// Number of training epochs
    pub epochs: usize,
}

impl Default for DustTrainingEvaluator {
    fn default() -> Self {
        Self {
            dims: vec![16, 24, 8],
            seq_len: 12,
            epochs: 60,
        }
    }
}

impl DustTrainingEvaluator {
    /// Runs a full empirical training evaluation and returns a performance report.
    pub fn evaluate(&self) -> DustTrainingReport {
        let mut rng = DeterministicGaussianRng::new(1337);
        let activations = vec![ActivationType::ReLU, ActivationType::Identity];
        let mut network = DustMultiLayerNetwork::new(&self.dims, &activations, self.seq_len, &mut rng);

        // Generate synthetic training sequence: inputs X and target Y
        let d_in = self.dims[0];
        let d_out = *self.dims.last().unwrap();
        let mut inputs = vec![0.0f32; self.seq_len * d_in];
        let mut targets = vec![0.0f32; self.seq_len * d_out];

        for t in 0..self.seq_len {
            for j in 0..d_in {
                inputs[t * d_in + j] = ((t as f32 * 0.3) + (j as f32 * 0.1)).sin();
            }
            for j in 0..d_out {
                // Target is a deterministic non-linear transformation of inputs
                targets[t * d_out + j] = ((t as f32 * 0.2) + (j as f32 * 0.4)).cos() * 0.5;
            }
        }

        let config = DustTrainerConfig {
            perturbation_sigma: 0.04,
            num_draws: 32,
            causal_gamma: 0.90,
            learning_rate: 0.015,
            weight_decay: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            seed: 42,
        };

        let mut trainer = DustTrainer::new(config, self.seq_len, 32);

        // Compute initial loss
        let initial_outputs = network.forward_sequence(&inputs, self.seq_len);
        let mut initial_loss = 0.0f32;
        for t in 0..self.seq_len {
            for j in 0..d_out {
                let diff = initial_outputs[t * d_out + j] - targets[t * d_out + j];
                initial_loss += diff * diff;
            }
        }
        initial_loss /= (self.seq_len * d_out) as f32;

        // Measure training loop latency
        let start = Instant::now();
        let mut final_loss = initial_loss;
        for _ in 0..self.epochs {
            final_loss = trainer.train_step(&mut network, &inputs, &targets, self.seq_len);
        }
        let elapsed = start.elapsed();
        let step_latency_us = (elapsed.as_micros() as f32) / (self.epochs as f32);

        let loss_reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100.0;

        // Calculate gradient cosine similarity vs exact analytical gradient on last layer
        let analytical_grad = Self::compute_analytical_last_layer_gradient(&network, &inputs, &targets, self.seq_len);
        let last_layer = network.layers.last().unwrap();
        let cos_sim = Self::cosine_similarity(&last_layer.grad_weights, &analytical_grad);

        // Memory footprint calculation
        let mut total_params = 0;
        let mut total_bytes = 0;
        for l in &network.layers {
            let p = l.weights.len() + l.bias.len();
            total_params += p;
            // weights, bias, grad_weights, grad_bias, m_weights, v_weights, m_bias, v_bias
            total_bytes += (p * 4) * 4;
        }

        DustTrainingReport {
            num_layers: network.layers.len(),
            total_parameters: total_params,
            memory_footprint_kib: (total_bytes as f32) / 1024.0,
            initial_loss,
            final_loss,
            loss_reduction_pct,
            mean_gradient_cosine_similarity: cos_sim,
            step_latency_us,
            epochs_completed: self.epochs,
        }
    }

    /// Computes exact analytical gradient for the last linear layer to verify Dust estimator accuracy.
    fn compute_analytical_last_layer_gradient(
        network: &DustMultiLayerNetwork,
        _inputs: &[f32],
        targets: &[f32],
        seq_len: usize,
    ) -> Vec<f32> {
        let last_idx = network.layers.len() - 1;
        let d_in = network.layers[last_idx].d_in;
        let d_out = network.layers[last_idx].d_out;
        let outputs = &network.activations[last_idx + 1][..seq_len * d_out];
        let prev_act = &network.activations[last_idx][..seq_len * d_in];

        let mut grad = vec![0.0f32; d_out * d_in];
        let factor = 2.0 / ((seq_len * d_out) as f32);

        for t in 0..seq_len {
            for i in 0..d_out {
                let err = (outputs[t * d_out + i] - targets[t * d_out + i]) * factor;
                let row_offset = i * d_in;
                for j in 0..d_in {
                    grad[row_offset + j] += err * prev_act[t * d_in + j];
                }
            }
        }
        grad
    }

    /// Computes cosine similarity between two vector slices.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            dot / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let relu = ActivationType::ReLU;
        assert_eq!(relu.apply(-2.5), 0.0);
        assert_eq!(relu.apply(3.5), 3.5);

        let id = ActivationType::Identity;
        assert_eq!(id.apply(-1.0), -1.0);
        assert_eq!(id.apply(1.0), 1.0);

        let gelu = ActivationType::GELU;
        assert!((gelu.apply(0.0) - 0.0).abs() < 1e-5);
        assert!(gelu.apply(2.0) > 1.9);
    }

    #[test]
    fn test_dust_forward_pass_dimensions() {
        let mut rng = DeterministicGaussianRng::new(42);
        let dims = vec![8, 16, 4];
        let acts = vec![ActivationType::ReLU, ActivationType::Identity];
        let mut net = DustMultiLayerNetwork::new(&dims, &acts, 10, &mut rng);

        let inputs = vec![0.5f32; 10 * 8];
        let out = net.forward_sequence(&inputs, 10);
        assert_eq!(out.len(), 10 * 4);
    }

    #[test]
    fn test_dust_training_loss_convergence() {
        let evaluator = DustTrainingEvaluator {
            dims: vec![12, 16, 6],
            seq_len: 8,
            epochs: 50,
        };
        let report = evaluator.evaluate();

        assert!(
            report.loss_reduction_pct > 25.0,
            "Dust training failed to achieve > 25% loss reduction: was {:.2}%",
            report.loss_reduction_pct
        );
        assert!(
            report.final_loss < report.initial_loss,
            "Final loss ({:.4}) must be lower than initial loss ({:.4})",
            report.final_loss,
            report.initial_loss
        );
        assert!(
            report.memory_footprint_kib < 120.0,
            "Working memory footprint ({:.2} KiB) exceeded 120 KiB limit",
            report.memory_footprint_kib
        );
    }
}
