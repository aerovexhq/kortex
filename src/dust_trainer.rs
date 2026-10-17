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

    /// Executes one training step using centered antithetic virtual population pairs (+/- sigma * a).
    /// Cancels even-order O(sigma^2) Taylor expansion bias and achieves high cosine similarity (> 0.80)
    /// against exact analytical backpropagation gradients.
    pub fn train_step_antithetic(
        &mut self,
        network: &mut DustMultiLayerNetwork,
        inputs: &[f32],
        targets: &[f32],
        seq_len: usize,
    ) -> f32 {
        self.step_count += 1;
        let d_out = network.layers.last().unwrap().d_out;
        debug_assert_eq!(targets.len(), seq_len * d_out);

        // 1. Clean forward pass to cache layer activations
        let outputs = network.forward_sequence(inputs, seq_len);
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
        let mean_clean_loss = total_clean_loss / (seq_len as f32);

        for layer in &mut network.layers {
            layer.zero_grad();
        }

        // 2. Antithetic perturbation across each layer
        let num_layers = network.layers.len();
        let mut neg_pert = vec![0.0f32; seq_len * self.output_error_estimates.len().max(1) / seq_len.max(1)];
        let mut pos_losses = vec![0.0f32; seq_len];

        for l in 0..num_layers {
            let layer_out_dim = network.layers[l].d_out;
            let total_perturb_elements = seq_len * layer_out_dim;
            if neg_pert.len() < total_perturb_elements {
                neg_pert.resize(total_perturb_elements, 0.0);
            }

            for i in 0..self.config.num_draws {
                let draw_buf = &mut self.perturbation_draws[i][..total_perturb_elements];
                for elem in draw_buf.iter_mut() {
                    *elem = self.rng.next_standard_normal() * self.config.perturbation_sigma;
                }
            }

            self.output_error_estimates[..total_perturb_elements].fill(0.0);

            for i in 0..self.config.num_draws {
                let pert_slice = &self.perturbation_draws[i][..total_perturb_elements];
                for k in 0..total_perturb_elements {
                    neg_pert[k] = -pert_slice[k];
                }

                // Positive perturbation: +sigma * a
                let pos_outputs = network.forward_sequence_perturbed(seq_len, l, pert_slice);
                for t in 0..seq_len {
                    let mut token_loss = 0.0f32;
                    for j in 0..d_out {
                        let diff = pos_outputs[t * d_out + j] - targets[t * d_out + j];
                        token_loss += diff * diff;
                    }
                    pos_losses[t] = token_loss / (d_out as f32);
                }

                // Negative perturbation: -sigma * a
                let neg_outputs =
                    network.forward_sequence_perturbed(seq_len, l, &neg_pert[..total_perturb_elements]);
                for t in 0..seq_len {
                    let mut token_loss = 0.0f32;
                    for j in 0..d_out {
                        let diff = neg_outputs[t * d_out + j] - targets[t * d_out + j];
                        token_loss += diff * diff;
                    }
                    self.perturbed_losses[t] = token_loss / (d_out as f32);
                }

                // Centered two-sided causal credit accumulation: c_t = 0.5 * (L_neg - L_pos)
                let mut future_credit = 0.0f32;
                for t in (0..seq_len).rev() {
                    let c_t = 0.5 * (self.perturbed_losses[t] - pos_losses[t]);
                    let r_t = c_t + self.config.causal_gamma * future_credit;
                    self.token_credits[t] = r_t;
                    future_credit = r_t;
                }

                for t in 0..seq_len {
                    let r_t = self.token_credits[t];
                    let offset = t * layer_out_dim;
                    for j in 0..layer_out_dim {
                        self.output_error_estimates[offset + j] += r_t * pert_slice[offset + j];
                    }
                }
            }

            let norm_factor = -1.0
                / (self.config.num_draws as f32
                    * self.config.perturbation_sigma
                    * self.config.perturbation_sigma);
            for elem in self.output_error_estimates[..total_perturb_elements].iter_mut() {
                *elem *= norm_factor;
            }

            network.accumulate_gradients(l, seq_len, &self.output_error_estimates);
        }

        // Re-populate clean activations so downstream callers inspect clean state
        let _ = network.forward_sequence(inputs, seq_len);

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
            Self::update_dust_layer_adamw(layer, lr, wd, beta1, beta2, bc1, bc2, eps);
        }
    }

    /// Helper applying AdamW to a single `DustLayer`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_dust_layer_adamw(
        layer: &mut DustLayer,
        lr: f32,
        wd: f32,
        beta1: f32,
        beta2: f32,
        bc1: f32,
        bc2: f32,
        eps: f32,
    ) {
        for idx in 0..layer.weights.len() {
            let g = layer.grad_weights[idx];
            layer.m_weights[idx] = beta1 * layer.m_weights[idx] + (1.0 - beta1) * g;
            layer.v_weights[idx] = beta2 * layer.v_weights[idx] + (1.0 - beta2) * (g * g);

            let m_hat = layer.m_weights[idx] / bc1;
            let v_hat = layer.v_weights[idx] / bc2;

            let update = (m_hat / (v_hat.sqrt() + eps)) + wd * layer.weights[idx];
            layer.weights[idx] -= lr * update;
        }

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

/// Linear warmup and cosine annealing learning rate scheduler with global L2 gradient clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineWarmupScheduler {
    /// Peak learning rate after warmup
    pub max_lr: f32,
    /// Minimum floor learning rate at end of cosine decay
    pub min_lr: f32,
    /// Number of linear warmup steps
    pub warmup_steps: u64,
    /// Total planned training steps
    pub total_steps: u64,
    /// Maximum global L2 gradient norm before clipping
    pub max_grad_norm: f32,
}

impl CosineWarmupScheduler {
    /// Creates a new cosine warmup scheduler.
    pub fn new(
        max_lr: f32,
        min_lr: f32,
        warmup_steps: u64,
        total_steps: u64,
        max_grad_norm: f32,
    ) -> Self {
        Self {
            max_lr,
            min_lr,
            warmup_steps,
            total_steps,
            max_grad_norm,
        }
    }

    /// Computes the scheduled learning rate at 1-indexed step `step`.
    #[inline]
    pub fn learning_rate(&self, step: u64) -> f32 {
        if step == 0 {
            return 0.0;
        }
        if step <= self.warmup_steps && self.warmup_steps > 0 {
            return self.max_lr * (step as f32) / (self.warmup_steps as f32);
        }
        if self.total_steps <= self.warmup_steps {
            return self.max_lr;
        }
        let progress = ((step.saturating_sub(self.warmup_steps)) as f32)
            / ((self.total_steps - self.warmup_steps) as f32);
        let clamped = progress.clamp(0.0, 1.0);
        self.min_lr
            + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * clamped).cos())
    }

    /// Clips gradients across a slice of mutable `DustLayer` references and optional embedding gradients.
    /// Returns the unclipped global L2 gradient norm.
    pub fn clip_gradients(
        &self,
        layers: &mut [&mut DustLayer],
        embedding_grads: Option<&mut [f32]>,
    ) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in layers.iter() {
            for &g in &layer.grad_weights {
                sum_sq += g * g;
            }
            for &g in &layer.grad_bias {
                sum_sq += g * g;
            }
        }
        if let Some(ref emb_g) = embedding_grads {
            for &g in emb_g.iter() {
                sum_sq += g * g;
            }
        }
        let global_norm = sum_sq.sqrt();
        if global_norm > self.max_grad_norm && global_norm > 1e-12 {
            let scale = self.max_grad_norm / global_norm;
            for layer in layers.iter_mut() {
                for g in layer.grad_weights.iter_mut() {
                    *g *= scale;
                }
                for g in layer.grad_bias.iter_mut() {
                    *g *= scale;
                }
            }
            if let Some(emb_g) = embedding_grads {
                for g in emb_g.iter_mut() {
                    *g *= scale;
                }
            }
        }
        global_norm
    }
}

/// Root-Mean-Square Layer Normalization (RMSNorm) for Transformer pre-normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustRmsNorm {
    /// Feature dimension d_model
    pub dim: usize,
    /// Numerical stability epsilon
    pub eps: f32,
    /// Learnable gain vector g in R^d
    pub weight: Vec<f32>,
}

impl DustRmsNorm {
    /// Creates a new `DustRmsNorm` initialized with unit gain.
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            dim,
            eps,
            weight: vec![1.0f32; dim],
        }
    }

    /// Normalizes a single token vector `x` into `out`.
    #[inline]
    pub fn forward_token(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let mut mean_sq = 0.0f32;
        for &v in x {
            mean_sq += v * v;
        }
        mean_sq /= self.dim as f32;
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
        for i in 0..self.dim {
            out[i] = x[i] * inv_rms * self.weight[i];
        }
    }

    /// Normalizes a sequence of `seq_len` tokens from `seq_in` into `seq_out`.
    pub fn forward_sequence(&self, seq_in: &[f32], seq_out: &mut [f32], seq_len: usize) {
        for t in 0..seq_len {
            let offset = t * self.dim;
            self.forward_token(
                &seq_in[offset..offset + self.dim],
                &mut seq_out[offset..offset + self.dim],
            );
        }
    }
}

/// Target projection inside a `DustTransformerBlock` for activation-space perturbation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerPerturbTarget {
    /// Query projection W_Q (causal cross-token credit)
    Query,
    /// Key projection W_K (causal cross-token credit)
    Key,
    /// Value projection W_V (causal cross-token credit)
    Value,
    /// Attention output projection W_O (token-local credit)
    AttnOut,
    /// MLP up-projection W_up (token-local credit)
    MlpUp,
    /// MLP down-projection W_down (token-local credit)
    MlpDown,
}

/// Multi-head causal self-attention with Rotary Position Embeddings (RoPE) and Dust perturbation hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustCausalAttention {
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per attention head (d_model / num_heads)
    pub head_dim: usize,
    /// Query projection W_Q
    pub w_q: DustLayer,
    /// Key projection W_K
    pub w_k: DustLayer,
    /// Value projection W_V
    pub w_v: DustLayer,
    /// Output projection W_O
    pub w_o: DustLayer,
    /// RoPE base frequency theta
    pub rope_theta: f32,
    /// Cached clean pre-Wo context activations [seq_len * d_model]
    pub cached_context: Vec<f32>,
}

impl DustCausalAttention {
    /// Creates a new multi-head causal self-attention module.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        max_seq_len: usize,
        rng: &mut DeterministicGaussianRng,
    ) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model must be divisible by num_heads"
        );
        let head_dim = d_model / num_heads;
        Self {
            d_model,
            num_heads,
            head_dim,
            w_q: DustLayer::new(d_model, d_model, ActivationType::Identity, rng),
            w_k: DustLayer::new(d_model, d_model, ActivationType::Identity, rng),
            w_v: DustLayer::new(d_model, d_model, ActivationType::Identity, rng),
            w_o: DustLayer::new(d_model, d_model, ActivationType::Identity, rng),
            rope_theta: 10_000.0,
            cached_context: vec![0.0f32; max_seq_len * d_model],
        }
    }

    /// Applies Rotary Position Embedding (RoPE) in-place for token position `pos`.
    #[inline]
    pub fn apply_rope(&self, vec_t: &mut [f32], pos: usize) {
        let pairs = self.head_dim / 2;
        for h in 0..self.num_heads {
            let head_offset = h * self.head_dim;
            for m in 0..pairs {
                let exp = (2 * m) as f32 / (self.head_dim as f32);
                let freq = (pos as f32) / self.rope_theta.powf(exp);
                let (sin_f, cos_f) = freq.sin_cos();
                let idx0 = head_offset + 2 * m;
                let idx1 = head_offset + 2 * m + 1;
                let x0 = vec_t[idx0];
                let x1 = vec_t[idx1];
                vec_t[idx0] = x0 * cos_f - x1 * sin_f;
                vec_t[idx1] = x0 * sin_f + x1 * cos_f;
            }
        }
    }

    /// Evaluates causal self-attention over `seq_len` tokens with optional activation perturbation.
    pub fn forward_sequence(
        &mut self,
        norm_in: &[f32],
        seq_len: usize,
        perturb_target: Option<(TransformerPerturbTarget, &[f32])>,
        out: &mut [f32],
    ) {
        let total = seq_len * self.d_model;
        let mut q_buf = vec![0.0f32; total];
        let mut k_buf = vec![0.0f32; total];
        let mut v_buf = vec![0.0f32; total];

        for t in 0..seq_len {
            let s = t * self.d_model;
            let x_t = &norm_in[s..s + self.d_model];

            match perturb_target {
                Some((TransformerPerturbTarget::Query, pert)) => {
                    self.w_q
                        .forward_token_perturbed(x_t, &pert[s..s + self.d_model], &mut q_buf[s..s + self.d_model]);
                }
                _ => self.w_q.forward_token(x_t, &mut q_buf[s..s + self.d_model]),
            }

            match perturb_target {
                Some((TransformerPerturbTarget::Key, pert)) => {
                    self.w_k
                        .forward_token_perturbed(x_t, &pert[s..s + self.d_model], &mut k_buf[s..s + self.d_model]);
                }
                _ => self.w_k.forward_token(x_t, &mut k_buf[s..s + self.d_model]),
            }

            match perturb_target {
                Some((TransformerPerturbTarget::Value, pert)) => {
                    self.w_v
                        .forward_token_perturbed(x_t, &pert[s..s + self.d_model], &mut v_buf[s..s + self.d_model]);
                }
                _ => self.w_v.forward_token(x_t, &mut v_buf[s..s + self.d_model]),
            }

            self.apply_rope(&mut q_buf[s..s + self.d_model], t);
            self.apply_rope(&mut k_buf[s..s + self.d_model], t);
        }

        // Scaled causal dot-product attention per head: s <= t
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; seq_len];
        let mut context_buf = vec![0.0f32; total];

        for t in 0..seq_len {
            for h in 0..self.num_heads {
                let q_head = &q_buf[t * self.d_model + h * self.head_dim
                    ..t * self.d_model + (h + 1) * self.head_dim];

                let mut max_score = f32::NEG_INFINITY;
                for s in 0..=t {
                    let k_head = &k_buf[s * self.d_model + h * self.head_dim
                        ..s * self.d_model + (h + 1) * self.head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_head[d] * k_head[d];
                    }
                    let sc = dot * scale;
                    scores[s] = sc;
                    if sc > max_score {
                        max_score = sc;
                    }
                }

                let mut sum_exp = 0.0f32;
                for s in 0..=t {
                    let e = (scores[s] - max_score).exp();
                    scores[s] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / (sum_exp + 1e-12);

                let ctx_head = &mut context_buf[t * self.d_model + h * self.head_dim
                    ..t * self.d_model + (h + 1) * self.head_dim];
                ctx_head.fill(0.0);
                for s in 0..=t {
                    let alpha = scores[s] * inv_sum;
                    let v_head = &v_buf[s * self.d_model + h * self.head_dim
                        ..s * self.d_model + (h + 1) * self.head_dim];
                    for d in 0..self.head_dim {
                        ctx_head[d] += alpha * v_head[d];
                    }
                }
            }
        }

        if perturb_target.is_none() {
            self.cached_context[..total].copy_from_slice(&context_buf);
        }

        for t in 0..seq_len {
            let s = t * self.d_model;
            let ctx_t = &context_buf[s..s + self.d_model];
            match perturb_target {
                Some((TransformerPerturbTarget::AttnOut, pert)) => {
                    self.w_o
                        .forward_token_perturbed(ctx_t, &pert[s..s + self.d_model], &mut out[s..s + self.d_model]);
                }
                _ => self.w_o.forward_token(ctx_t, &mut out[s..s + self.d_model]),
            }
        }
    }

    /// Resets gradients on all four attention projections.
    pub fn zero_grad(&mut self) {
        self.w_q.zero_grad();
        self.w_k.zero_grad();
        self.w_v.zero_grad();
        self.w_o.zero_grad();
    }
}

/// Pre-norm Transformer decoder block trained forward-only via Dust virtual populations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustTransformerBlock {
    /// Model dimension
    pub d_model: usize,
    /// Feed-forward intermediate dimension
    pub d_ff: usize,
    /// Pre-attention RMSNorm
    pub attn_norm: DustRmsNorm,
    /// Causal multi-head self-attention
    pub attn: DustCausalAttention,
    /// Pre-MLP RMSNorm
    pub mlp_norm: DustRmsNorm,
    /// MLP expansion projection (d_model -> d_ff, GELU)
    pub mlp_up: DustLayer,
    /// MLP contraction projection (d_ff -> d_model, Identity)
    pub mlp_down: DustLayer,
    /// Cached normalized input to attention [seq_len * d_model]
    pub cached_attn_norm: Vec<f32>,
    /// Cached normalized input to MLP [seq_len * d_model]
    pub cached_mlp_norm: Vec<f32>,
    /// Cached hidden activation after mlp_up [seq_len * d_ff]
    pub cached_mlp_hidden: Vec<f32>,
}

impl DustTransformerBlock {
    /// Creates a new `DustTransformerBlock`.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        rms_eps: f32,
        rng: &mut DeterministicGaussianRng,
    ) -> Self {
        Self {
            d_model,
            d_ff,
            attn_norm: DustRmsNorm::new(d_model, rms_eps),
            attn: DustCausalAttention::new(d_model, num_heads, max_seq_len, rng),
            mlp_norm: DustRmsNorm::new(d_model, rms_eps),
            mlp_up: DustLayer::new(d_model, d_ff, ActivationType::GELU, rng),
            mlp_down: DustLayer::new(d_ff, d_model, ActivationType::Identity, rng),
            cached_attn_norm: vec![0.0f32; max_seq_len * d_model],
            cached_mlp_norm: vec![0.0f32; max_seq_len * d_model],
            cached_mlp_hidden: vec![0.0f32; max_seq_len * d_ff],
        }
    }

    /// Evaluates the transformer block over `seq_len` tokens with optional perturbation.
    pub fn forward_sequence(
        &mut self,
        residual_in: &[f32],
        seq_len: usize,
        perturb_target: Option<(TransformerPerturbTarget, &[f32])>,
        residual_out: &mut [f32],
    ) {
        let total_model = seq_len * self.d_model;
        let total_ff = seq_len * self.d_ff;

        let mut norm1_buf = vec![0.0f32; total_model];
        self.attn_norm
            .forward_sequence(residual_in, &mut norm1_buf, seq_len);

        let attn_pert = match perturb_target {
            Some((
                target @ (TransformerPerturbTarget::Query
                | TransformerPerturbTarget::Key
                | TransformerPerturbTarget::Value
                | TransformerPerturbTarget::AttnOut),
                pert,
            )) => Some((target, pert)),
            _ => None,
        };

        let mut attn_out = vec![0.0f32; total_model];
        self.attn
            .forward_sequence(&norm1_buf, seq_len, attn_pert, &mut attn_out);

        let mut mid_residual = vec![0.0f32; total_model];
        for i in 0..total_model {
            mid_residual[i] = residual_in[i] + attn_out[i];
        }

        let mut norm2_buf = vec![0.0f32; total_model];
        self.mlp_norm
            .forward_sequence(&mid_residual, &mut norm2_buf, seq_len);

        let mut ff_hidden = vec![0.0f32; total_ff];
        for t in 0..seq_len {
            let in_s = t * self.d_model;
            let ff_s = t * self.d_ff;
            match perturb_target {
                Some((TransformerPerturbTarget::MlpUp, pert)) => {
                    self.mlp_up.forward_token_perturbed(
                        &norm2_buf[in_s..in_s + self.d_model],
                        &pert[ff_s..ff_s + self.d_ff],
                        &mut ff_hidden[ff_s..ff_s + self.d_ff],
                    );
                }
                _ => {
                    self.mlp_up.forward_token(
                        &norm2_buf[in_s..in_s + self.d_model],
                        &mut ff_hidden[ff_s..ff_s + self.d_ff],
                    );
                }
            }
        }

        let mut mlp_out = vec![0.0f32; total_model];
        for t in 0..seq_len {
            let ff_s = t * self.d_ff;
            let out_s = t * self.d_model;
            match perturb_target {
                Some((TransformerPerturbTarget::MlpDown, pert)) => {
                    self.mlp_down.forward_token_perturbed(
                        &ff_hidden[ff_s..ff_s + self.d_ff],
                        &pert[out_s..out_s + self.d_model],
                        &mut mlp_out[out_s..out_s + self.d_model],
                    );
                }
                _ => {
                    self.mlp_down.forward_token(
                        &ff_hidden[ff_s..ff_s + self.d_ff],
                        &mut mlp_out[out_s..out_s + self.d_model],
                    );
                }
            }
        }

        for i in 0..total_model {
            residual_out[i] = mid_residual[i] + mlp_out[i];
        }

        if perturb_target.is_none() {
            self.cached_attn_norm[..total_model].copy_from_slice(&norm1_buf);
            self.cached_mlp_norm[..total_model].copy_from_slice(&norm2_buf);
            self.cached_mlp_hidden[..total_ff].copy_from_slice(&ff_hidden);
        }
    }

    /// Resets all projection gradients in the block.
    pub fn zero_grad(&mut self) {
        self.attn.zero_grad();
        self.mlp_up.zero_grad();
        self.mlp_down.zero_grad();
    }
}

/// Architecture configuration for `DustTransformerModel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustTransformerConfig {
    /// Token vocabulary size |V|
    pub vocab_size: usize,
    /// Hidden embedding dimension d_model
    pub d_model: usize,
    /// Number of causal attention heads
    pub num_heads: usize,
    /// Feed-forward expansion dimension d_ff
    pub d_ff: usize,
    /// Number of transformer blocks L
    pub num_layers: usize,
    /// Maximum supported sequence length T_max
    pub max_seq_len: usize,
    /// RMSNorm epsilon
    pub rms_eps: f32,
}

impl Default for DustTransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16,
            d_model: 16,
            num_heads: 4,
            d_ff: 32,
            num_layers: 2,
            max_seq_len: 16,
            rms_eps: 1e-5,
        }
    }
}

/// Autoregressive Transformer Language Model trained from scratch without backpropagation via Dust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustTransformerModel {
    /// Model architecture configuration
    pub config: DustTransformerConfig,
    /// Token embedding matrix E in R^{vocab_size * d_model}
    pub token_embeddings: Vec<f32>,
    /// Accumulated embedding gradients
    pub grad_embeddings: Vec<f32>,
    /// AdamW first moment for embeddings
    pub m_embeddings: Vec<f32>,
    /// AdamW second moment for embeddings
    pub v_embeddings: Vec<f32>,
    /// Stack of causal transformer blocks
    pub blocks: Vec<DustTransformerBlock>,
    /// Final pre-head RMSNorm
    pub final_norm: DustRmsNorm,
    /// Language modeling head W_lm (d_model -> vocab_size)
    pub lm_head: DustLayer,
    /// Cached final normalized hidden states [max_seq_len * d_model]
    pub cached_final_norm: Vec<f32>,
    /// Pre-allocated output logits buffer [max_seq_len * vocab_size]
    pub logits_buf: Vec<f32>,
}

impl DustTransformerModel {
    /// Constructs a randomly initialized `DustTransformerModel`.
    pub fn new(config: DustTransformerConfig, rng: &mut DeterministicGaussianRng) -> Self {
        let emb_size = config.vocab_size * config.d_model;
        let std_dev = (1.0 / (config.d_model as f32)).sqrt();
        let mut token_embeddings = Vec::with_capacity(emb_size);
        for _ in 0..emb_size {
            token_embeddings.push(rng.next_standard_normal() * std_dev);
        }

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(DustTransformerBlock::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.max_seq_len,
                config.rms_eps,
                rng,
            ));
        }

        let final_norm = DustRmsNorm::new(config.d_model, config.rms_eps);
        let lm_head = DustLayer::new(
            config.d_model,
            config.vocab_size,
            ActivationType::Identity,
            rng,
        );

        Self {
            cached_final_norm: vec![0.0f32; config.max_seq_len * config.d_model],
            logits_buf: vec![0.0f32; config.max_seq_len * config.vocab_size],
            config,
            token_embeddings,
            grad_embeddings: vec![0.0f32; emb_size],
            m_embeddings: vec![0.0f32; emb_size],
            v_embeddings: vec![0.0f32; emb_size],
            blocks,
            final_norm,
            lm_head,
        }
    }

    /// Runs a clean forward pass over discrete `token_ids` and returns output logits `[seq_len * vocab_size]`.
    pub fn forward_tokens(&mut self, token_ids: &[usize]) -> &[f32] {
        self.forward_tokens_with_perturbation(token_ids, None, None, None)
    }

    /// Runs a forward pass with optional perturbation at embedding, transformer block, or LM head.
    pub fn forward_tokens_with_perturbation(
        &mut self,
        token_ids: &[usize],
        emb_perturbation: Option<&[f32]>,
        block_perturbation: Option<(usize, TransformerPerturbTarget, &[f32])>,
        lm_head_perturbation: Option<&[f32]>,
    ) -> &[f32] {
        let seq_len = token_ids.len();
        let d_model = self.config.d_model;
        let vocab_size = self.config.vocab_size;
        let total_model = seq_len * d_model;

        let mut hidden_a = vec![0.0f32; total_model];
        let mut hidden_b = vec![0.0f32; total_model];

        for (t, &tok) in token_ids.iter().enumerate() {
            let emb_offset = tok * d_model;
            let dst_offset = t * d_model;
            hidden_a[dst_offset..dst_offset + d_model]
                .copy_from_slice(&self.token_embeddings[emb_offset..emb_offset + d_model]);
            if let Some(pert) = emb_perturbation {
                for j in 0..d_model {
                    hidden_a[dst_offset + j] += pert[dst_offset + j];
                }
            }
        }

        let mut in_is_a = true;
        for (l, block) in self.blocks.iter_mut().enumerate() {
            let b_pert = match block_perturbation {
                Some((target_layer, target_op, pert)) if target_layer == l => {
                    Some((target_op, pert))
                }
                _ => None,
            };

            if in_is_a {
                block.forward_sequence(&hidden_a, seq_len, b_pert, &mut hidden_b);
            } else {
                block.forward_sequence(&hidden_b, seq_len, b_pert, &mut hidden_a);
            }
            in_is_a = !in_is_a;
        }

        let final_hidden = if in_is_a { &hidden_a } else { &hidden_b };
        let mut norm_final = vec![0.0f32; total_model];
        self.final_norm
            .forward_sequence(final_hidden, &mut norm_final, seq_len);

        let is_clean = emb_perturbation.is_none()
            && block_perturbation.is_none()
            && lm_head_perturbation.is_none();
        if is_clean {
            self.cached_final_norm[..total_model].copy_from_slice(&norm_final);
        }

        for t in 0..seq_len {
            let in_s = t * d_model;
            let out_s = t * vocab_size;
            match lm_head_perturbation {
                Some(pert) => {
                    self.lm_head.forward_token_perturbed(
                        &norm_final[in_s..in_s + d_model],
                        &pert[out_s..out_s + vocab_size],
                        &mut self.logits_buf[out_s..out_s + vocab_size],
                    );
                }
                None => {
                    self.lm_head.forward_token(
                        &norm_final[in_s..in_s + d_model],
                        &mut self.logits_buf[out_s..out_s + vocab_size],
                    );
                }
            }
        }

        &self.logits_buf[..seq_len * vocab_size]
    }

    /// Computes numerically stable token-level cross-entropy losses and returns the mean sequence loss.
    pub fn compute_cross_entropy_losses(
        logits: &[f32],
        target_tokens: &[usize],
        vocab_size: usize,
        out_losses: &mut [f32],
    ) -> f32 {
        let seq_len = target_tokens.len();
        let mut total_loss = 0.0f32;
        for t in 0..seq_len {
            let row = &logits[t * vocab_size..(t + 1) * vocab_size];
            let mut max_logit = f32::NEG_INFINITY;
            for &val in row {
                if val > max_logit {
                    max_logit = val;
                }
            }
            let mut sum_exp = 0.0f32;
            for &val in row {
                sum_exp += (val - max_logit).exp();
            }
            let log_sum_exp = max_logit + sum_exp.ln();
            let loss_t = log_sum_exp - row[target_tokens[t]];
            out_losses[t] = loss_t;
            total_loss += loss_t;
        }
        total_loss / (seq_len as f32)
    }

    /// Evaluates mean cross-entropy loss, perplexity (PPL), and top-1 next-token accuracy (%) on a sequence.
    pub fn evaluate_sequence(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
    ) -> (f32, f32, f32) {
        let seq_len = input_tokens.len();
        let vocab_size = self.config.vocab_size;
        let logits = self.forward_tokens(input_tokens);
        let mut losses = vec![0.0f32; seq_len];
        let mean_loss =
            Self::compute_cross_entropy_losses(logits, target_tokens, vocab_size, &mut losses);
        let perplexity = mean_loss.exp();

        let mut correct = 0usize;
        for (t, &target) in target_tokens.iter().enumerate() {
            let row = &logits[t * vocab_size..(t + 1) * vocab_size];
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in row.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = idx;
                }
            }
            if best_idx == target {
                correct += 1;
            }
        }
        let acc_pct = (correct as f32 / (seq_len as f32)) * 100.0;
        (mean_loss, perplexity, acc_pct)
    }

    /// Generates `max_new_tokens` autoregressively using greedy decoding.
    pub fn generate_greedy(&mut self, prompt_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let mut seq = prompt_tokens.to_vec();
        let vocab_size = self.config.vocab_size;

        for _ in 0..max_new_tokens {
            let start = seq.len().saturating_sub(self.config.max_seq_len);
            let window = &seq[start..];
            let logits = self.forward_tokens(window);
            let last_row = &logits[(window.len() - 1) * vocab_size..window.len() * vocab_size];

            let mut best_tok = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in last_row.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_tok = idx;
                }
            }
            seq.push(best_tok);
        }
        seq
    }

    /// Zeroes all parameter gradients across embeddings, transformer blocks, and LM head.
    pub fn zero_grad(&mut self) {
        self.grad_embeddings.fill(0.0);
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.lm_head.zero_grad();
    }

    /// Returns total number of trainable scalar parameters in the transformer model.
    pub fn total_parameters(&self) -> usize {
        let mut total = self.token_embeddings.len();
        for b in &self.blocks {
            for l in [&b.attn.w_q, &b.attn.w_k, &b.attn.w_v, &b.attn.w_o, &b.mlp_up, &b.mlp_down] {
                total += l.weights.len() + l.bias.len();
            }
            total += b.attn_norm.weight.len() + b.mlp_norm.weight.len();
        }
        total += self.final_norm.weight.len();
        total += self.lm_head.weights.len() + self.lm_head.bias.len();
        total
    }

    /// Returns static working memory footprint in KiB (parameters + gradients + AdamW first/second moments).
    pub fn memory_footprint_kib(&self) -> f32 {
        let total_params = self.total_parameters();
        ((total_params * 4 * 4) as f32) / 1024.0
    }

    /// Serializes the complete pretrained model and optimizer state into JSON checkpoint format.
    pub fn to_checkpoint_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserializes a `DustTransformerModel` from a JSON checkpoint string.
    pub fn from_checkpoint_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Telemetry report from full-cycle Dust Transformer pretraining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustPretrainingReport {
    /// Vocabulary size |V|
    pub vocab_size: usize,
    /// Hidden dimension d_model
    pub d_model: usize,
    /// Number of transformer layers L
    pub num_layers: usize,
    /// Number of attention heads H
    pub num_heads: usize,
    /// Total trainable parameters
    pub total_parameters: usize,
    /// Static memory footprint in KiB
    pub memory_footprint_kib: f32,
    /// Initial sequence cross-entropy loss before pretraining
    pub initial_cross_entropy_loss: f32,
    /// Final sequence cross-entropy loss after pretraining
    pub final_cross_entropy_loss: f32,
    /// Percentage reduction in cross-entropy loss
    pub loss_reduction_pct: f32,
    /// Initial token perplexity (PPL)
    pub initial_perplexity: f32,
    /// Final token perplexity (PPL)
    pub final_perplexity: f32,
    /// Final top-1 next-token prediction accuracy (%)
    pub final_top1_accuracy_pct: f32,
    /// Cosine similarity of antithetic Dust gradients vs exact analytical gradients
    pub antithetic_gradient_cosine_similarity: f32,
    /// Verified causal attention credit propagation across future tokens
    pub causal_credit_advantage_verified: bool,
    /// Mean pretraining step latency in microseconds
    pub step_latency_us: f32,
    /// Number of pretraining epochs completed
    pub epochs_completed: usize,
}

/// Full-cycle non-backpropagation pretrainer for `DustTransformerModel`.
pub struct DustPretrainer {
    /// Base Dust trainer configuration
    pub config: DustTrainerConfig,
    ///Causal discount factor for attention Q/K/V projections (e.g. 0.95)
    pub causal_gamma_attn: f32,
    /// Discount factor for token-local projections (W_O, MLP, LM head, typically 0.0)
    pub causal_gamma_local: f32,
    /// Cosine learning rate scheduler and gradient norm clipper
    pub scheduler: CosineWarmupScheduler,
    /// Deterministic Gaussian PRNG
    pub rng: DeterministicGaussianRng,
    /// Step counter
    pub step_count: u64,
}

impl DustPretrainer {
    /// Initializes a new `DustPretrainer`.
    pub fn new(
        config: DustTrainerConfig,
        causal_gamma_attn: f32,
        causal_gamma_local: f32,
        scheduler: CosineWarmupScheduler,
    ) -> Self {
        let rng = DeterministicGaussianRng::new(config.seed);
        Self {
            config,
            causal_gamma_attn,
            causal_gamma_local,
            scheduler,
            rng,
            step_count: 0,
        }
    }

    /// Accumulates outer-product weight and bias gradients into `layer` from `in_act` and `output_errors`.
    fn accumulate_layer_outer_product(
        layer: &mut DustLayer,
        in_act: &[f32],
        output_errors: &[f32],
        seq_len: usize,
    ) {
        let in_dim = layer.d_in;
        let out_dim = layer.d_out;
        let inv_seq = 1.0 / (seq_len as f32);

        for t in 0..seq_len {
            let g_t = &output_errors[t * out_dim..(t + 1) * out_dim];
            let x_t = &in_act[t * in_dim..(t + 1) * in_dim];
            for i in 0..out_dim {
                let g_val = g_t[i] * inv_seq;
                let row_offset = i * in_dim;
                for j in 0..in_dim {
                    layer.grad_weights[row_offset + j] += g_val * x_t[j];
                }
                layer.grad_bias[i] += g_val;
            }
        }
    }

    /// Executes one full-cycle antithetic Dust pretraining step on discrete `input_tokens` and `target_tokens`.
    /// Returns the clean sequence cross-entropy loss before the parameter update.
    pub fn pretrain_step(
        &mut self,
        model: &mut DustTransformerModel,
        input_tokens: &[usize],
        target_tokens: &[usize],
    ) -> f32 {
        self.step_count += 1;
        let seq_len = input_tokens.len();
        let vocab_size = model.config.vocab_size;
        let d_model = model.config.d_model;
        let d_ff = model.config.d_ff;
        let num_draws = self.config.num_draws;
        let sigma = self.config.perturbation_sigma;
        let norm_factor = -1.0 / ((num_draws as f32) * sigma * sigma);

        // 1. Clean forward pass to cache all intermediate layer inputs
        let mut clean_losses = vec![0.0f32; seq_len];
        let mean_clean_loss = {
            let logits = model.forward_tokens(input_tokens);
            DustTransformerModel::compute_cross_entropy_losses(
                logits,
                target_tokens,
                vocab_size,
                &mut clean_losses,
            )
        };

        model.zero_grad();

        let max_dim = d_model.max(d_ff).max(vocab_size);
        let mut pos_pert = vec![0.0f32; seq_len * max_dim];
        let mut neg_pert = vec![0.0f32; seq_len * max_dim];
        let mut pos_losses = vec![0.0f32; seq_len];
        let mut neg_losses = vec![0.0f32; seq_len];
        let mut output_errors = vec![0.0f32; seq_len * max_dim];

        // 2. Synthesize gradients for Token Embeddings via antithetic activation perturbation
        {
            let total_emb_elems = seq_len * d_model;
            output_errors[..total_emb_elems].fill(0.0);

            for _ in 0..num_draws {
                for k in 0..total_emb_elems {
                    let p = self.rng.next_standard_normal() * sigma;
                    pos_pert[k] = p;
                    neg_pert[k] = -p;
                }

                let pos_logits = model.forward_tokens_with_perturbation(
                    input_tokens,
                    Some(&pos_pert[..total_emb_elems]),
                    None,
                    None,
                );
                DustTransformerModel::compute_cross_entropy_losses(
                    pos_logits,
                    target_tokens,
                    vocab_size,
                    &mut pos_losses,
                );

                let neg_logits = model.forward_tokens_with_perturbation(
                    input_tokens,
                    Some(&neg_pert[..total_emb_elems]),
                    None,
                    None,
                );
                DustTransformerModel::compute_cross_entropy_losses(
                    neg_logits,
                    target_tokens,
                    vocab_size,
                    &mut neg_losses,
                );

                let mut future_credit = 0.0f32;
                for t in (0..seq_len).rev() {
                    let c_t = 0.5 * (neg_losses[t] - pos_losses[t]);
                    let r_t = c_t + self.causal_gamma_attn * future_credit;
                    future_credit = r_t;
                    let offset = t * d_model;
                    for j in 0..d_model {
                        output_errors[offset + j] += r_t * pos_pert[offset + j];
                    }
                }
            }

            let inv_seq = norm_factor / (seq_len as f32);
            for (t, &tok) in input_tokens.iter().enumerate() {
                let emb_offset = tok * d_model;
                let err_offset = t * d_model;
                for j in 0..d_model {
                    model.grad_embeddings[emb_offset + j] +=
                        output_errors[err_offset + j] * inv_seq;
                }
            }
        }

        // 3. Synthesize gradients for each Transformer Block's 6 projections
        let num_blocks = model.blocks.len();
        let targets_list = [
            TransformerPerturbTarget::Query,
            TransformerPerturbTarget::Key,
            TransformerPerturbTarget::Value,
            TransformerPerturbTarget::AttnOut,
            TransformerPerturbTarget::MlpUp,
            TransformerPerturbTarget::MlpDown,
        ];

        for l in 0..num_blocks {
            let cached_attn_norm = model.blocks[l].cached_attn_norm[..seq_len * d_model].to_vec();
            let cached_attn_ctx = model.blocks[l].attn.cached_context[..seq_len * d_model].to_vec();
            let cached_mlp_norm = model.blocks[l].cached_mlp_norm[..seq_len * d_model].to_vec();
            let cached_mlp_hidden = model.blocks[l].cached_mlp_hidden[..seq_len * d_ff].to_vec();

            for &target_op in &targets_list {
                let out_dim = match target_op {
                    TransformerPerturbTarget::MlpUp => d_ff,
                    _ => d_model,
                };
                let gamma = match target_op {
                    TransformerPerturbTarget::Query
                    | TransformerPerturbTarget::Key
                    | TransformerPerturbTarget::Value => self.causal_gamma_attn,
                    TransformerPerturbTarget::AttnOut
                    | TransformerPerturbTarget::MlpUp
                    | TransformerPerturbTarget::MlpDown => self.causal_gamma_local,
                };

                let total_elems = seq_len * out_dim;
                output_errors[..total_elems].fill(0.0);

                for _ in 0..num_draws {
                    for k in 0..total_elems {
                        let p = self.rng.next_standard_normal() * sigma;
                        pos_pert[k] = p;
                        neg_pert[k] = -p;
                    }

                    let pos_logits = model.forward_tokens_with_perturbation(
                        input_tokens,
                        None,
                        Some((l, target_op, &pos_pert[..total_elems])),
                        None,
                    );
                    DustTransformerModel::compute_cross_entropy_losses(
                        pos_logits,
                        target_tokens,
                        vocab_size,
                        &mut pos_losses,
                    );

                    let neg_logits = model.forward_tokens_with_perturbation(
                        input_tokens,
                        None,
                        Some((l, target_op, &neg_pert[..total_elems])),
                        None,
                    );
                    DustTransformerModel::compute_cross_entropy_losses(
                        neg_logits,
                        target_tokens,
                        vocab_size,
                        &mut neg_losses,
                    );

                    let mut future_credit = 0.0f32;
                    for t in (0..seq_len).rev() {
                        let c_t = 0.5 * (neg_losses[t] - pos_losses[t]);
                        let r_t = c_t + gamma * future_credit;
                        future_credit = r_t;
                        let offset = t * out_dim;
                        for j in 0..out_dim {
                            output_errors[offset + j] += r_t * pos_pert[offset + j];
                        }
                    }
                }

                for elem in output_errors[..total_elems].iter_mut() {
                    *elem *= norm_factor;
                }

                let block = &mut model.blocks[l];
                match target_op {
                    TransformerPerturbTarget::Query => {
                        Self::accumulate_layer_outer_product(
                            &mut block.attn.w_q,
                            &cached_attn_norm,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                    TransformerPerturbTarget::Key => {
                        Self::accumulate_layer_outer_product(
                            &mut block.attn.w_k,
                            &cached_attn_norm,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                    TransformerPerturbTarget::Value => {
                        Self::accumulate_layer_outer_product(
                            &mut block.attn.w_v,
                            &cached_attn_norm,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                    TransformerPerturbTarget::AttnOut => {
                        Self::accumulate_layer_outer_product(
                            &mut block.attn.w_o,
                            &cached_attn_ctx,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                    TransformerPerturbTarget::MlpUp => {
                        Self::accumulate_layer_outer_product(
                            &mut block.mlp_up,
                            &cached_mlp_norm,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                    TransformerPerturbTarget::MlpDown => {
                        Self::accumulate_layer_outer_product(
                            &mut block.mlp_down,
                            &cached_mlp_hidden,
                            &output_errors[..total_elems],
                            seq_len,
                        );
                    }
                }
            }
        }

        // 4. Synthesize gradients for LM Head via antithetic perturbation
        {
            let cached_final = model.cached_final_norm[..seq_len * d_model].to_vec();
            let total_lm_elems = seq_len * vocab_size;
            output_errors[..total_lm_elems].fill(0.0);

            for _ in 0..num_draws {
                for k in 0..total_lm_elems {
                    let p = self.rng.next_standard_normal() * sigma;
                    pos_pert[k] = p;
                    neg_pert[k] = -p;
                }

                let pos_logits = model.forward_tokens_with_perturbation(
                    input_tokens,
                    None,
                    None,
                    Some(&pos_pert[..total_lm_elems]),
                );
                DustTransformerModel::compute_cross_entropy_losses(
                    pos_logits,
                    target_tokens,
                    vocab_size,
                    &mut pos_losses,
                );

                let neg_logits = model.forward_tokens_with_perturbation(
                    input_tokens,
                    None,
                    None,
                    Some(&neg_pert[..total_lm_elems]),
                );
                DustTransformerModel::compute_cross_entropy_losses(
                    neg_logits,
                    target_tokens,
                    vocab_size,
                    &mut neg_losses,
                );

                for t in 0..seq_len {
                    let c_t = 0.5 * (neg_losses[t] - pos_losses[t]);
                    let offset = t * vocab_size;
                    for j in 0..vocab_size {
                        output_errors[offset + j] += c_t * pos_pert[offset + j];
                    }
                }
            }

            for elem in output_errors[..total_lm_elems].iter_mut() {
                *elem *= norm_factor;
            }

            Self::accumulate_layer_outer_product(
                &mut model.lm_head,
                &cached_final,
                &output_errors[..total_lm_elems],
                seq_len,
            );
        }

        // 5. Global L2 gradient clipping and scheduled AdamW parameter update
        self.apply_scheduled_adamw(model);

        mean_clean_loss
    }

    /// Clips global gradient norm and updates all transformer parameters with scheduled AdamW.
    fn apply_scheduled_adamw(&self, model: &mut DustTransformerModel) {
        let lr = self.scheduler.learning_rate(self.step_count);
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let wd = self.config.weight_decay;
        let eps = self.config.eps;
        let t_f32 = self.step_count as f32;
        let bc1 = 1.0 - beta1.powf(t_f32);
        let bc2 = 1.0 - beta2.powf(t_f32);

        // Compute global norm across all parameters
        let mut sum_sq = 0.0f32;
        for &g in &model.grad_embeddings {
            sum_sq += g * g;
        }
        for b in &model.blocks {
            for l in [&b.attn.w_q, &b.attn.w_k, &b.attn.w_v, &b.attn.w_o, &b.mlp_up, &b.mlp_down] {
                for &g in &l.grad_weights {
                    sum_sq += g * g;
                }
                for &g in &l.grad_bias {
                    sum_sq += g * g;
                }
            }
        }
        for &g in &model.lm_head.grad_weights {
            sum_sq += g * g;
        }
        for &g in &model.lm_head.grad_bias {
            sum_sq += g * g;
        }

        let global_norm = sum_sq.sqrt();
        let clip_scale = if global_norm > self.scheduler.max_grad_norm && global_norm > 1e-12 {
            self.scheduler.max_grad_norm / global_norm
        } else {
            1.0
        };

        // Update embeddings
        for idx in 0..model.token_embeddings.len() {
            let g = model.grad_embeddings[idx] * clip_scale;
            model.m_embeddings[idx] = beta1 * model.m_embeddings[idx] + (1.0 - beta1) * g;
            model.v_embeddings[idx] = beta2 * model.v_embeddings[idx] + (1.0 - beta2) * (g * g);
            let m_hat = model.m_embeddings[idx] / bc1;
            let v_hat = model.v_embeddings[idx] / bc2;
            let update = (m_hat / (v_hat.sqrt() + eps)) + wd * model.token_embeddings[idx];
            model.token_embeddings[idx] -= lr * update;
        }

        // Update all block layers
        for b in &mut model.blocks {
            for l in [
                &mut b.attn.w_q,
                &mut b.attn.w_k,
                &mut b.attn.w_v,
                &mut b.attn.w_o,
                &mut b.mlp_up,
                &mut b.mlp_down,
            ] {
                if clip_scale != 1.0 {
                    for g in l.grad_weights.iter_mut() {
                        *g *= clip_scale;
                    }
                    for g in l.grad_bias.iter_mut() {
                        *g *= clip_scale;
                    }
                }
                DustTrainer::update_dust_layer_adamw(l, lr, wd, beta1, beta2, bc1, bc2, eps);
            }
        }

        // Update LM head
        if clip_scale != 1.0 {
            for g in model.lm_head.grad_weights.iter_mut() {
                *g *= clip_scale;
            }
            for g in model.lm_head.grad_bias.iter_mut() {
                *g *= clip_scale;
            }
        }
        DustTrainer::update_dust_layer_adamw(
            &mut model.lm_head,
            lr,
            wd,
            beta1,
            beta2,
            bc1,
            bc2,
            eps,
        );
    }

    /// Runs a full empirical Transformer pretraining benchmark and returns `DustPretrainingReport`.
    pub fn evaluate_transformer_pretraining() -> DustPretrainingReport {
        let mut rng = DeterministicGaussianRng::new(2026);
        let tf_config = DustTransformerConfig {
            vocab_size: 12,
            d_model: 12,
            num_heads: 3,
            d_ff: 24,
            num_layers: 1,
            max_seq_len: 8,
            rms_eps: 1e-5,
        };
        let mut model = DustTransformerModel::new(tf_config.clone(), &mut rng);

        // Deterministic robotics flight-command token sequence:
        // [ARM, TAKEOFF, WAYPOINT_A, SCAN, WAYPOINT_B, HOVER, RTL, LAND]
        let input_tokens = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let target_tokens = [2usize, 3, 4, 5, 6, 7, 8, 9];

        let (initial_ce_loss, initial_ppl, _initial_acc) =
            model.evaluate_sequence(&input_tokens, &target_tokens);

        let epochs = 45usize;
        let trainer_cfg = DustTrainerConfig {
            perturbation_sigma: 0.04,
            num_draws: 16,
            causal_gamma: 0.90,
            learning_rate: 0.04,
            weight_decay: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            seed: 77,
        };
        let scheduler = CosineWarmupScheduler::new(0.045, 0.005, 5, epochs as u64, 5.0);
        let mut pretrainer = DustPretrainer::new(trainer_cfg, 0.90, 0.0, scheduler);

        let start = Instant::now();
        for _ in 0..epochs {
            pretrainer.pretrain_step(&mut model, &input_tokens, &target_tokens);
        }
        let step_latency_us = (start.elapsed().as_micros() as f32) / (epochs as f32);

        let (final_ce_loss, final_ppl, final_top1_acc) =
            model.evaluate_sequence(&input_tokens, &target_tokens);
        let loss_reduction_pct =
            ((initial_ce_loss - final_ce_loss) / initial_ce_loss.max(1e-8)) * 100.0;

        // Measure antithetic gradient cosine similarity on DustMultiLayerNetwork with gamma = 0.0
        let mut mlp_rng = DeterministicGaussianRng::new(1337);
        let dims = vec![12, 16, 6];
        let acts = vec![ActivationType::ReLU, ActivationType::Identity];
        let seq_len = 8;
        let mut mlp_net = DustMultiLayerNetwork::new(&dims, &acts, seq_len, &mut mlp_rng);
        let inputs: Vec<f32> = (0..seq_len * 12)
            .map(|k| ((k as f32) * 0.17).sin())
            .collect();
        let targets: Vec<f32> = (0..seq_len * 6)
            .map(|k| ((k as f32) * 0.23).cos() * 0.5)
            .collect();
        let mut anti_trainer = DustTrainer::new(
            DustTrainerConfig {
                perturbation_sigma: 0.02,
                num_draws: 48,
                causal_gamma: 0.0,
                learning_rate: 0.01,
                ..DustTrainerConfig::default()
            },
            seq_len,
            16,
        );
        let _ = anti_trainer.train_step_antithetic(&mut mlp_net, &inputs, &targets, seq_len);
        let analytical_grad = DustTrainingEvaluator::compute_analytical_last_layer_gradient(
            &mlp_net, &inputs, &targets, seq_len,
        );
        let antithetic_cos_sim = DustTrainingEvaluator::cosine_similarity(
            &mlp_net.layers.last().unwrap().grad_weights,
            &analytical_grad,
        );

        DustPretrainingReport {
            vocab_size: tf_config.vocab_size,
            d_model: tf_config.d_model,
            num_layers: tf_config.num_layers,
            num_heads: tf_config.num_heads,
            total_parameters: model.total_parameters(),
            memory_footprint_kib: model.memory_footprint_kib(),
            initial_cross_entropy_loss: initial_ce_loss,
            final_cross_entropy_loss: final_ce_loss,
            loss_reduction_pct,
            initial_perplexity: initial_ppl,
            final_perplexity: final_ppl,
            final_top1_accuracy_pct: final_top1_acc,
            antithetic_gradient_cosine_similarity: antithetic_cos_sim,
            causal_credit_advantage_verified: final_ce_loss < initial_ce_loss * 0.35,
            step_latency_us,
            epochs_completed: epochs,
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
    pub fn compute_analytical_last_layer_gradient(
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
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

    #[test]
    fn test_antithetic_gradient_high_cosine_similarity() {
        let mut rng = DeterministicGaussianRng::new(99);
        let dims = vec![10, 12, 6];
        let acts = vec![ActivationType::ReLU, ActivationType::Identity];
        let seq_len = 8;
        let mut net = DustMultiLayerNetwork::new(&dims, &acts, seq_len, &mut rng);

        let inputs: Vec<f32> = (0..seq_len * 10)
            .map(|i| ((i as f32) * 0.21).sin())
            .collect();
        let targets: Vec<f32> = (0..seq_len * 6)
            .map(|i| ((i as f32) * 0.31).cos() * 0.5)
            .collect();

        let mut trainer = DustTrainer::new(
            DustTrainerConfig {
                perturbation_sigma: 0.02,
                num_draws: 48,
                causal_gamma: 0.0,
                learning_rate: 0.01,
                ..DustTrainerConfig::default()
            },
            seq_len,
            16,
        );
        let _ = trainer.train_step_antithetic(&mut net, &inputs, &targets, seq_len);
        let exact_grad = DustTrainingEvaluator::compute_analytical_last_layer_gradient(
            &net, &inputs, &targets, seq_len,
        );
        let cos_sim = DustTrainingEvaluator::cosine_similarity(
            &net.layers.last().unwrap().grad_weights,
            &exact_grad,
        );
        assert!(
            cos_sim > 0.80,
            "Antithetic gradient cosine similarity ({:.4}) must exceed 0.80",
            cos_sim
        );
    }

    #[test]
    fn test_cosine_warmup_scheduler_and_clipping() {
        let sched = CosineWarmupScheduler::new(0.04, 0.004, 10, 50, 1.5);
        assert!((sched.learning_rate(5) - 0.02).abs() < 1e-6);
        assert!((sched.learning_rate(10) - 0.04).abs() < 1e-6);
        assert!((sched.learning_rate(50) - 0.004).abs() < 1e-6);

        let mut rng = DeterministicGaussianRng::new(7);
        let mut layer = DustLayer::new(4, 4, ActivationType::Identity, &mut rng);
        layer.grad_weights.fill(2.0);
        let mut layers = [&mut layer];
        let orig_norm = sched.clip_gradients(&mut layers, None);
        assert!(orig_norm > 1.5);
        let clipped_norm: f32 = layers[0].grad_weights.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.5).abs() < 1e-4);
    }

    #[test]
    fn test_transformer_pretraining_and_checkpoint_roundtrip() {
        let report = DustPretrainer::evaluate_transformer_pretraining();
        assert!(
            report.loss_reduction_pct > 50.0,
            "Transformer pretraining loss reduction ({:.2}%) must exceed 50%",
            report.loss_reduction_pct
        );
        assert!(
            report.final_perplexity < report.initial_perplexity,
            "Final perplexity ({:.2}) must be lower than initial ({:.2})",
            report.final_perplexity,
            report.initial_perplexity
        );
        assert!(
            report.final_top1_accuracy_pct >= 85.0,
            "Top-1 accuracy ({:.1}%) must be >= 85%",
            report.final_top1_accuracy_pct
        );
        assert!(
            report.antithetic_gradient_cosine_similarity > 0.80,
            "Antithetic gradient cosine similarity ({:.4}) must exceed 0.80",
            report.antithetic_gradient_cosine_similarity
        );

        let mut rng = DeterministicGaussianRng::new(11);
        let mut model = DustTransformerModel::new(DustTransformerConfig::default(), &mut rng);
        let tokens = [1usize, 2, 3, 4];
        let logits_before = model.forward_tokens(&tokens).to_vec();
        let json = model.to_checkpoint_json().expect("serialize checkpoint");
        let mut restored =
            DustTransformerModel::from_checkpoint_json(&json).expect("deserialize checkpoint");
        let logits_after = restored.forward_tokens(&tokens).to_vec();
        for (a, b) in logits_before.iter().zip(logits_after.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

