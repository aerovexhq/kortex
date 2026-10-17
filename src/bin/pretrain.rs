//! Standalone CLI binary for full-cycle Dust non-backpropagation Transformer pretraining.
//!
//! Trains an autoregressive `DustTransformerModel` from random initialization using
//! Q Labs' Dust activation-space virtual populations with antithetic noise draws,
//! operator-aware causal attention credit assignment, and cosine warmup AdamW.

#![deny(unsafe_code)]

use clap::Parser;
use kortex::{
    CosineWarmupScheduler, DeterministicGaussianRng, DustPretrainer, DustTrainerConfig,
    DustTransformerConfig, DustTransformerModel,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "pretrain",
    about = "Kortex Dust Non-Backpropagation Transformer Pretraining Engine"
)]
struct PretrainArgs {
    /// Vocabulary size |V|
    #[arg(long, default_value_t = 16)]
    vocab_size: usize,

    /// Hidden embedding dimension d_model
    #[arg(long, default_value_t = 16)]
    d_model: usize,

    /// Number of causal attention heads
    #[arg(long, default_value_t = 4)]
    heads: usize,

    /// Feed-forward intermediate dimension d_ff
    #[arg(long, default_value_t = 32)]
    d_ff: usize,

    /// Number of Transformer decoder blocks
    #[arg(long, default_value_t = 2)]
    layers: usize,

    /// Training epochs
    #[arg(long, default_value_t = 50)]
    epochs: usize,

    /// Antithetic virtual population draws per operator per step
    #[arg(long, default_value_t = 16)]
    draws: usize,

    /// Perturbation standard deviation sigma
    #[arg(long, default_value_t = 0.04)]
    sigma: f32,

    /// Peak learning rate after warmup
    #[arg(long, default_value_t = 0.04)]
    lr: f32,

    /// Causal attention credit discount factor gamma_attn
    #[arg(long, default_value_t = 0.92)]
    gamma_attn: f32,

    /// Deterministic PRNG seed
    #[arg(long, default_value_t = 2026)]
    seed: u64,

    /// Optional path to export JSON model checkpoint after pretraining
    #[arg(long)]
    save_checkpoint: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = PretrainArgs::parse();

    let input_tokens: Vec<usize> = (1..=8).map(|t| t % args.vocab_size).collect();
    let target_tokens: Vec<usize> = (2..=9).map(|t| t % args.vocab_size).collect();
    let seq_len = input_tokens.len();

    let tf_config = DustTransformerConfig {
        vocab_size: args.vocab_size,
        d_model: args.d_model,
        num_heads: args.heads,
        d_ff: args.d_ff,
        num_layers: args.layers,
        max_seq_len: seq_len.max(16),
        rms_eps: 1e-5,
    };

    let mut init_rng = DeterministicGaussianRng::new(args.seed);
    let mut model = DustTransformerModel::new(tf_config, &mut init_rng);

    let (init_loss, init_ppl, init_acc) = model.evaluate_sequence(&input_tokens, &target_tokens);

    println!("================================================================================");
    println!("  KORTEX DUST NON-BACKPROPAGATION TRANSFORMER PRETRAINING ENGINE");
    println!("================================================================================");
    println!(
        "  Architecture : L={} Layers | d_model={} | H={} Heads | d_ff={} | |V|={}",
        args.layers, args.d_model, args.heads, args.d_ff, args.vocab_size
    );
    println!(
        "  Parameters   : {} params | Static Footprint: {:.2} KiB (Zero Autograd Tape)",
        model.total_parameters(),
        model.memory_footprint_kib()
    );
    println!(
        "  Dust Config  : K={} Antithetic Draws | sigma={:.3} | gamma_attn={:.2} | max_lr={:.4}",
        args.draws, args.sigma, args.gamma_attn, args.lr
    );
    println!("--------------------------------------------------------------------------------");
    println!(
        "  [Epoch   0/{:3}] CE Loss: {:.4} | Perplexity: {:6.2} | Top-1 Acc: {:5.1}%",
        args.epochs, init_loss, init_ppl, init_acc
    );

    let trainer_cfg = DustTrainerConfig {
        perturbation_sigma: args.sigma,
        num_draws: args.draws,
        causal_gamma: args.gamma_attn,
        learning_rate: args.lr,
        weight_decay: 0.0001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        seed: args.seed.wrapping_add(77),
    };
    let warmup_steps = (args.epochs / 8).max(1) as u64;
    let scheduler =
        CosineWarmupScheduler::new(args.lr, args.lr * 0.1, warmup_steps, args.epochs as u64, 5.0);
    let mut pretrainer = DustPretrainer::new(trainer_cfg, args.gamma_attn, 0.0, scheduler);

    let start = Instant::now();
    let log_interval = (args.epochs / 5).max(1);

    for epoch in 1..=args.epochs {
        pretrainer.pretrain_step(&mut model, &input_tokens, &target_tokens);
        if epoch % log_interval == 0 || epoch == args.epochs {
            let (loss, ppl, acc) = model.evaluate_sequence(&input_tokens, &target_tokens);
            let cur_lr = pretrainer.scheduler.learning_rate(pretrainer.step_count);
            println!(
                "  [Epoch {:3}/{:3}] CE Loss: {:.4} | Perplexity: {:6.2} | Top-1 Acc: {:5.1}% | LR: {:.5}",
                epoch, args.epochs, loss, ppl, acc, cur_lr
            );
        }
    }

    let elapsed = start.elapsed();
    let step_latency_us = (elapsed.as_micros() as f32) / (args.epochs as f32);
    let (final_loss, final_ppl, final_acc) = model.evaluate_sequence(&input_tokens, &target_tokens);
    let reduction_pct = ((init_loss - final_loss) / init_loss.max(1e-8)) * 100.0;

    let generated = model.generate_greedy(&input_tokens[..2], seq_len - 1);

    println!("--------------------------------------------------------------------------------");
    println!(
        "  Summary      : Loss {:.4} -> {:.4} (-{:.1}%) | PPL {:.2} -> {:.2} | Acc: {:.1}%",
        init_loss, final_loss, reduction_pct, init_ppl, final_ppl, final_acc
    );
    println!(
        "  Step Latency : {:.2} us/step | Greedy Rollout: {:?}",
        step_latency_us, generated
    );

    if let Some(ref ckpt_path) = args.save_checkpoint {
        let json = model.to_checkpoint_json()?;
        fs::write(ckpt_path, json)?;
        println!("  Checkpoint   : Saved to {}", ckpt_path.display());
    }
    println!("================================================================================");

    Ok(())
}
