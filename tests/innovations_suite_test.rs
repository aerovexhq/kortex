//! Comprehensive integration verification and empirical benchmark suite for Kortex
//! edge inference innovations:
//! - Distributed Edge KV Mesh & Peer Pipelining (`mesh`)
//! - Hardware Micro-Kernel Code Generation & SIMD Vector Kernels (`kernels`)
//! - Zero-Backprop On-Device Continual Adaptation — Dust / Kortex-ZeroAdapt (`zero_adapt`)
//! - Zeroth-Order Self-Speculative Decoding — ZO-SpecDec (`zo_specdec`)
//! - Kähler-Bergman Reproducing Kernel Continuous KV-Cache — KB-RKHS (`kb_rkhs`)
//! - Chronos Multi-World Seqlock Rollout Verifier & Kestrel Avionics Interconnect (`avionics_bridge`)

#![deny(unsafe_code)]

use kortex::{
    AvionicsBridgeEvaluator, DistributedMeshEvaluator, KbRkhsEvaluator, MicroKernelEvaluator,
    ZeroAdaptEvaluator, ZoSpecDecEvaluator,
};

#[test]
fn verify_all_kortex_innovation_subsystems() {
    let mesh_report = DistributedMeshEvaluator::default().evaluate();
    let kernel_report = MicroKernelEvaluator::evaluate();
    let adapt_report = ZeroAdaptEvaluator::default().evaluate();
    let specdec_report = ZoSpecDecEvaluator::evaluate();
    let kb_report = KbRkhsEvaluator::evaluate();
    let avionics_report = AvionicsBridgeEvaluator::evaluate();

    println!("\n=== KORTEX INNOVATION SUBSYSTEMS EMPIRICAL BENCHMARK REPORT ===");
    println!(
        "[Mesh]       Nodes: {} | VNodes: {} | Lookup: {:.1} ns | CDR Compression: {:.2}x | Failover: {:.1} us | Swarm Speedup: {:.2}x ({:.2} -> {:.2} tok/s)",
        mesh_report.active_swarm_nodes,
        mesh_report.total_virtual_nodes,
        mesh_report.ring_lookup_latency_ns,
        mesh_report.wire_compression_ratio,
        mesh_report.failover_recovery_us,
        mesh_report.pipeline_speedup_mult,
        mesh_report.single_node_tok_per_sec,
        mesh_report.pipelined_swarm_tok_per_sec
    );
    println!(
        "[Kernels]    Tile: {}B (align {}B) | Panel: {:.2} GFLOP/s (err {:.2e}) | E8: {:.1} ns ({:.1}% parity) | Bergman: {:.1} ns (err {:.2e})",
        kernel_report.tile_size_bytes,
        kernel_report.tile_alignment_bytes,
        kernel_report.contraction_gflops,
        kernel_report.contraction_max_rel_err,
        kernel_report.e8_quant_latency_ns,
        kernel_report.e8_valid_parity_pct,
        kernel_report.bergman_eval_latency_ns,
        kernel_report.bergman_rel_err_vs_f64
    );
    println!(
        "[ZeroAdapt]  Footprint: {:.2} KiB | Step Latency: {:.2} us | Loss: {:.4} -> {:.4} ({:.1}% red) | Grad Cos: {:.4} | Flight Reward: {:.3} -> {:.3}",
        adapt_report.adapter_memory_kib,
        adapt_report.step_latency_us,
        adapt_report.initial_loss,
        adapt_report.final_loss,
        adapt_report.loss_reduction_pct,
        adapt_report.mean_gradient_cosine_sim,
        adapt_report.initial_flight_reward,
        adapt_report.adapted_flight_reward
    );
    println!(
        "[ZO-SpecDec] Heads: {:.1} KiB | Acceptance: {:.1}% -> {:.1}% | Head MSE: {:.4} -> {:.4} | Emit: {:.2} tok/step | Speedup: {:.2}x | Calib: {:.2} us",
        specdec_report.speculative_heads_kib,
        specdec_report.initial_acceptance_rate * 100.0,
        specdec_report.calibrated_acceptance_rate * 100.0,
        specdec_report.initial_head_mse,
        specdec_report.calibrated_head_mse,
        specdec_report.expected_tokens_per_step,
        specdec_report.decoding_speedup_factor,
        specdec_report.calibration_step_latency_us
    );
    println!(
        "[KB-RKHS]    Context: {} | FP16: {:.1} MiB -> KB-RKHS: {:.4} MiB ({:.1}x) | Spectral Energy: {:.4}% | Query: {:.2} us",
        kb_report.context_length,
        kb_report.uncompressed_fp16_mib,
        kb_report.kb_rkhs_footprint_mib,
        kb_report.compression_ratio,
        kb_report.spectral_energy_preservation_pct,
        kb_report.query_latency_us
    );
    println!(
        "[Avionics]   Seqlock Read: {:.2} ns | 128-World Rollout: {:.3} ms | Safe Tokens: {}/{} | Watchdog Fallback: {} | Sonon Frames: {}",
        avionics_report.seqlock_read_latency_ns,
        avionics_report.rollout_128_worlds_latency_ms,
        avionics_report.safe_candidates_retained,
        avionics_report.total_candidates,
        avionics_report.watchdog_fallback_verified,
        avionics_report.sonon_frames_streamed
    );

    // 1. Distributed Mesh gates
    assert!(mesh_report.wire_compression_ratio > 90.0);
    assert!(mesh_report.pipeline_speedup_mult > 1.5);
    assert_eq!(mesh_report.stages.len(), 4);

    // 2. Hardware Micro-Kernels gates
    assert_eq!(kernel_report.tile_size_bytes, 64);
    assert_eq!(kernel_report.tile_alignment_bytes, 64);
    assert!(kernel_report.contraction_max_rel_err < 1e-5);
    assert!((kernel_report.e8_valid_parity_pct - 100.0).abs() < 1e-6);

    // 3. Dust / ZeroAdapt gates
    assert!(adapt_report.adapter_memory_kib < 60.0);
    assert!(adapt_report.step_latency_us < 15.0);
    assert!(adapt_report.final_loss < adapt_report.initial_loss);
    assert!(adapt_report.adapted_flight_reward > adapt_report.initial_flight_reward);

    // 4. ZO-SpecDec gates
    assert!(specdec_report.decoding_speedup_factor >= 2.45);
    assert!(specdec_report.calibrated_head_mse < specdec_report.initial_head_mse);

    // 5. KB-RKHS gates
    assert!(kb_report.kb_rkhs_footprint_mib <= 0.75);
    assert!(kb_report.compression_ratio > 520.0);
    assert!(kb_report.spectral_energy_preservation_pct >= 99.94);

    // 6. Chronos & Kestrel Avionics Bridge gates
    assert!(avionics_report.seqlock_read_latency_ns < 50.0);
    assert!(avionics_report.rollout_128_worlds_latency_ms < 30.0);
    assert_eq!(avionics_report.safe_candidates_retained, 2);
    assert!(avionics_report.watchdog_fallback_verified);
    assert_eq!(avionics_report.sonon_frames_streamed, 64);
}
