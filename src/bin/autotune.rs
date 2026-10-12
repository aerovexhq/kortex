//! Automated LLM Method Optimizer & Deep Efficiency Score Tester.
//!
//! Automatically scans `models/*.gguf`, benchmarks all candidate inference methods,
//! calculates the in-depth multi-objective efficiency score, and automatically
//! determines the winning method and architectural rationale for each model.

use std::fs;
use std::path::PathBuf;
use anyhow::Result;

use kortex::autotune::{AutoTuner, BenchmarkProfile, ModelTuningReport};

fn format_winner_summary(reports: &[ModelTuningReport]) {
    println!("\n=========================================================================================================================================================");
    println!("                                                AUTOMATED OPTIMIZER: WINNING METHODS SUMMARY");
    println!("=========================================================================================================================================================");
    println!(
        "| {:<24} | {:<28} | {:<7} | {:>9} | {:>8} | {:>8} | {:>8} | {:>10} |",
        "Model Name", "Winning Method", "KV Prec", "RAM (MiB)", "TTFT(ms)", "Gen(t/s)", "P50 (ms)", "Score"
    );
    println!("|--------------------------|------------------------------|---------|-----------|----------|----------|----------|------------|");

    for rep in reports {
        let w = rep.winner();
        println!(
            "| {:<24} | {:<28} | {:<7} | {:>9.1} | {:>8.1} | {:>8.1} | {:>8.2} | {:>10.1} |",
            rep.model_name.replace(".gguf", ""),
            w.method_name,
            w.kv_precision,
            w.memory.total_memory_mib(),
            w.report.time_to_first_token.as_secs_f64() * 1000.0,
            w.report.generation_tok_per_sec,
            w.report.latency_p50_ms,
            w.score.composite_score
        );
    }
    println!("=========================================================================================================================================================\n");

    println!(">>> Detailed Architectural Diagnoses & Rationales:\n");
    for (i, rep) in reports.iter().enumerate() {
        println!("  [{}] {}:", i + 1, rep.model_name);
        println!("      Winner: {} ({})", rep.winner().method_name, rep.winner().kv_precision);
        println!("      Rationale: {}", rep.architectural_rationale);
        println!();
    }
}

fn main() -> Result<()> {
    kortex::engine::silence_llama_logs();

    let models_dir = PathBuf::from("models");
    if !models_dir.exists() {
        eprintln!("Directory 'models' not found.");
        return Ok(());
    }

    // Auto-discover all GGUF models
    let mut model_paths: Vec<PathBuf> = fs::read_dir(&models_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "gguf"))
        .collect();

    model_paths.sort();

    if model_paths.is_empty() {
        eprintln!("No GGUF models found in 'models/'.");
        return Ok(());
    }

    println!("#########################################################################################");
    println!("#  AUTOMATED LLM INFERENCE METHOD OPTIMIZER & DEEP EFFICIENCY BENCHMARK                 #");
    println!("#  Hardware: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB) | AMD Ryzen 5 6600H (12T)    #");
    println!("#########################################################################################\n");

    println!(">>> Discovered {} GGUF models for automated profiling:", model_paths.len());
    for (idx, p) in model_paths.iter().enumerate() {
        println!("    {}. {}", idx + 1, p.file_name().unwrap().to_string_lossy());
    }
    println!();

    let tuner = AutoTuner::new(BenchmarkProfile::Balanced);
    let mut reports: Vec<ModelTuningReport> = Vec::new();

    for p in &model_paths {
        let name = p.file_name().unwrap().to_string_lossy();
        println!("-----------------------------------------------------------------------------------------");
        println!(">>> Optimizing Model: {}", name);
        println!("-----------------------------------------------------------------------------------------");

        let report = tuner.tune_model(p)?;

        println!("\n    Ranking for {}:", name);
        println!(
            "    | {:<4} | {:<28} | {:>8} | {:>8} | {:>8} | {:>8} | {:>9} |",
            "Rank", "Candidate Method", "Gen(t/s)", "TTFT(ms)", "P50(ms)", "RAM(MiB)", "Score"
        );
        println!("    |------|------------------------------|----------|----------|----------|----------|-----------|");

        for (rank, cand) in report.candidates.iter().enumerate() {
            let medal = match rank {
                0 => "🥇 1",
                1 => "🥈 2",
                2 => "🥉 3",
                _ => "",
            };
            let r_str = if medal.is_empty() { format!("{:>4}", rank + 1) } else { medal.to_string() };

            println!(
                "    | {:<4} | {:<28} | {:>8.1} | {:>8.1} | {:>8.2} | {:>8.1} | {:>9.1} |",
                r_str,
                cand.method_name,
                cand.report.generation_tok_per_sec,
                cand.report.time_to_first_token.as_secs_f64() * 1000.0,
                cand.report.latency_p50_ms,
                cand.memory.total_memory_mib(),
                cand.score.composite_score
            );
        }
        println!();

        reports.push(report);
    }

    format_winner_summary(&reports);

    Ok(())
}
