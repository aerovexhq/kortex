use anyhow::Result;
use kortex::engine::{EngineConfig, GenerationParams, KvPrecision, ModelEngine};
use kortex::engines::quantized_kv::QuantizedKvEngine;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_path = PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}", model_path);
        return Ok(());
    }

    let prompt = "Explain the difference between lossless compression and quantization in computer science. Be concise.";
    let params = GenerationParams {
        max_tokens: 80,
        temperature: 0.0, // greedy argmax to test deterministic output divergence
        top_p: 1.0,
        seed: 42,
    };

    println!("===============================================================");
    println!("KV CACHE QUALITY & DIVERGENCE TEST (GREEDY ARGMAX, TEMP=0.0)");
    println!("Prompt: \"{}\"", prompt);
    println!("===============================================================\n");

    let precisions = [KvPrecision::F16, KvPrecision::Q8_0, KvPrecision::Q4_0];
    let mut outputs = Vec::new();

    for &prec in &precisions {
        let config = EngineConfig {
            model_path: model_path.clone(),
            device_index: Some(1),
            n_gpu_layers: 999,
            ctx_size: 512,
            kv_precision: prec,
            n_threads: Some(6),
        };

        let mut engine = QuantizedKvEngine::new(config, prec)?;
        let gen = engine.generate(prompt, &params, None)?;
        println!("Precision: {:?}", prec);
        println!("KV Memory: {:.2} KiB/token", engine.memory_profile().kv_bytes_per_token as f64 / 1024.0);
        println!("Speed: {:.1} tok/s (TTFT: {:.2} ms)", gen.report.generation_tok_per_sec, gen.report.time_to_first_token.as_secs_f64() * 1000.0);
        println!("Generated Text:\n{}\n", gen.text.trim());
        println!("---------------------------------------------------------------");
        outputs.push((prec, gen.text));
    }

    let fp16_text = &outputs[0].1;
    let q8_text = &outputs[1].1;
    let q4_text = &outputs[2].1;

    println!("\n=== OUTPUT DIVERGENCE ANALYSIS ===");
    println!("Q8_0 identical to FP16? {}", fp16_text == q8_text);
    println!("Q4_0 identical to FP16? {}", fp16_text == q4_text);

    Ok(())
}
