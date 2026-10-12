use std::num::NonZeroU32;
use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::benchmark::TokenSpeedCalculator;
use crate::engine::{EngineConfig, GenerationOutput, GenerationParams, ModelEngine};

/// High-Level Safe Engine using the idiomatic `llama-cpp-2` Rust wrapper.
pub struct SafeLlamaEngine {
    model: LlamaModel,
    config: EngineConfig,
    device_label: String,
}

impl SafeLlamaEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let backend = crate::engine::global_backend();
        let devices = llama_cpp_2::list_llama_ggml_backend_devices();

        let device_label = if let Some(idx) = config.device_index {
            devices
                .get(idx)
                .map(|d| format!("{} ({})", d.name, d.description))
                .unwrap_or_else(|| format!("Device {}", idx))
        } else {
            "CPU (Default)".to_string()
        };

        let mut model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);

        if let Some(dev_idx) = config.device_index {
            if dev_idx < devices.len() {
                model_params = model_params
                    .with_devices(&[dev_idx])
                    .context("Failed setting device on LlamaModelParams")?;
            }
        }

        let model = LlamaModel::load_from_file(backend, &config.model_path, &model_params)
            .context("Failed to load model file")?;

        Ok(Self {
            model,
            config,
            device_label,
        })
    }
}

impl ModelEngine for SafeLlamaEngine {
    fn name(&self) -> &'static str {
        "SafeLlamaEngine (Safe Rust wrapper with Vulkan)"
    }

    fn device_name(&self) -> String {
        self.device_label.clone()
    }

    fn memory_profile(&self) -> crate::engine::ModelMemoryProfile {
        crate::engine::ModelMemoryProfile::new(
            self.model.size(),
            self.model.n_params(),
            self.model.n_layer(),
            self.model.n_head(),
            self.model.n_head_kv(),
            self.model.n_embd() as u32,
            self.config.ctx_size,
            self.config.kv_precision,
        )
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        let mut profiler = TokenSpeedCalculator::new();
        profiler.start();

        // 1. Create fresh context
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.config.ctx_size).unwrap()))
            .with_n_batch(512)
            .with_type_k(self.config.kv_precision.to_kv_cache_type())
            .with_type_v(self.config.kv_precision.to_kv_cache_type());

        if let Some(t) = self.config.n_threads {
            ctx_params = ctx_params.with_n_threads(t);
        }

        let mut ctx = self
            .model
            .new_context(crate::engine::global_backend(), ctx_params)
            .context("Failed creating llama context")?;

        // 2. Tokenize
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );
        let tokens = self.model.vocab().tokenize(formatted.as_bytes(), true, true);
        if tokens.is_empty() {
            bail!("Prompt yielded 0 tokens");
        }

        // 3. Evaluate prompt batch
        let mut batch = LlamaBatch::new(512, 1);
        let last_idx = (tokens.len() - 1) as i32;

        profiler.start_prompt();
        for (i, token) in tokens.iter().enumerate() {
            let is_last = (i as i32) == last_idx;
            batch
                .add(*token, i as i32, &[0], is_last)
                .context("Failed adding token to batch")?;
        }
        ctx.decode(&mut batch).context("Prompt decode failed")?;
        profiler.finish_prompt(tokens.len());

        // 4. Sampler setup
        let mut sampler = if params.temperature <= 0.001 {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_p(params.top_p, 1),
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(params.seed),
            ])
        };

        // 5. Autoregressive loop
        let mut n_cur = tokens.len() as i32;
        let max_target = n_cur + params.max_tokens as i32;
        let mut output_text = String::new();
        let mut generated_count = 0;

        while n_cur < max_target {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.vocab().is_eog(token) {
                break;
            }

            profiler.record_token();
            generated_count += 1;

            let bytes = self.model.vocab().detokenize(&[token], false, true);
            if let Ok(piece) = std::str::from_utf8(&bytes) {
                output_text.push_str(piece);
                if let Some(cb) = on_token.as_mut() {
                    let should_continue = cb(piece);
                    if !should_continue {
                        break;
                    }
                }
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("Failed adding generated token to batch")?;
            ctx.decode(&mut batch).context("Token decode failed")?;

            n_cur += 1;
        }

        let report = profiler.stop();
        let internal_timings = ctx.timings();

        println!("    [Internal llama.cpp C++ Core Profiler]:");
        for line in format!("{}", internal_timings).lines() {
            if !line.trim().is_empty() {
                println!("      {}", line.trim());
            }
        }

        let memory = self.memory_profile();

        Ok(GenerationOutput {
            text: output_text,
            generated_tokens: generated_count,
            report,
            memory,
        })
    }
}
