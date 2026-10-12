use std::ffi::{c_char, CString};
use std::ptr;
use anyhow::{bail, Context, Result};
use llama_cpp_sys_2::{
    ggml_backend_dev_get, ggml_backend_dev_t, llama_batch, llama_batch_free, llama_batch_init,
    llama_context, llama_context_default_params, llama_decode, llama_free, llama_get_logits_ith,
    llama_init_from_model, llama_model, llama_model_default_params, llama_model_free,
    llama_model_get_vocab, llama_model_load_from_file, llama_model_n_embd, llama_model_n_head,
    llama_model_n_head_kv, llama_model_n_layer, llama_model_n_params, llama_model_size,
    llama_token, llama_token_to_piece, llama_tokenize, llama_vocab, llama_vocab_is_eog,
    llama_vocab_n_tokens,
};

use crate::benchmark::TokenSpeedCalculator;
use crate::engine::{EngineConfig, GenerationOutput, GenerationParams, ModelEngine, ModelMemoryProfile};

/// Low-Level Engine bypassing all safe wrapper layers.
/// Directly manages C pointers, manual batch allocations, raw logits memory,
/// and performs greedy argmax directly over the raw C float buffers.
pub struct RawFfiEngine {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
    batch: llama_batch,
    config: EngineConfig,
    device_label: String,
    #[allow(dead_code)]
    devices_buf: Box<[ggml_backend_dev_t; 16]>,
}

unsafe impl Send for RawFfiEngine {}

impl RawFfiEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        let _backend = crate::engine::global_backend();

        let mut devices_buf = Box::new([ptr::null_mut(); 16]);
        let mut device_label = "CPU".to_string();

        unsafe {
            let mut mparams = llama_model_default_params();
            mparams.n_gpu_layers = config.n_gpu_layers as i32;

            if let Some(dev_idx) = config.device_index {
                let dev = ggml_backend_dev_get(dev_idx);
                if !dev.is_null() {
                    devices_buf[0] = dev;
                    mparams.devices = devices_buf.as_mut_ptr();
                    device_label = format!("Vulkan Device [{}] via Raw C FFI", dev_idx);
                }
            }

            let c_path = CString::new(
                config
                    .model_path
                    .to_str()
                    .context("Invalid model path UTF-8")?,
            )?;

            let model = llama_model_load_from_file(c_path.as_ptr(), mparams);
            if model.is_null() {
                bail!("llama_model_load_from_file returned NULL pointer");
            }

            let mut cparams = llama_context_default_params();
            cparams.n_ctx = config.ctx_size;
            cparams.n_batch = 512;
            cparams.type_k = config.kv_precision.to_ggml_type();
            cparams.type_v = config.kv_precision.to_ggml_type();
            if let Some(threads) = config.n_threads {
                cparams.n_threads = threads;
            }

            let ctx = llama_init_from_model(model, cparams);
            if ctx.is_null() {
                llama_model_free(model);
                bail!("llama_init_from_model returned NULL context pointer");
            }

            let vocab = llama_model_get_vocab(model);
            if vocab.is_null() {
                llama_free(ctx);
                llama_model_free(model);
                bail!("llama_model_get_vocab returned NULL vocab pointer");
            }

            // Allocate raw C batch buffer (512 tokens, 0 embd, 1 sequence)
            let batch = llama_batch_init(512, 0, 1);

            Ok(Self {
                model,
                ctx,
                vocab,
                batch,
                config,
                device_label,
                devices_buf,
            })
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    #[inline]
    unsafe fn add_to_batch(&mut self, token: llama_token, pos: i32, logits: bool) {
        let idx = self.batch.n_tokens as usize;
        *self.batch.token.add(idx) = token;
        *self.batch.pos.add(idx) = pos;
        *self.batch.n_seq_id.add(idx) = 1;
        *(*self.batch.seq_id.add(idx)).add(0) = 0;
        *self.batch.logits.add(idx) = if logits { 1 } else { 0 };
        self.batch.n_tokens += 1;
    }

    #[inline]
    unsafe fn clear_batch(&mut self) {
        self.batch.n_tokens = 0;
    }

    /// Low-level greedy argmax: reads raw `*mut f32` logit array directly from the
    /// C context memory and returns the highest scoring token index.
    unsafe fn sample_greedy_argmax(&self) -> llama_token {
        let n_vocab = llama_vocab_n_tokens(self.vocab);
        let logits_ptr = llama_get_logits_ith(self.ctx, -1);
        if logits_ptr.is_null() {
            return 0;
        }

        let logits = std::slice::from_raw_parts(logits_ptr, n_vocab as usize);

        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx as llama_token
    }
}

impl Drop for RawFfiEngine {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.batch);
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}

impl ModelEngine for RawFfiEngine {
    fn name(&self) -> &'static str {
        "RawFfiEngine (Direct C ABI FFI, raw pointer logits & manual argmax)"
    }

    fn device_name(&self) -> String {
        self.device_label.clone()
    }

    fn memory_profile(&self) -> ModelMemoryProfile {
        unsafe {
            let weights_bytes = llama_model_size(self.model);
            let n_params = llama_model_n_params(self.model);
            let n_layer = llama_model_n_layer(self.model) as u32;
            let n_head = llama_model_n_head(self.model) as u32;
            let n_head_kv = llama_model_n_head_kv(self.model) as u32;
            let n_embd = llama_model_n_embd(self.model) as u32;

            ModelMemoryProfile::new(
                weights_bytes,
                n_params,
                n_layer,
                n_head,
                n_head_kv,
                n_embd,
                self.config.ctx_size,
                self.config.kv_precision,
            )
        }
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        let mut profiler = TokenSpeedCalculator::new();
        profiler.start();

        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );
        let c_text = CString::new(formatted)?;

        let max_alloc = (c_text.to_bytes().len() + 512).max((self.config.ctx_size as usize) + 512);
        let mut tokens = vec![0 as llama_token; max_alloc];
        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                c_text.to_bytes().len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            )
        };

        if n_tokens <= 0 {
            bail!("Raw tokenize failed or returned 0 tokens");
        }
        tokens.truncate(n_tokens as usize);

        // 1. Evaluate prompt with chunked prefill
        profiler.start_prompt();
        unsafe {
            let last_idx = tokens.len() - 1;
            for (chunk_idx, chunk) in tokens.chunks(512).enumerate() {
                self.clear_batch();
                let base_pos = chunk_idx * 512;
                for (offset, &tok) in chunk.iter().enumerate() {
                    let global_idx = base_pos + offset;
                    let is_last = global_idx == last_idx;
                    self.add_to_batch(tok, global_idx as i32, is_last);
                }

                let status = llama_decode(self.ctx, self.batch);
                if status != 0 {
                    bail!("Chunked prefill llama_decode failed with error code: {}", status);
                }
            }
        }
        profiler.finish_prompt(tokens.len());

        // 2. Autoregressive loop
        let mut n_cur = tokens.len() as i32;
        let max_target = n_cur + params.max_tokens as i32;
        let mut output_text = String::new();
        let mut generated_count = 0;
        let mut piece_buf = [0 as c_char; 256];

        while n_cur < max_target {
            let token = unsafe { self.sample_greedy_argmax() };

            if unsafe { llama_vocab_is_eog(self.vocab, token) } {
                break;
            }

            profiler.record_token();
            generated_count += 1;

            // Convert token to piece
            let bytes_written = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    piece_buf.as_mut_ptr(),
                    piece_buf.len() as i32,
                    0,
                    true,
                )
            };

            if bytes_written > 0 {
                let u8_slice = unsafe {
                    std::slice::from_raw_parts(piece_buf.as_ptr().cast::<u8>(), bytes_written as usize)
                };
                if let Ok(piece) = std::str::from_utf8(u8_slice) {
                    output_text.push_str(piece);
                    if let Some(cb) = on_token.as_mut() {
                        let cont = cb(piece);
                        if !cont {
                            break;
                        }
                    }
                }
            }

            // Step single token
            unsafe {
                self.clear_batch();
                self.add_to_batch(token, n_cur, true);
                let status = llama_decode(self.ctx, self.batch);
                if status != 0 {
                    bail!("Token step llama_decode failed with status {}", status);
                }
            }

            n_cur += 1;
        }

        let report = profiler.stop();

        let memory = self.memory_profile();

        Ok(GenerationOutput {
            text: output_text,
            generated_tokens: generated_count,
            report,
            memory,
        })
    }
}
