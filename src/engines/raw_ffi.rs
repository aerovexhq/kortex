use std::ffi::{c_char, CString};
use std::ptr;
use anyhow::{bail, Context, Result};
use llama_cpp_sys_2::{
    ggml_backend_dev_get, ggml_backend_dev_t, llama_batch, llama_batch_free, llama_batch_init,
    llama_context, llama_context_default_params, llama_decode, llama_free, llama_get_memory, llama_init_from_model, llama_memory_clear, llama_model,
    llama_model_default_params, llama_model_free,
    llama_model_get_vocab, llama_model_load_from_file, llama_model_n_embd, llama_model_n_head,
    llama_model_n_head_kv, llama_model_n_layer, llama_model_n_params, llama_model_size,
    llama_token, llama_token_to_piece, llama_tokenize, llama_vocab, llama_vocab_is_eog,
};

use crate::benchmark::TokenSpeedCalculator;
use crate::sampling::SamplerChain;
use crate::engine::{
    EngineConfig, GenerationOutput, GenerationParams, ModelEngine, ModelMemoryProfile,
    Utf8StreamDecoder,
};

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
    /// Reusable tokenization buffer (avoids a per-call heap allocation).
    token_scratch: Vec<llama_token>,
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
                token_scratch: Vec::with_capacity(1024),
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

        // 0. Reset KV cache before evaluating new prompt
        unsafe {
            let mem = llama_get_memory(self.ctx);
            llama_memory_clear(mem, true);
        }

        let formatted = if prompt.starts_with("<|im_start|>") || prompt.starts_with("<|start_header_id|>") {
            prompt.to_string()
        } else {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        };
        // Tokenize straight from the UTF-8 bytes (explicit length, no NUL needed) into the
        // reusable scratch buffer.
        let text_bytes = formatted.as_bytes();
        let mut tokens = std::mem::take(&mut self.token_scratch);
        tokens.clear();
        tokens.resize(
            (text_bytes.len() + 512).max((self.config.ctx_size as usize) + 512),
            0 as llama_token,
        );
        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                text_bytes.as_ptr().cast::<c_char>(),
                text_bytes.len() as i32,
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
        // Hand the buffer back for the next call.
        let n_prompt = tokens.len() as i32;
        self.token_scratch = tokens;

        // 2. Autoregressive loop
        let mut n_cur = n_prompt;
        let max_target = n_cur + params.max_tokens as i32;
        let mut output_text = String::new();
        let mut generated_count = 0;
        let mut piece_buf = [0 as c_char; 256];
        let mut utf8 = Utf8StreamDecoder::new();
        let mut sampler = SamplerChain::from_params(params);

        while n_cur < max_target {
            let token = unsafe { sampler.sample(self.ctx, -1) };

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
                if let Some(piece) = utf8.feed(u8_slice) {
                    output_text.push_str(&piece);
                    if let Some(cb) = on_token.as_mut() {
                        let cont = cb(&piece);
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

        if let Some(rest) = utf8.flush() {
            output_text.push_str(&rest);
            if let Some(cb) = on_token.as_mut() {
                let _ = cb(&rest);
            }
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
