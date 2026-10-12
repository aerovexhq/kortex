//! Sliding-Window Attention & StreamingLLM Engine.
//!
//! Controls the KV cache at the C memory layer via `llama_memory_seq_rm` and `llama_memory_seq_add`.
//! Retains initial attention-sink tokens (crucial for autoregressive attention stability) and
//! maintains a rolling window of recent tokens, evicting middle KV cache entries on-the-fly.
//!
//! Guarantees strictly bounded O(1) VRAM consumption regardless of sequence length.

use anyhow::{bail, Context, Result};
use std::ffi::{c_char, CString};
use std::ptr;
use llama_cpp_sys_2::{
    ggml_backend_dev_get, ggml_backend_dev_t, llama_batch, llama_batch_free, llama_batch_init,
    llama_context, llama_context_default_params, llama_decode, llama_free,
    llama_get_memory, llama_init_from_model, llama_memory_clear, llama_memory_seq_add,
    llama_memory_seq_rm, llama_model, llama_model_default_params, llama_model_free, llama_model_get_vocab,
    llama_model_load_from_file, llama_model_n_embd, llama_model_n_head,
    llama_model_n_head_kv, llama_model_n_layer, llama_model_n_params, llama_model_size,
    llama_token, llama_token_to_piece, llama_tokenize, llama_vocab, llama_vocab_is_eog,
};

use crate::benchmark::TokenSpeedCalculator;
use crate::sampling::SamplerChain;
use crate::engine::{
    global_backend, EngineConfig, GenerationOutput, GenerationParams, ModelEngine,
    ModelMemoryProfile, Utf8StreamDecoder,
};

/// Engine with active KV cache eviction and positional shifting.
pub struct SlidingWindowEngine {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
    batch: llama_batch,
    config: EngineConfig,
    device_label: String,
    sink_tokens: i32,
    window_size: i32,
    evict_chunk: i32,
    pub n_past: i32,
    /// Reusable tokenization buffer (avoids a per-call heap allocation).
    token_scratch: Vec<llama_token>,
    #[allow(dead_code)]
    devices_buf: Box<[ggml_backend_dev_t; 16]>,
}

unsafe impl Send for SlidingWindowEngine {}
unsafe impl Sync for SlidingWindowEngine {}

impl SlidingWindowEngine {
    /// Creates a new engine with explicit attention sink and window bounds.
    pub fn new(
        mut config: EngineConfig,
        sink_tokens: i32,
        window_size: i32,
        evict_chunk: i32,
    ) -> Result<Self> {
        let _backend = global_backend();

        let total_budget = sink_tokens + window_size + evict_chunk + 16;
        if config.ctx_size < total_budget as u32 {
            config.ctx_size = total_budget as u32;
        }

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

            let batch = llama_batch_init(512, 0, 1);

            Ok(Self {
                model,
                ctx,
                vocab,
                batch,
                config,
                device_label,
                sink_tokens,
                window_size,
                evict_chunk,
                n_past: 0,
                token_scratch: Vec::with_capacity(1024),
                devices_buf,
            })
        }
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

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

impl Drop for SlidingWindowEngine {
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

impl SlidingWindowEngine {
    /// Clears the entire KV cache and resets the sequence position to zero.
    fn reset_kv(&mut self) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            llama_memory_clear(mem, true);
        }
        self.n_past = 0;
    }

    /// Evicts `count` tokens immediately after the attention-sink region and shifts the
    /// trailing window backwards so sequence positions remain contiguous.
    unsafe fn evict_middle(&mut self, count: i32) {
        let mem = llama_get_memory(self.ctx);
        let p0 = self.sink_tokens;
        let p1 = p0 + count;
        // Evict intermediate tokens [p0, p1)
        llama_memory_seq_rm(mem, 0, p0, p1);
        // Shift trailing window tokens backwards by `count`
        llama_memory_seq_add(mem, 0, p1, self.n_past, -count);
        self.n_past -= count;
    }

    /// Shared generation core.
    ///
    /// * `incremental == false`: the KV cache is wiped and `prompt` is evaluated from position 0.
    /// * `incremental == true` (with an active context): `prompt` is only the new suffix and is
    ///   appended after the existing `n_past` tokens, evicting the middle of the window if needed.
    fn run_generation(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        mut on_token: Option<&mut dyn FnMut(&str) -> bool>,
        incremental: bool,
    ) -> Result<GenerationOutput> {
        let mut profiler = TokenSpeedCalculator::new();
        profiler.start();

        // A fresh sequence is required for stateless calls or when no prior context exists.
        let fresh = !incremental || self.n_past == 0;
        if fresh {
            self.reset_kv();
        }

        // Suffix prompts and already-templated prompts are used verbatim.
        let formatted = if !fresh
            || prompt.starts_with("<|im_start|>")
            || prompt.starts_with("<|start_header_id|>")
        {
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
        tokens.resize(text_bytes.len() + 512, 0 as llama_token);
        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                text_bytes.as_ptr().cast::<c_char>(),
                text_bytes.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                // BOS/special insertion only belongs at the start of a sequence.
                fresh,
                true,
            )
        };

        if n_tokens <= 0 {
            bail!("Raw tokenize failed or returned 0 tokens");
        }
        tokens.truncate(n_tokens as usize);

        let max_capacity = (self.sink_tokens + self.window_size) as usize;
        if fresh {
            // StreamingLLM Attention-Sink Pruning if prompt exceeds window budget
            if tokens.len() > max_capacity {
                let sink_count = (self.sink_tokens as usize).min(tokens.len());
                let tail_count = max_capacity.saturating_sub(sink_count);
                let mut pruned = Vec::with_capacity(max_capacity);
                pruned.extend_from_slice(&tokens[..sink_count]);
                let tail_start = tokens.len().saturating_sub(tail_count);
                pruned.extend_from_slice(&tokens[tail_start..]);
                tokens = pruned;
            }
        } else {
            // Make room for the suffix by evicting the middle of the existing window.
            let cap = max_capacity as i32;
            let overflow = self.n_past + tokens.len() as i32 - cap;
            if overflow > 0 {
                let chunk = self.evict_chunk.max(1);
                let rounded = ((overflow + chunk - 1) / chunk) * chunk;
                let evictable = (self.n_past - self.sink_tokens).max(0);
                let evict = rounded.min(evictable);
                if evict > 0 {
                    unsafe { self.evict_middle(evict) };
                }
                // If the suffix alone still exceeds the remaining room, keep its tail.
                let room = (cap - self.n_past).max(1) as usize;
                if tokens.len() > room {
                    let drop_count = tokens.len() - room;
                    tokens.drain(..drop_count);
                }
            }
        }

        // 1. Evaluate (suffix) prompt with chunked prefill starting at the current position
        profiler.start_prompt();
        let start_pos = self.n_past;
        unsafe {
            let last_idx = tokens.len() - 1;
            for (chunk_idx, chunk) in tokens.chunks(512).enumerate() {
                self.clear_batch();
                let base = chunk_idx * 512;
                for (offset, &tok) in chunk.iter().enumerate() {
                    let idx = base + offset;
                    let is_last = idx == last_idx;
                    self.add_to_batch(tok, start_pos + idx as i32, is_last);
                }

                let status = llama_decode(self.ctx, self.batch);
                if status != 0 {
                    bail!("Chunked prefill llama_decode failed with error code: {}", status);
                }
            }
        }
        self.n_past = start_pos + tokens.len() as i32;
        profiler.finish_prompt(tokens.len());
        // Hand the buffer back for the next call.
        self.token_scratch = tokens;

        // 2. Autoregressive loop with StreamingLLM Attention-Sink & Window Eviction
        let mut generated_count = 0;
        let mut output_text = String::new();
        let mut piece_buf = [0 as c_char; 256];
        let mut utf8 = Utf8StreamDecoder::new();
        let mut sampler = SamplerChain::from_params(params);
        let max_target = params.max_tokens;
        let cap = self.sink_tokens + self.window_size;

        while generated_count < max_target {
            let token = unsafe { sampler.sample(self.ctx, -1) };

            if unsafe { llama_vocab_is_eog(self.vocab, token) } {
                break;
            }

            profiler.record_token();
            generated_count += 1;

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
                // Multi-byte code points may be split across tokens; only emit complete ones.
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

            // KV Cache Surgery: Check if we reached the attention-sink + window bound
            if self.n_past >= cap {
                let chunk = self.evict_chunk;
                unsafe { self.evict_middle(chunk) };
            }

            // Step token at the current sequence position
            unsafe {
                self.clear_batch();
                let pos = self.n_past;
                self.add_to_batch(token, pos, true);
                let status = llama_decode(self.ctx, self.batch);
                if status != 0 {
                    bail!("llama_decode step failed with error code: {}", status);
                }
            }
            self.n_past += 1;
        }

        // Flush any trailing partial bytes so nothing is silently dropped.
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

impl ModelEngine for SlidingWindowEngine {
    fn name(&self) -> &'static str {
        "Sliding-Window Attention (Sink=4, Win=64, O(1) Cache)"
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

            // Bounded window size determines the effective maximum KV footprint!
            let effective_kv_tokens = (self.sink_tokens + self.window_size) as u32;

            ModelMemoryProfile::new(
                weights_bytes,
                n_params,
                n_layer,
                n_head,
                n_head_kv,
                n_embd,
                effective_kv_tokens,
                self.config.kv_precision,
            )
        }
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        let result = self.run_generation(prompt, params, on_token, false);
        if result.is_err() {
            // Leave the engine in a clean state so the next turn re-prefills from scratch.
            self.reset_kv();
        }
        result
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn n_past(&self) -> i32 {
        self.n_past
    }

    fn reset_context(&mut self) -> Result<()> {
        self.reset_kv();
        Ok(())
    }

    fn generate_incremental(
        &mut self,
        suffix_prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        let result = self.run_generation(suffix_prompt, params, on_token, true);
        if result.is_err() {
            self.reset_kv();
        }
        result
    }
}
