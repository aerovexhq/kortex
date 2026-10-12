//! Dual-Model Speculative Decoding Engine.
//!
//! Co-hosts Qwen 2.5 0.5B (Draft Model) and Qwen 2.5 1.5B (Target Model)
//! simultaneously in RTX 3050 Ti VRAM (Combined weights: ~1.52 GiB / 4 GiB).
//!
//! Executes K-step autoregressive drafting with the fast 0.5B model, followed by
//! single-pass parallel batch verification on the 1.5B target model.

use anyhow::{bail, Context, Result};
use std::ffi::{c_char, CString};
use std::ptr;
use llama_cpp_sys_2::{
    ggml_backend_dev_get, ggml_backend_dev_t, llama_batch, llama_batch_free, llama_batch_init,
    llama_context, llama_context_default_params, llama_decode, llama_free, llama_get_logits_ith,
    llama_get_memory, llama_init_from_model, llama_memory_clear, llama_memory_seq_rm,
    llama_model, llama_model_default_params, llama_model_free, llama_model_get_vocab,
    llama_model_load_from_file, llama_model_size, llama_token, llama_token_to_piece,
    llama_tokenize, llama_vocab, llama_vocab_is_eog, llama_vocab_n_tokens,
};

use crate::benchmark::TokenSpeedCalculator;
use crate::engine::{
    global_backend, EngineConfig, GenerationOutput, GenerationParams, KvPrecision,
    ModelEngine, ModelMemoryProfile,
};

pub struct SpeculativeStats {
    pub total_generated: usize,
    pub total_drafted: usize,
    pub total_accepted: usize,
    pub acceptance_rate: f64,
}

pub struct SpeculativeEngine {
    model_draft: *mut llama_model,
    ctx_draft: *mut llama_context,
    model_target: *mut llama_model,
    ctx_target: *mut llama_context,
    vocab: *const llama_vocab,
    draft_batch: llama_batch,
    target_batch: llama_batch,
    config: EngineConfig,
    device_label: String,
    #[allow(dead_code)]
    initial_k: usize,
    /// Exponential moving average of acceptance rate (0.0 to 1.0)
    ema_acceptance: f64,
    /// Dynamic draft horizon K in [1, 6]
    current_k: usize,
    #[allow(dead_code)]
    devices_buf: Box<[ggml_backend_dev_t; 16]>,
    pub last_stats: SpeculativeStats,
}

unsafe impl Send for SpeculativeEngine {}
unsafe impl Sync for SpeculativeEngine {}

impl SpeculativeEngine {
    pub fn current_k(&self) -> usize {
        self.current_k
    }

    pub fn ema_acceptance(&self) -> f64 {
        self.ema_acceptance
    }

    pub fn new(
        draft_path: &std::path::Path,
        target_path: &std::path::Path,
        config: EngineConfig,
        draft_k: usize,
    ) -> Result<Self> {
        let _backend = global_backend();

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
                    device_label = format!("Vulkan Device [{}] Speculative", dev_idx);
                }
            }

            // 1. Load Draft Model (0.5B)
            let c_draft = CString::new(draft_path.to_str().context("Invalid draft path")?)?;
            let model_draft = llama_model_load_from_file(c_draft.as_ptr(), mparams);
            if model_draft.is_null() {
                bail!("Failed loading draft model");
            }

            // 2. Load Target Model (1.5B)
            let c_target = CString::new(target_path.to_str().context("Invalid target path")?)?;
            let model_target = llama_model_load_from_file(c_target.as_ptr(), mparams);
            if model_target.is_null() {
                llama_model_free(model_draft);
                bail!("Failed loading target model");
            }

            // 3. Create Contexts with Q8_0 KV Cache
            let mut cparams = llama_context_default_params();
            cparams.n_ctx = config.ctx_size;
            cparams.n_batch = 512;
            cparams.type_k = KvPrecision::Q8_0.to_ggml_type();
            cparams.type_v = KvPrecision::Q8_0.to_ggml_type();

            let ctx_draft = llama_init_from_model(model_draft, cparams);
            if ctx_draft.is_null() {
                llama_model_free(model_target);
                llama_model_free(model_draft);
                bail!("Failed initializing draft context");
            }

            let ctx_target = llama_init_from_model(model_target, cparams);
            if ctx_target.is_null() {
                llama_free(ctx_draft);
                llama_model_free(model_target);
                llama_model_free(model_draft);
                bail!("Failed initializing target context");
            }

            let vocab = llama_model_get_vocab(model_target);
            let draft_batch = llama_batch_init(512, 0, 1);
            let target_batch = llama_batch_init(512, 0, 1);

            let initial_k = draft_k.clamp(1, 6);

            Ok(Self {
                model_draft,
                ctx_draft,
                model_target,
                ctx_target,
                vocab,
                draft_batch,
                target_batch,
                config,
                device_label,
                initial_k,
                ema_acceptance: 0.70, // Baseline assumption: 70% acceptance
                current_k: initial_k,
                devices_buf,
                last_stats: SpeculativeStats {
                    total_generated: 0,
                    total_drafted: 0,
                    total_accepted: 0,
                    acceptance_rate: 0.0,
                },
            })
        }
    }

    #[inline]
    unsafe fn add_to_batch(batch: &mut llama_batch, token: llama_token, pos: i32, logits: bool) {
        let idx = batch.n_tokens as usize;
        *batch.token.add(idx) = token;
        *batch.pos.add(idx) = pos;
        *batch.n_seq_id.add(idx) = 1;
        *(*batch.seq_id.add(idx)).add(0) = 0;
        *batch.logits.add(idx) = if logits { 1 } else { 0 };
        batch.n_tokens += 1;
    }

    #[inline]
    unsafe fn sample_greedy(ctx: *mut llama_context, vocab: *const llama_vocab, batch_idx: i32) -> llama_token {
        let logits_ptr = llama_get_logits_ith(ctx, batch_idx);
        if logits_ptr.is_null() {
            return 0;
        }
        let n_vocab = llama_vocab_n_tokens(vocab) as usize;
        let logits = std::slice::from_raw_parts(logits_ptr, n_vocab);
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

impl Drop for SpeculativeEngine {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.draft_batch);
            llama_batch_free(self.target_batch);
            if !self.ctx_draft.is_null() {
                llama_free(self.ctx_draft);
            }
            if !self.ctx_target.is_null() {
                llama_free(self.ctx_target);
            }
            if !self.model_draft.is_null() {
                llama_model_free(self.model_draft);
            }
            if !self.model_target.is_null() {
                llama_model_free(self.model_target);
            }
        }
    }
}

impl ModelEngine for SpeculativeEngine {
    fn name(&self) -> &'static str {
        "Dual-Model Speculative Decoding (0.5B Draft -> 1.5B Target, Q8_0 KV)"
    }

    fn device_name(&self) -> String {
        self.device_label.clone()
    }

    fn memory_profile(&self) -> ModelMemoryProfile {
        unsafe {
            let draft_bytes = llama_model_size(self.model_draft);
            let target_bytes = llama_model_size(self.model_target);
            let total_weights = draft_bytes + target_bytes;

            let n_params = llama_cpp_sys_2::llama_model_n_params(self.model_target);
            let n_layer = llama_cpp_sys_2::llama_model_n_layer(self.model_target) as u32;
            let n_head = llama_cpp_sys_2::llama_model_n_head(self.model_target) as u32;
            let n_head_kv = llama_cpp_sys_2::llama_model_n_head_kv(self.model_target) as u32;
            let n_embd = llama_cpp_sys_2::llama_model_n_embd(self.model_target) as u32;

            ModelMemoryProfile::new(
                total_weights,
                n_params,
                n_layer,
                n_head,
                n_head_kv,
                n_embd,
                self.config.ctx_size,
                KvPrecision::Q8_0,
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

        // 0. Reset KV caches for both draft and target before evaluating new prompt
        unsafe {
            let mem_dft = llama_get_memory(self.ctx_draft);
            llama_memory_clear(mem_dft, true);
            let mem_tgt = llama_get_memory(self.ctx_target);
            llama_memory_clear(mem_tgt, true);
        }

        let formatted = if prompt.starts_with("<|im_start|>") || prompt.starts_with("<|start_header_id|>") {
            prompt.to_string()
        } else {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        };
        let c_text = CString::new(formatted)?;

        let max_alloc = c_text.to_bytes().len() + 512;
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
            bail!("Tokenize failed");
        }
        tokens.truncate(n_tokens as usize);

        // 1. Prefill both Draft and Target contexts
        profiler.start_prompt();
        unsafe {
            let last_idx = tokens.len() - 1;
            for (chunk_idx, chunk) in tokens.chunks(512).enumerate() {
                self.draft_batch.n_tokens = 0;
                self.target_batch.n_tokens = 0;
                let base_pos = chunk_idx * 512;

                for (offset, &tok) in chunk.iter().enumerate() {
                    let global_idx = base_pos + offset;
                    let is_last = global_idx == last_idx;
                    Self::add_to_batch(&mut self.draft_batch, tok, global_idx as i32, is_last);
                    Self::add_to_batch(&mut self.target_batch, tok, global_idx as i32, is_last);
                }

                if llama_decode(self.ctx_draft, self.draft_batch) != 0 {
                    bail!("Draft prefill failed");
                }
                if llama_decode(self.ctx_target, self.target_batch) != 0 {
                    bail!("Target prefill failed");
                }
            }
        }
        profiler.finish_prompt(tokens.len());

        // 2. Speculative Autoregressive Generation Loop
        let mut n_cur = tokens.len() as i32;
        let mut generated_count = 0;
        let mut drafted_count = 0;
        let mut accepted_count = 0;
        let mut output_text = String::new();
        let mut piece_buf = [0 as c_char; 256];

        let mut current_token = unsafe { Self::sample_greedy(self.ctx_target, self.vocab, -1) };

        while generated_count < params.max_tokens {
            if unsafe { llama_vocab_is_eog(self.vocab, current_token) } {
                break;
            }

            profiler.record_token();
            generated_count += 1;

            // Emit current token
            let bytes_written = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    current_token,
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
                        if !cb(piece) {
                            break;
                        }
                    }
                }
            }

            // Step A: Draft K candidate tokens from 0.5B model using dynamic horizon
            let active_k = self.current_k;
            let mut drafts: Vec<llama_token> = Vec::with_capacity(active_k);
            let mut draft_tok = current_token;

            for step in 0..active_k {
                unsafe {
                    self.draft_batch.n_tokens = 0;
                    Self::add_to_batch(&mut self.draft_batch, draft_tok, n_cur + step as i32, true);
                    if llama_decode(self.ctx_draft, self.draft_batch) != 0 {
                        break;
                    }
                    draft_tok = Self::sample_greedy(self.ctx_draft, self.vocab, 0);
                    drafts.push(draft_tok);
                    drafted_count += 1;
                    if llama_vocab_is_eog(self.vocab, draft_tok) {
                        break;
                    }
                }
            }

            if drafts.is_empty() {
                // Fallback standard step on target
                unsafe {
                    self.target_batch.n_tokens = 0;
                    Self::add_to_batch(&mut self.target_batch, current_token, n_cur, true);
                    llama_decode(self.ctx_target, self.target_batch);
                    current_token = Self::sample_greedy(self.ctx_target, self.vocab, 0);
                }
                n_cur += 1;
                continue;
            }

            // Step B: Parallel batch verification on 1.5B Target Model
            let num_drafts = drafts.len();
            unsafe {
                self.target_batch.n_tokens = 0;
                // Add current_token at n_cur
                Self::add_to_batch(&mut self.target_batch, current_token, n_cur, true);
                // Add drafts at n_cur + 1 .. n_cur + num_drafts
                for (i, &d) in drafts.iter().enumerate() {
                    Self::add_to_batch(&mut self.target_batch, d, n_cur + 1 + i as i32, true);
                }

                if llama_decode(self.ctx_target, self.target_batch) != 0 {
                    bail!("Target verification decode failed");
                }
            }

            // Step C: Rejection verification
            let mut accept_len = 0;
            let mut next_token = 0;

            for (i, &draft_tok) in drafts.iter().enumerate().take(num_drafts) {
                let target_predicted = unsafe { Self::sample_greedy(self.ctx_target, self.vocab, i as i32) };
                if target_predicted == draft_tok {
                    accept_len += 1;
                    accepted_count += 1;
                } else {
                    next_token = target_predicted;
                    break;
                }
            }

            // Dynamic EMA Adaptation:
            // alpha = 0.20 smoothing factor
            let round_rate = accept_len as f64 / num_drafts as f64;
            self.ema_acceptance = 0.80 * self.ema_acceptance + 0.20 * round_rate;

            // Adjust horizon K for subsequent steps:
            // High acceptance (>80%) -> expand horizon up to 6
            // Moderate acceptance (50-80%) -> maintain or gently converge
            // Low acceptance (<50%) -> contract horizon down to 1 to minimize wasted work
            if self.ema_acceptance > 0.85 && self.current_k < 6 {
                self.current_k += 1;
            } else if self.ema_acceptance < 0.45 && self.current_k > 1 {
                self.current_k -= 1;
            }

            // Emit accepted draft tokens
            for &tok in drafts.iter().take(accept_len) {
                profiler.record_token();
                generated_count += 1;

                let b_len = unsafe {
                    llama_token_to_piece(
                        self.vocab,
                        tok,
                        piece_buf.as_mut_ptr(),
                        piece_buf.len() as i32,
                        0,
                        true,
                    )
                };
                if b_len > 0 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(piece_buf.as_ptr().cast::<u8>(), b_len as usize)
                    };
                    if let Ok(piece) = std::str::from_utf8(slice) {
                        output_text.push_str(piece);
                        if let Some(cb) = on_token.as_mut() {
                            if !cb(piece) {
                                break;
                            }
                        }
                    }
                }
            }

            if accept_len == num_drafts {
                // All accepted! Next token is sampled from the last draft position
                next_token = unsafe { Self::sample_greedy(self.ctx_target, self.vocab, num_drafts as i32) };
            } else {
                // Prune rejected positions from KV caches
                let rejected_start = n_cur + 1 + accept_len as i32;
                unsafe {
                    let mem_tgt = llama_get_memory(self.ctx_target);
                    let mem_dft = llama_get_memory(self.ctx_draft);
                    llama_memory_seq_rm(mem_tgt, 0, rejected_start, -1);
                    llama_memory_seq_rm(mem_dft, 0, rejected_start, -1);
                }
            }

            n_cur += 1 + accept_len as i32;
            current_token = next_token;
        }

        let report = profiler.stop();
        let memory = self.memory_profile();

        self.last_stats = SpeculativeStats {
            total_generated: generated_count,
            total_drafted: drafted_count,
            total_accepted: accepted_count,
            acceptance_rate: if drafted_count > 0 {
                (accepted_count as f64 / drafted_count as f64) * 100.0
            } else {
                0.0
            },
        };

        Ok(GenerationOutput {
            text: output_text,
            generated_tokens: generated_count,
            report,
            memory,
        })
    }
}
