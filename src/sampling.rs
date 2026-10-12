//! Token sampling backed by llama.cpp's native sampler chain.
//!
//! Replaces the hand-rolled scalar argmax over the full vocabulary and makes
//! `GenerationParams` (temperature, top-p, seed) actually take effect in the raw FFI engines.

use llama_cpp_sys_2::{
    llama_context, llama_sampler, llama_sampler_chain_add, llama_sampler_chain_default_params,
    llama_sampler_chain_init, llama_sampler_free, llama_sampler_init_dist,
    llama_sampler_init_greedy, llama_sampler_init_temp, llama_sampler_init_top_k,
    llama_sampler_init_top_p, llama_sampler_sample, llama_token,
};

use crate::engine::GenerationParams;

/// Temperatures at or below this threshold are treated as deterministic greedy decoding.
const GREEDY_TEMP_EPSILON: f32 = 0.001;
/// Candidate pool cap applied before nucleus filtering to keep sampling cheap on large vocabularies.
const TOP_K: i32 = 40;

/// Owning wrapper around a `llama_sampler` chain. Freed on drop.
pub struct SamplerChain {
    ptr: *mut llama_sampler,
    greedy: bool,
}

// The chain is only ever used from the thread that owns the engine.
unsafe impl Send for SamplerChain {}

impl SamplerChain {
    /// Builds a chain from generation hyperparameters.
    ///
    /// * `temperature <= 0.001` -> pure greedy (deterministic; used by benchmarks and the tuner).
    /// * otherwise -> top-k -> top-p -> temperature -> seeded distribution sampling.
    pub fn from_params(params: &GenerationParams) -> Self {
        unsafe {
            let chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            let greedy = params.temperature <= GREEDY_TEMP_EPSILON;
            if greedy {
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            } else {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(TOP_K));
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p.clamp(0.0, 1.0), 1));
                llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
                llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed));
            }
            Self { ptr: chain, greedy }
        }
    }

    /// True when this chain decodes deterministically.
    pub fn is_greedy(&self) -> bool {
        self.greedy
    }

    /// Samples the next token from the logits at batch index `idx` (-1 = last decoded token)
    /// and records it in the sampler state.
    ///
    /// # Safety
    /// `ctx` must be a valid context whose most recent decode produced logits at `idx`.
    pub unsafe fn sample(&mut self, ctx: *mut llama_context, idx: i32) -> llama_token {
        llama_sampler_sample(self.ptr, ctx, idx)
    }
}

impl Drop for SamplerChain {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_sampler_free(self.ptr) };
        }
    }
}
