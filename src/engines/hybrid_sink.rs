//! Hybrid Quantized Attention Sink Engine.
//!
//! Fuses 8-bit quantized KV cache (`type_k = Q8_0`, `type_v = Q8_0`) with low-level
//! attention-sink eviction (`llama_memory_seq_rm` + `llama_memory_seq_add`).
//!
//! Delivers ultra-low O(1) memory footprint (e.g., 0.40 MiB KV cache for 68 tokens)
//! while allowing configurable chunk eviction (k = 1, 4, 8, 16, 32) to evaluate
//! the sweet spot for inter-token latency jitter (σ).

use anyhow::Result;
use crate::engine::{EngineConfig, GenerationOutput, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile};
use crate::engines::sliding_window::SlidingWindowEngine;

#[allow(dead_code)]
pub struct HybridSinkEngine {
    inner: SlidingWindowEngine,
    sink_tokens: i32,
    window_size: i32,
    evict_chunk: i32,
    precision: KvPrecision,
}

impl HybridSinkEngine {
    /// Creates a new hybrid engine fusing KV quantization and sliding-window attention sinks.
    pub fn new(
        mut config: EngineConfig,
        precision: KvPrecision,
        sink_tokens: i32,
        window_size: i32,
        evict_chunk: i32,
    ) -> Result<Self> {
        config.kv_precision = precision;
        let inner = SlidingWindowEngine::new(config, sink_tokens, window_size, evict_chunk)?;
        Ok(Self {
            inner,
            sink_tokens,
            window_size,
            evict_chunk,
            precision,
        })
    }

    /// Creates a hybrid sink engine sharing an already loaded model handle.
    pub fn from_shared_model(
        shared: std::sync::Arc<crate::engine::LlamaModelHandle>,
        mut config: EngineConfig,
        precision: KvPrecision,
        sink_tokens: i32,
        window_size: i32,
        evict_chunk: i32,
    ) -> Result<Self> {
        config.kv_precision = precision;
        let inner = SlidingWindowEngine::from_shared_model(shared, config, sink_tokens, window_size, evict_chunk)?;
        Ok(Self {
            inner,
            sink_tokens,
            window_size,
            evict_chunk,
            precision,
        })
    }

    pub fn evict_chunk(&self) -> i32 {
        self.evict_chunk
    }

    pub fn precision(&self) -> KvPrecision {
        self.precision
    }
}

impl ModelEngine for HybridSinkEngine {
    fn name(&self) -> &'static str {
        match (self.precision, self.evict_chunk) {
            (KvPrecision::Q8_0, 1) => "Hybrid Q8_0 Sink (Chunk k=1 Continuous Shift)",
            (KvPrecision::Q8_0, 8) => "Hybrid Q8_0 Sink (Chunk k=8 Amortized Eviction)",
            (KvPrecision::Q8_0, 16) => "Hybrid Q8_0 Sink (Chunk k=16 Amortized Eviction)",
            (KvPrecision::Q8_0, 32) => "Hybrid Q8_0 Sink (Chunk k=32 Block Eviction)",
            _ => "Hybrid Quantized Attention Sink Engine",
        }
    }

    fn device_name(&self) -> String {
        self.inner.device_name()
    }

    fn memory_profile(&self) -> ModelMemoryProfile {
        self.inner.memory_profile()
    }

    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        self.inner.generate(prompt, params, on_token)
    }

    fn is_stateful(&self) -> bool {
        self.inner.is_stateful()
    }

    fn n_past(&self) -> i32 {
        self.inner.n_past()
    }

    fn reset_context(&mut self) -> Result<()> {
        self.inner.reset_context()
    }

    fn generate_incremental(
        &mut self,
        suffix_prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        self.inner.generate_incremental(suffix_prompt, params, on_token)
    }
}
