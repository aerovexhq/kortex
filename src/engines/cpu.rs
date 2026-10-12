//! Pure CPU host execution engine.
//!
//! Benchmarks LLM inference without GPU offloading (`n_gpu_layers = 0`), utilizing
//! multi-threaded CPU SIMD execution across AMD Ryzen 5 6600H cores and DDR5 host RAM.

use anyhow::Result;
use crate::engine::{EngineConfig, GenerationOutput, GenerationParams, ModelEngine, ModelMemoryProfile};
use crate::engines::raw_ffi::RawFfiEngine;

/// Pure CPU inference engine executing in host system DDR5 RAM.
pub struct CpuEngine {
    inner: RawFfiEngine,
    threads_used: u32,
}

impl CpuEngine {
    /// Creates a CPU engine, forcing 0 GPU layers and configuring thread parallelism.
    pub fn new(mut config: EngineConfig) -> Result<Self> {
        config.n_gpu_layers = 0;
        config.device_index = None;
        let threads = config.n_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get() as i32)
                .unwrap_or(6)
        });
        config.n_threads = Some(threads);

        let inner = RawFfiEngine::new(config)?;
        Ok(Self {
            inner,
            threads_used: threads as u32,
        })
    }

    pub fn threads_used(&self) -> u32 {
        self.threads_used
    }

    pub fn config(&self) -> &EngineConfig {
        self.inner.config()
    }
}

impl ModelEngine for CpuEngine {
    fn name(&self) -> &'static str {
        "Pure CPU Engine (Host DDR5 RAM, 0 GPU Offload)"
    }

    fn device_name(&self) -> String {
        format!("AMD Ryzen 5 6600H CPU ({} SIMD Threads)", self.threads_used)
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
}
