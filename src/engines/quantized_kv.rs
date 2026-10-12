//! Dedicated engine for evaluating KV cache quantization (FP16 vs Q8_0 vs Q4_0).
//!
//! Controls the hardware KV cache precision via `cparams.type_k` and `cparams.type_v`,
//! directly evaluating the impact on VRAM consumption per token and throughput.

use anyhow::Result;
use crate::engine::{EngineConfig, GenerationOutput, GenerationParams, KvPrecision, ModelEngine, ModelMemoryProfile};
use crate::engines::raw_ffi::RawFfiEngine;

/// Engine that explicitly configures and benchmarks KV cache quantization.
pub struct QuantizedKvEngine {
    inner: RawFfiEngine,
    precision: KvPrecision,
}

impl QuantizedKvEngine {
    /// Creates a new engine configured with the specified KV cache quantization.
    pub fn new(mut config: EngineConfig, precision: KvPrecision) -> Result<Self> {
        config.kv_precision = precision;
        let inner = RawFfiEngine::new(config)?;
        Ok(Self { inner, precision })
    }

    pub fn precision(&self) -> KvPrecision {
        self.precision
    }

    pub fn config(&self) -> &EngineConfig {
        self.inner.config()
    }
}

impl ModelEngine for QuantizedKvEngine {
    fn name(&self) -> &'static str {
        match self.precision {
            KvPrecision::F16 => "Hardware C-ABI (FP16 KV Cache)",
            KvPrecision::Q8_0 => "Hardware C-ABI (Q8_0 Quantized KV Cache)",
            KvPrecision::Q4_0 => "Hardware C-ABI (Q4_0 Quantized KV Cache)",
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
}
