use std::path::PathBuf;
use std::sync::OnceLock;
use anyhow::Result;
use llama_cpp_2::llama_backend::LlamaBackend;

use crate::benchmark::BenchmarkReport;

static GLOBAL_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

/// Retrieve or initialize the process-wide LlamaBackend singleton.
pub fn global_backend() -> &'static LlamaBackend {
    GLOBAL_BACKEND.get_or_init(|| {
        LlamaBackend::init().expect("Failed to initialize LlamaBackend")
    })
}

/// Suppress all internal llama.cpp and ggml logging to keep stdout clean.
pub fn silence_llama_logs() {
    unsafe extern "C" fn null_log_cb(
        _level: llama_cpp_sys_2::ggml_log_level,
        _text: *const ::std::os::raw::c_char,
        _user_data: *mut ::std::os::raw::c_void,
    ) {}
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(null_log_cb), std::ptr::null_mut());
    }
}

/// KV Cache precision quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    F16,
    Q8_0,
    Q4_0,
}

impl Default for KvPrecision {
    fn default() -> Self {
        Self::F16
    }
}

impl KvPrecision {
    pub fn label(&self) -> &'static str {
        match self {
            Self::F16 => "FP16 (16-bit)",
            Self::Q8_0 => "Q8_0 (8-bit)",
            Self::Q4_0 => "Q4_0 (4-bit)",
        }
    }

    pub fn bytes_per_element(&self) -> f64 {
        match self {
            Self::F16 => 2.0,
            Self::Q8_0 => 1.0,
            Self::Q4_0 => 0.5,
        }
    }

    pub fn to_ggml_type(&self) -> llama_cpp_sys_2::ggml_type {
        match self {
            Self::F16 => llama_cpp_sys_2::GGML_TYPE_F16,
            Self::Q8_0 => llama_cpp_sys_2::GGML_TYPE_Q8_0,
            Self::Q4_0 => llama_cpp_sys_2::GGML_TYPE_Q4_0,
        }
    }

    pub fn to_kv_cache_type(&self) -> llama_cpp_2::context::params::KvCacheType {
        match self {
            Self::F16 => llama_cpp_2::context::params::KvCacheType::F16,
            Self::Q8_0 => llama_cpp_2::context::params::KvCacheType::Q8_0,
            Self::Q4_0 => llama_cpp_2::context::params::KvCacheType::Q4_0,
        }
    }
}

/// Engine initialization configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to GGUF model file
    pub model_path: PathBuf,

    /// Hardware backend device index (e.g., Some(1) for RTX 3050 Ti, None for default/CPU)
    pub device_index: Option<usize>,

    /// Number of transformer layers to offload to GPU (e.g. 999 for full offload, 0 for CPU)
    pub n_gpu_layers: u32,

    /// Context size in tokens
    pub ctx_size: u32,

    /// KV Cache data precision (FP16, Q8_0, Q4_0)
    pub kv_precision: KvPrecision,

    /// Thread count for CPU compute
    pub n_threads: Option<i32>,
}

impl EngineConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            device_index: Some(1), // Default to RTX 3050 Ti on this machine
            n_gpu_layers: 999,
            ctx_size: 2048,
            kv_precision: KvPrecision::F16,
            n_threads: None,
        }
    }

    pub fn with_device(mut self, device: Option<usize>) -> Self {
        self.device_index = device;
        self
    }

    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.n_gpu_layers = layers;
        self
    }

    pub fn with_ctx_size(mut self, ctx_size: u32) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub fn with_kv_precision(mut self, precision: KvPrecision) -> Self {
        self.kv_precision = precision;
        self
    }
}

/// Generation runtime hyperparameters.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
        }
    }
}

/// Detailed memory profile of the model and its KV cache.
#[derive(Debug, Clone)]
pub struct ModelMemoryProfile {
    pub model_weights_bytes: u64,
    pub total_params: u64,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub kv_precision: KvPrecision,
    pub kv_bytes_per_token: usize,
    pub ctx_size: u32,
    pub total_kv_cache_bytes: usize,
    pub total_memory_bytes: u64,
}

impl ModelMemoryProfile {
    pub fn new(
        weights_bytes: u64,
        n_params: u64,
        n_layers: u32,
        n_heads: u32,
        n_kv_heads: u32,
        n_embd: u32,
        ctx_size: u32,
        precision: KvPrecision,
    ) -> Self {
        let head_dim = if n_heads > 0 { n_embd / n_heads } else { 0 };
        let bytes_per_element = precision.bytes_per_element();
        let kv_bytes_per_token = (2.0 * n_layers as f64 * n_kv_heads as f64 * head_dim as f64 * bytes_per_element).ceil() as usize;
        let total_kv_cache_bytes = kv_bytes_per_token * ctx_size as usize;
        let total_memory_bytes = weights_bytes + total_kv_cache_bytes as u64;

        Self {
            model_weights_bytes: weights_bytes,
            total_params: n_params,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_precision: precision,
            kv_bytes_per_token,
            ctx_size,
            total_kv_cache_bytes,
            total_memory_bytes,
        }
    }

    /// KV cache memory in KiB per token
    pub fn kv_kib_per_token(&self) -> f64 {
        self.kv_bytes_per_token as f64 / 1024.0
    }

    /// Model weights size in MiB
    pub fn weights_mib(&self) -> f64 {
        self.model_weights_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Total KV cache capacity in MiB
    pub fn kv_cache_mib(&self) -> f64 {
        self.total_kv_cache_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Total working memory (Weights + KV Cache) in MiB
    pub fn total_memory_mib(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl std::fmt::Display for ModelMemoryProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let billions = self.total_params as f64 / 1_000_000_000.0;
        writeln!(f, "┌──────────────────────────────────────────────┬──────────────────┐")?;
        writeln!(f, "│ Memory / Architecture Metric                 │ Value            │")?;
        writeln!(f, "├──────────────────────────────────────────────┼──────────────────┤")?;
        writeln!(f, "│ Model Parameters                             │ {:>13.2} B │", billions)?;
        writeln!(f, "│ Base RAM/VRAM for Weights                    │ {:>10.2} MiB │", self.weights_mib())?;
        writeln!(f, "│ Transformer Depth (Layers)                   │ {:>10} layers │", self.n_layers)?;
        writeln!(f, "│ Attention Heads (Query / KV)                 │ {:>5} Q / {:>2} KV │", self.n_heads, self.n_kv_heads)?;
        writeln!(f, "│ Head Dimension (d_head)                      │ {:>12} dim │", self.head_dim)?;
        writeln!(f, "├──────────────────────────────────────────────┼──────────────────┤")?;
        writeln!(f, "│ KV Cache Precision Format                    │ {:>16} │", self.kv_precision.label())?;
        writeln!(f, "│ \x1b[1;36mKV Cache Footprint (Per Token)               │ {:>10.2} KiB/tok\x1b[0m │", self.kv_kib_per_token())?;
        writeln!(f, "│ Context Window Configured                    │ {:>10} tokens │", self.ctx_size)?;
        writeln!(f, "│ \x1b[1;36mTotal KV Cache RAM/VRAM                      │ {:>10.2} MiB\x1b[0m │", self.kv_cache_mib())?;
        writeln!(f, "├──────────────────────────────────────────────┼──────────────────┤")?;
        writeln!(f, "│ \x1b[1;32mTotal Base Working Set (Weights + KV Cache)  │ {:>10.2} MiB\x1b[0m │", self.total_memory_mib())?;
        write!(f,   "└──────────────────────────────────────────────┴──────────────────┘")
    }
}

/// Result of a completed generation run.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub generated_tokens: usize,
    pub report: BenchmarkReport,
    pub memory: ModelMemoryProfile,
}

/// Streaming UTF-8 decoder buffering partial multi-byte code units across token boundaries.
#[derive(Debug, Default, Clone)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::with_capacity(16),
        }
    }

    /// Feed raw token piece bytes, returning any fully formed valid UTF-8 string chunk.
    pub fn feed(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(valid) => {
                let s = valid.to_string();
                self.pending.clear();
                Some(s)
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    let s = unsafe { std::str::from_utf8_unchecked(&self.pending[..valid_up_to]) }.to_string();
                    self.pending.drain(..valid_up_to);
                    Some(s)
                } else if e.error_len().is_some() {
                    let s = String::from_utf8_lossy(&self.pending).to_string();
                    self.pending.clear();
                    Some(s)
                } else {
                    None
                }
            }
        }
    }

    /// Flush any remaining buffered bytes at end of generation.
    pub fn flush(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            None
        } else {
            let s = String::from_utf8_lossy(&self.pending).to_string();
            self.pending.clear();
            Some(s)
        }
    }

    /// Clear buffered bytes.
    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

/// Generalized trait for LLM inference engines.
/// Allows swapping between high-level safe engines, low-level raw FFI engines,
/// and different execution backends.
pub trait ModelEngine {
    /// Name and brief description of this engine implementation
    fn name(&self) -> &'static str;

    /// Computing device name (e.g., "NVIDIA GeForce RTX 3050 Ti Laptop GPU", "CPU")
    fn device_name(&self) -> String;

    /// Query the exact memory profile (base weights RAM, KV cache KB/token, total VRAM)
    fn memory_profile(&self) -> ModelMemoryProfile;

    /// Run inference with an optional streaming callback for each token piece.
    /// Returns the full text, speed report, and memory footprint.
    fn generate(
        &mut self,
        prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput>;

    /// Whether this engine supports stateful incremental continuation across turns.
    fn is_stateful(&self) -> bool {
        false
    }

    /// Current token position in the active KV sequence.
    fn n_past(&self) -> i32 {
        0
    }

    /// Reset any active conversation context and clear the KV cache.
    fn reset_context(&mut self) -> Result<()> {
        Ok(())
    }

    /// Generate continuation from the active KV cache, prefilling only the new suffix prompt.
    fn generate_incremental(
        &mut self,
        suffix_prompt: &str,
        params: &GenerationParams,
        on_token: Option<&mut dyn FnMut(&str) -> bool>,
    ) -> Result<GenerationOutput> {
        self.generate(suffix_prompt, params, on_token)
    }

    /// Programmatic benchmark run without IO/printing overhead.
    fn benchmark(&mut self, prompt: &str, n_tokens: usize) -> Result<BenchmarkReport> {
        let params = GenerationParams {
            max_tokens: n_tokens,
            temperature: 0.0, // greedy for deterministic benchmarking
            ..Default::default()
        };
        let out = self.generate(prompt, &params, None)?;
        Ok(out.report)
    }
}

#[cfg(test)]
mod utf8_decoder_tests {
    use super::Utf8StreamDecoder;

    #[test]
    fn passes_through_ascii() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.feed(b"hello").as_deref(), Some("hello"));
        assert_eq!(d.flush(), None);
    }

    #[test]
    fn reassembles_split_multibyte_codepoint() {
        // U+1F680 (rocket) is 4 bytes: F0 9F 9A 80, split across three tokens.
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.feed(&[0xF0, 0x9F]), None);
        assert_eq!(d.feed(&[0x9A]), None);
        assert_eq!(d.feed(&[0x80]).as_deref(), Some("\u{1F680}"));
        assert_eq!(d.flush(), None);
    }

    #[test]
    fn emits_valid_prefix_and_keeps_partial_tail() {
        // "a" followed by the first byte of a 2-byte sequence (U+00E9 = C3 A9).
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.feed(&[b'a', 0xC3]).as_deref(), Some("a"));
        assert_eq!(d.feed(&[0xA9]).as_deref(), Some("\u{e9}"));
    }

    #[test]
    fn flush_recovers_dangling_bytes_lossily() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.feed(&[0xE2, 0x82]), None);
        assert_eq!(d.flush().as_deref(), Some("\u{FFFD}"));
        assert_eq!(d.flush(), None);
    }

    #[test]
    fn invalid_bytes_do_not_stall_the_stream() {
        let mut d = Utf8StreamDecoder::new();
        assert!(d.feed(&[0xFF]).is_some());
        assert_eq!(d.feed(b"ok").as_deref(), Some("ok"));
    }
}
