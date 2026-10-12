//! Model Profile & Champion Runner Cache System.
//!
//! Hashes GGUF model files via cryptographic SHA-256 over header metadata and file specs,
//! caching the optimal runner configuration and efficiency benchmarks.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};


/// Runner algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunnerType {
    /// Hybrid Quantized Attention Sink (Q8_0 KV + Sinks + Window + Chunk Eviction)
    HybridSink,
    /// Sliding Window Attention Sinks (FP16 KV + Sinks + Window)
    SlidingWindow,
    /// Low-Level Hardware C-ABI Raw FFI
    RawFfi,
    /// Quantized Hardware KV Cache (FP16 / Q8_0 / Q4_0)
    QuantizedKv,
    /// Pure Host DDR5 CPU SIMD Engine
    Cpu,
}

impl RunnerType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::HybridSink => "Hybrid Quantized Attention Sink (O(1) Bounded VRAM)",
            Self::SlidingWindow => "Sliding Window Attention Sinks (FP16)",
            Self::RawFfi => "Hardware C-ABI Direct FFI (FP16)",
            Self::QuantizedKv => "Quantized Hardware KV Cache",
            Self::Cpu => "Host CPU DDR5 Multi-Threaded",
        }
    }
}

/// Cached optimal runner configuration for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalRunnerProfile {
    pub runner_type: RunnerType,
    pub kv_precision: String,
    pub sink_tokens: i32,
    pub window_size: i32,
    pub evict_chunk: i32,
    pub efficiency_score: f64,
    pub expected_tok_s: f64,
    pub notes: String,
}

impl Default for OptimalRunnerProfile {
    fn default() -> Self {
        Self {
            runner_type: RunnerType::HybridSink,
            kv_precision: "Q8_0".to_string(),
            sink_tokens: 4,
            window_size: 256,
            evict_chunk: 16,
            efficiency_score: 1396.9,
            expected_tok_s: 180.0,
            notes: "Production Champion: 4 Sinks, 256 Window, Chunk 16, Q8_0 Precision".to_string(),
        }
    }
}

/// Cached metadata for a hashed model file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheEntry {
    pub model_path: String,
    pub file_hash: String,
    pub file_size_bytes: u64,
    pub model_architecture: String,
    pub optimal_runner: OptimalRunnerProfile,
    pub last_updated: String,
}

/// Persistent model cache manager.
pub struct ModelCache {
    cache_path: PathBuf,
    entries: HashMap<String, ModelCacheEntry>,
}

impl ModelCache {
    /// Resolve default cache path at ~/.kortex/model_profile_cache.json
    pub fn default_cache_path() -> PathBuf {
        if let Ok(home) = std::env::var("HOME") {
            PathBuf::from(home).join(".kortex").join("model_profile_cache.json")
        } else {
            PathBuf::from("analysis/model_profile_cache.json")
        }
    }

    /// Load or initialize the model cache.
    /// Prefers ~/.kortex/model_profile_cache.json, with fallback to analysis/ or aerox/.
    pub fn load() -> Self {
        let home_cache = Self::default_cache_path();

        let candidate_paths = [
            home_cache.clone(),
            PathBuf::from("analysis/model_profile_cache.json"),
            PathBuf::from("aerox/model_profile_cache.json"),
            PathBuf::from("../../analysis/modules/kortex/model_profile_cache.json"),
            PathBuf::from("model_profile_cache.json"),
        ];

        let mut entries = HashMap::new();
        let mut loaded_from = None;

        for path in &candidate_paths {
            if path.exists() {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(parsed) = serde_json::from_str::<HashMap<String, ModelCacheEntry>>(&content) {
                        entries = parsed;
                        loaded_from = Some(path.clone());
                        break;
                    }
                }
            }
        }

        let cache = Self {
            cache_path: home_cache,
            entries,
        };

        // If loaded from a fallback and home cache doesn't exist yet, persist to ~/.kortex/
        if loaded_from.is_some() && !cache.cache_path.exists() {
            let _ = cache.save();
        }

        cache
    }

    /// Save cache to disk.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.entries)?;
        fs::write(&self.cache_path, json)?;
        Ok(())
    }

    /// Compute cryptographic SHA-256 hash of a GGUF model file.
    /// Fast header + footer + size hash takes < 15ms even on 7GB files.
    /// Uses a fixed 64 KB scratch buffer to avoid allocating 8 MB on the heap.
    pub fn hash_model_file<P: AsRef<Path>>(path: P) -> Result<String> {
        let path = path.as_ref();
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open model file for hashing: {:?}", path))?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();

        let mut hasher = Sha256::new();
        hasher.update(file_len.to_le_bytes());

        // Stream the first 8 MB in 64 KB chunks
        let sample_size = (8 * 1024 * 1024).min(file_len);
        let mut scratch = [0u8; 64 * 1024];
        let mut bytes_left = sample_size;

        while bytes_left > 0 {
            let to_read = (bytes_left as usize).min(scratch.len());
            file.read_exact(&mut scratch[..to_read])?;
            hasher.update(&scratch[..to_read]);
            bytes_left -= to_read as u64;
        }

        // Read last 64 KB (footer/trailing alignment)
        if file_len > (8 * 1024 * 1024 + 65536) {
            file.seek(SeekFrom::End(-65536))?;
            file.read_exact(&mut scratch[..65536])?;
            hasher.update(&scratch[..65536]);
        }

        let hash_result = hasher.finalize();
        let hex_str: String = hash_result.iter().map(|b| format!("{:02x}", b)).collect();
        Ok(hex_str)
    }

    /// Retrieve or compute optimal runner profile for a model.
    pub fn get_or_register<P: AsRef<Path>>(&mut self, path: P) -> Result<(String, ModelCacheEntry)> {
        let path = path.as_ref();
        let hash = Self::hash_model_file(path)?;
        let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();

        if let Some(entry) = self.entries.get(&hash) {
            return Ok((hash, entry.clone()));
        }

        // Determine optimal runner based on architecture and model size
        let file_size = fs::metadata(path)?.len();
        let optimal_runner = Self::determine_optimal_runner(&file_name, file_size);

        let entry = ModelCacheEntry {
            model_path: path.to_string_lossy().to_string(),
            file_hash: hash.clone(),
            file_size_bytes: file_size,
            model_architecture: if file_name.contains("llama") { "llama".to_string() } else { "qwen2".to_string() },
            optimal_runner,
            last_updated: chrono_now_iso(),
        };

        self.entries.insert(hash.clone(), entry.clone());
        let _ = self.save();

        Ok((hash, entry))
    }

    /// Automatically select the champion runner based on model file characteristics.
    fn determine_optimal_runner(file_name: &str, file_size: u64) -> OptimalRunnerProfile {
        let file_lower = file_name.to_lowercase();
        let size_mb = file_size as f64 / (1024.0 * 1024.0);

        if file_lower.contains("0.5b") {
            OptimalRunnerProfile {
                runner_type: RunnerType::HybridSink,
                kv_precision: "Q8_0".to_string(),
                sink_tokens: 4,
                window_size: 256,
                evict_chunk: 16,
                efficiency_score: 1396.9,
                expected_tok_s: 224.1,
                notes: "Champion: 0.5B Sweet Spot (Q8_0 KV, k=16 Amortized Block Eviction)".to_string(),
            }
        } else if file_lower.contains("llama") || file_lower.contains("1b") {
            OptimalRunnerProfile {
                runner_type: RunnerType::HybridSink,
                kv_precision: "Q8_0".to_string(),
                sink_tokens: 4,
                window_size: 256,
                evict_chunk: 16,
                efficiency_score: 1403.0,
                expected_tok_s: 202.4,
                notes: "Champion: Llama 3.2 1B (8 KV Heads, Q8_0 KV, 4 Sinks, k=16)".to_string(),
            }
        } else if file_lower.contains("1.5b") {
            OptimalRunnerProfile {
                runner_type: RunnerType::HybridSink,
                kv_precision: "Q8_0".to_string(),
                sink_tokens: 4,
                window_size: 256,
                evict_chunk: 16,
                efficiency_score: 1839.8,
                expected_tok_s: 155.0,
                notes: "Champion: Qwen 1.5B (Adaptive SVD & Q8_0 Sinks, k=16)".to_string(),
            }
        } else if file_lower.contains("3b") {
            OptimalRunnerProfile {
                runner_type: RunnerType::HybridSink,
                kv_precision: "Q8_0".to_string(),
                sink_tokens: 4,
                window_size: 256,
                evict_chunk: 16,
                efficiency_score: 962.4,
                expected_tok_s: 85.0,
                notes: "Champion: Qwen 3B (Q8_0 Sinks, 2k Ctx, 76-85 tok/s)".to_string(),
            }
        } else if file_lower.contains("7b") || size_mb > 2500.0 {
            OptimalRunnerProfile {
                runner_type: RunnerType::HybridSink,
                kv_precision: "Q8_0".to_string(),
                sink_tokens: 4,
                window_size: 256,
                evict_chunk: 16,
                efficiency_score: 459.0,
                expected_tok_s: 32.8,
                notes: "Champion: Qwen 7B (Fully Offloaded in 4GB VRAM, Q8_0 Sinks)".to_string(),
            }
        } else {
            OptimalRunnerProfile::default()
        }
    }

    /// List all cached entries.
    pub fn all_entries(&self) -> &HashMap<String, ModelCacheEntry> {
        &self.entries
    }
}

fn chrono_now_iso() -> String {
    // Lightweight timestamp without heavy chrono crate
    let now = std::time::SystemTime::now();
    let since_epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("timestamp-secs-{}", since_epoch.as_secs())
}
