# Kortex (`kortex`)

**Hardware-Accelerated Edge LLM Inference Kernel, Bounded $O(1)$ Attention Streaming, Mathematical KV-Cache Compression & Cryptographic Model Auto-Tuning in Rust**

[![Ecosystem](https://img.shields.io/badge/Ecosystem-Larvance%20%2F%20Aerovex-0ea5e9?style=flat-square)](https://github.com/larvance)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Rust%202021-orange?style=flat-square)](https://www.rust-lang.org/)
[![Acceleration](https://img.shields.io/badge/Backend-Vulkan%20GPU-red?style=flat-square)](#)

---

## Overview

**Kortex** is a hardened, ultra-low-latency edge Large Language Model (LLM) and Small Language Model (SLM) inference runtime engineered in pure native Rust with low-level C-ABI Vulkan GPU acceleration.

Designed specifically for resource-constrained hardware—such as autonomous robotics companion computers (NVIDIA Jetson, Raspberry Pi), edge edge-nodes, and laptops with limited VRAM (e.g., 4GB RTX 3050 Ti)—Kortex eliminates out-of-memory context cliffs, dramatically reduces KV-cache memory footprints through 17 mathematical compression paradigms, and provides an Ollama-style interactive terminal REPL with sub-millisecond token streaming.

```
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                     KORTEX INTERACTIVE CHAT REPL                          ║
 ║          High-Throughput Edge Inference Engine with Hash Optimization     ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Key Breakthroughs & Architecture

### 1. Bounded $O(1)$ Attention Sinks & Zero Context Cliffs
Standard LLM inference suffers from the **context cliff problem**: as conversation history grows, KV cache memory scales linearly ($O(N)$) until VRAM is exhausted, triggering an abrupt `OOM` crash.
* **Attention Sink Preservation:** Anchors the initial $k$ attention sink tokens ($k=4$) containing high-magnitude initial attention mass, preventing activation divergence.
* **Sliding Window Token Ring:** Maintains a rolling window of recent tokens ($W=256$) via dynamic memory sequence relocation (`llama_memory_seq_rm` and `llama_memory_seq_add`).
* **Amortized Block Eviction:** Triggers evictions in batched blocks ($k_{\text{evict}}=16$), cutting KV cache fragmentation and GPU sync stalls to near zero.
* **Infinite Multi-Turn Dialogue:** Conversations can stream millions of tokens indefinitely within a guaranteed, strictly bounded VRAM footprint.

### 2. 17 KV-Cache Mathematical Compression Paradigms
Kortex implements and benchmarks 17 distinct compression and approximation algorithms across linear algebra, signal processing, and differential topology:

| Category | Paradigm | Mathematical Mechanism | Key Metric / Result |
| :--- | :--- | :--- | :--- |
| **Strictly Lossless** | **Zero-Copy Prefix Sharing** | Pointer redirection via `llama_memory_seq_cp` | **0.000% error**, 0.032 ms fork, 0 B extra VRAM |
| **Near-Lossless** | **Ada-SVD (Entropy-Adaptive)** | Dynamic Truncated SVD rank $r \in [4, 24]$ | **$10.24\times$ compression**, $0.0012$ relative error |
| **Hybrid Linear** | **Hybrid SVD-Q** | Rank-$r$ projection + 2-bit residual quantization | **$13.44\times$ compression**, $0.0020$ relative error |
| **Cross-Layer** | **ResEntangle-KV** | Base layer anchors + low-rank inter-layer deltas | **$3.33\times$ compression**, score jumped from 7.1 to 1,110.9 |
| **Interpolative** | **KnotKV (B-Splines)** | Order-3 Cox-de Boor B-spline control point curves | **$8.13\times$ compression**, 202 tok/s on sub-1B models |
| **Multiresolution** | **Wavelet-KV** | Daubechies D4 discrete wavelet decomposition | **$4.00\times$ compression**, high-frequency thresholding |
| **Frequency** | **Discrete Cosine (DCT)** | Chebyshev energy compaction in spectral domain | Low-pass coefficient truncation |
| **Spreading** | **OPDM-RoPE** | Orthogonal Walsh-Hadamard code-division multiplexing | Spreads token features across sub-bands |
| **Token Pruning** | **Heavy-Hitter (SnapKV/H2O)** | Dynamic attention-weight accumulation masks | Retains top-$k$ attention-dense token heads |
| **Hardware Native** | **Quantized KV (Q8_0 / Q4_0)** | Hardware-accelerated 8-bit and 4-bit block quantization | $2\times$ to $4\times$ footprint reduction |

### 3. Cryptographic SHA-256 Model Profile Auto-Tuning
Kortex computes a fast cryptographic SHA-256 fingerprint over GGUF metadata headers and tensor configurations in `< 15 ms`.
* **Persistent Cache Database:** Indexes models in `~/.kortex/model_profile_cache.json`.
* **Zero-Calibrate Boot:** Instantly matches any downloaded or moved model file to its empirically proven champion runner configuration (precision, sink count, window size, chunk stride) without re-running expensive tuning passes.

### 4. Native Ollama-Style Interactive Terminal REPL
* **Sub-Millisecond Token Streaming:** Low-overhead typewriter effect with autovectorized greedy argmax token selection.
* **Auto-Detected Chat Templates:** Native support for both **ChatML** (`<|im_start|>`, `<|im_end|>`) and **Llama-3** (`<|start_header_id|>`, `<|eot_id|>`).
* **Interactive Slash Commands:**
  * `/help` — Display command reference and active keybindings.
  * `/info` — View model architecture, tensor parameters, and KV profile.
  * `/stats` — Toggle live generation metrics (tok/s, TTFT, P90 latency, jitter $\sigma$).
  * `/clear` — Reset conversation context and rewind attention sinks.
  * `/cache` — Inspect SHA-256 profile cache status and optimal runner parameters.
  * `/models` — Scan and list all local GGUF models with detected architectures.
  * `/exit` — Gracefully terminate the inference session.

---

## Empirical Benchmark Matrix

Tested on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)** + 16GB Host DDR5:

| Model Architecture | Parameter Count | Base Weights | KV KiB/tok | Champion Runner | Throughput | Compression Ratio | Efficiency Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 2.5 0.5B** | 0.63B | 469 MB | 12.00 | **KnotKV (B-Spline)** | **202.4 tok/s** | $8.13\times$ | **2,041.8** |
| **Llama 3.2 1B** | 1.24B | 771 MB | 32.00 | **KnotKV (B-Spline)** | **184.2 tok/s** | $8.13\times$ | **1,985.4** |
| **Qwen 2.5 1.5B** | 1.78B | 1.1 GB | 28.00 | **Ada-SVD ($r \in [4, 24]$)** | **145.8 tok/s** | $10.24\times$ | **1,839.8** |
| **Qwen 2.5 3B** | 3.40B | 2.0 GB | 36.00 | **Ada-SVD ($r \in [4, 24]$)** | **92.4 tok/s** | $10.24\times$ | **1,210.4** |
| **Qwen 2.5 7B (Q2_K)** | 7.62B | 2.9 GB | 56.00 | **Ada-SVD ($r \in [4, 24]$)** | **38.6 tok/s** | $10.24\times$ | **459.0** |

> **Key Theoretical Insight:** Sub-1B edge models exhibit smooth activation manifolds, making **B-Spline curve fitting (KnotKV)** optimal. Deep models ($\ge 1.5\text{B}$) develop low-rank semantic clustering across attention heads, making **Entropy-Adaptive SVD (Ada-SVD)** the dominant near-lossless compression paradigm.

---

## Quickstart

### Prerequisites
* **Rust 1.75+** (`cargo`, `rustc`)
* **CMake & Clang / GCC** (for compiling backend kernels)
* **Vulkan SDK & Drivers** (NVIDIA, AMD, or Intel GPU with Vulkan support)

### Installation
```bash
# Clone the repository
git clone git@github.com:aerovexhq/kortex.git
cd kortex

# Build release binaries
cargo build --release --bins
```

### Starting the Interactive Chat REPL
```bash
# Run with any local GGUF model
cargo run --release --bin chat -- -m path/to/model.gguf

# Specify custom sampling parameters
cargo run --release --bin chat -- -m path/to/model.gguf --temp 0.7 --top-p 0.9 --max-tokens 512
```

### Running the Master Benchmark Suite
```bash
# Run the 17-method compression suite across all downloaded models
cargo run --release --bin master_compression_suite

# Run the automated tuner to profile a model and cache its champion runner
cargo run --release --bin autotune -- -m path/to/model.gguf
```

---

## Project Structure

```
kortex/
├── Cargo.toml                  # Package manifest & Vulkan dependencies
├── src/
│   ├── lib.rs                  # Public crate exports
│   ├── engine.rs               # ModelEngine trait & memory profiling
│   ├── benchmark.rs            # Sub-50ns latency & throughput profiler
│   ├── model_cache.rs          # SHA-256 header hashing & cache lookup (~/.kortex/)
│   ├── autotune.rs             # Multi-dimensional efficiency optimizer
│   ├── engines/                # Inference execution backends
│   │   ├── raw_ffi.rs          # Direct C-ABI FFI Vulkan engine
│   │   ├── quantized_kv.rs     # Hardware Q8_0 / Q4_0 precision control
│   │   ├── sliding_window.rs   # StreamingLLM attention sink windowing
│   │   ├── hybrid_sink.rs      # Production Champion: Q8_0 + Sinks + Chunk Eviction
│   │   ├── speculative.rs      # Dual-model speculative draft-target decoder
│   │   └── cpu.rs              # Host DDR5 multi-threaded SIMD engine
│   ├── compression/            # 17 KV cache compression algorithms
│   │   ├── standard/           # Zero-Copy, SnapKV, ToMe, Residual Codec
│   │   └── innovative/         # Ada-SVD, KnotKV, WaveletKV, ResEntangle, OPDM-RoPE
│   └── bin/                    # Standalone CLI tools & REPL
│       ├── chat.rs             # Interactive Ollama-style terminal REPL
│       ├── master_compression_suite.rs # 17-method master evaluator
│       └── autotune.rs         # Automated runner tuning utility
└── LICENSE                     # MIT License
```

---

## License

Kortex is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
