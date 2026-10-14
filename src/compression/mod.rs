//! KV Cache Compression & Quality Retention Research Module.
//!
//! Encompasses:
//! 1. Standard / Near-lossless methods:
//!    - Zero-Copy Prefix Sharing & Deduplication (llama_memory_seq_cp)
//!    - SnapKV / Heavy Hitter Oracle (H2O)
//!    - Vector-Cosine Token Merging (ToMe-KV)
//!    - Cross-Layer Interleaved KV Sharing
//!    - Temporal Delta-Residual Prediction Codec
//!
//! 2. Creative & Evolved Boundary-Pushing Methods:
//!    - Innovation 1: Temporal RoPE Harmonic Superposition Folding
//!    - Innovation 1B: CDMA Orthogonal Phase Division Multiplexing (OPDM-KV)
//!    - Innovation 2: Streaming SVD Rank Collapse (Dynamic Holographic Memory)
//!    - Innovation 2A: Ada-SVD (Entropy-Adaptive Rank Collapse)
//!    - Innovation 2B: Hybrid SVD-Q (Low-Rank SVD Core + 2-bit Quantized Residuals)
//!    - Innovation 3: Polynomial Chebyshev Spline Memory (KV Curve Fitting)
//!    - Innovation 3B: Knot-Adaptive B-Spline Memory (KnotKV)
//!    - Innovation 4: Walsh-Hadamard Orthogonal Layer Entanglement
//!    - Innovation 4B: Cross-Layer Residual Low-Rank Entanglement (ResEntangle-KV)
//!    - Innovation 5: Discrete Cosine Transform (DCT) Frequency Sparsification
//!    - Innovation 5B: Multi-Resolution Wavelet Packet Decomposition (WaveletKV)

pub mod metrics;
pub mod standard;
pub mod innovative;

pub use metrics::*;
pub use standard::zero_copy::*;
pub use standard::radix_trie::*;
pub use standard::lossless_delta::*;
pub use standard::heavy_hitter::*;
pub use standard::token_merge::*;
pub use standard::cross_layer::*;
pub use standard::residual_codec::*;

pub use innovative::harmonic_rope::*;
pub use innovative::svd_hologram::*;
pub use innovative::spline_kv::*;
pub use innovative::orthogonal_holography::*;
pub use innovative::frequency_dct::*;

pub use innovative::opdm_rope::*;
pub use innovative::residual_entanglement::*;
pub use innovative::ada_svd::*;
pub use innovative::hybrid_svd_q::*;
pub use innovative::wavelet_kv::*;
pub use innovative::knot_spline::*;
pub use innovative::hyper_knot::*;
pub use innovative::turbo_ada_svd::*;
pub use innovative::symlet_wavelet::*;
pub use innovative::residual_entanglement_v2::*;
pub use innovative::spherical_geo::*;
