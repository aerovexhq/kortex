//! KV Cache Compression & Quality Retention Research Module.
//!
//! Encompasses:
//! 1. Standard / Near-lossless methods:
//!    - Zero-Copy Prefix Sharing & Deduplication (llama_memory_seq_cp)
//!    - Radix-Trie Multi-Branch Lossless Sharing (Deep, Standard, Shallow)
//!    - Bit-Packed Lossless Delta Codec (Bitplane, RLE-Golomb)
//!    - Streaming rANS Lossless Codec (Laplace delta modeling)
//!    - SnapKV / Heavy Hitter Oracle (H2O)
//!    - Token Merging (ToMe-KV)
//!    - Cross-Layer Interleaved KV Sharing
//!    - Temporal Delta-Residual Prediction Codec
//!
//! 2. Creative & Evolved Boundary-Pushing Methods:
//!    - Innovation 1: Temporal RoPE Harmonic Superposition Folding
//!    - Innovation 1B: CDMA Orthogonal Phase Division Multiplexing (OPDM-KV)
//!    - Innovation 1C: Lie Group Matrix Exponential (LieRE-KV in so(d))
//!    - Innovation 2: Streaming SVD Rank Collapse (Dynamic Holographic Memory)
//!    - Innovation 2A: Ada-SVD (Entropy-Adaptive Rank Collapse)
//!    - Innovation 2B: Hybrid SVD-Q (Low-Rank SVD Core + 2-bit Quantized Residuals)
//!    - Innovation 2C: Turbo Hierarchical Ada-SVD (Extreme, Balanced, Precision)
//!    - Innovation 3: Polynomial Chebyshev Spline Memory (KV Curve Fitting)
//!    - Innovation 3B: Knot-Adaptive B-Spline Memory (KnotKV)
//!    - Innovation 3C: Hyper-Knot B-Spline (Aggressive, Balanced, Precision)
//!    - Innovation 4: Walsh-Hadamard Orthogonal Layer Entanglement
//!    - Innovation 4B: Cross-Layer Residual Low-Rank Entanglement (ResEntangle-KV)
//!    - Innovation 4C: ResEntangle-KV V2 (8-Layer Grassmannian Subspace)
//!    - Innovation 5: Discrete Cosine Transform (DCT) Frequency Sparsification
//!    - Innovation 5B: Multi-Resolution Wavelet Packet Decomposition (WaveletKV - D4)
//!    - Innovation 5C: Symlet-4 Multiresolution WaveletKV
//!    - Innovation 5D: Coiflet-2 Multiresolution WaveletKV (Dual Vanishing Moments)
//!    - Innovation 5E: Biorthogonal CDF 9/7 WaveletKV (Linear Phase)
//!    - Innovation 6: Hyperspherical Geodesic Quantization (SphericalGeoKV)
//!    - Innovation 7: Grassmannian Geodesic Flow (GGF-KV)
//!    - Innovation 8: Normalizing Flow Vector Quantization (NF-VQ-KV)
//!    - Innovation 9: State-Space Model HiPPO Key Projection (SSM-HiPPO-KV)
//!    - Innovation 10: Sinkhorn-Wasserstein Barycentric KV with E8 Lattice
//!    - Innovation 11: Spectral Koopman Operator (Echo-KV)
//!    - Innovation 12: PyramidKV + RazorAttention Compensation

pub mod metrics;
pub mod standard;
pub mod innovative;

pub use metrics::*;
pub use standard::zero_copy::*;
pub use standard::radix_trie::*;
pub use standard::lossless_delta::*;
pub use standard::rans_codec::*;
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
pub use innovative::coiflet_wavelet::*;
pub use innovative::biorthogonal_wavelet::*;
pub use innovative::residual_entanglement_v2::*;
pub use innovative::spherical_geo::*;
pub use innovative::grassmannian_flow::*;
pub use innovative::normalizing_flow_vq::*;
pub use innovative::ssm_hippo::*;
pub use innovative::lie_group_rope::*;
pub use innovative::sinkhorn_wasserstein::*;
pub use innovative::spectral_koopman::*;
pub use innovative::pyramid_razor::*;
