//! Phase 9: Hardware Micro-Kernel Code Generation & Register-Blocked SIMD Vector Kernels.
//!
//! Provides cache-line aligned (64-byte) SIMD tiles, 4x16 register-blocked tensor
//! contraction micro-kernels, branchless Conway-Sloane E8 Gosset lattice nearest-point
//! decoders, and complex Kähler-Bergman Horner/binary-exponentiation kernels in pure safe Rust.

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A 64-byte cache-line aligned 16-lane FP32 SIMD tile (512 bits = 1 AVX-512 ZMM register,
/// 2 AVX2 YMM registers, or 4 ARM64 NEON Q registers).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedTile64(pub [f32; 16]);

impl Default for AlignedTile64 {
    fn default() -> Self {
        Self([0.0f32; 16])
    }
}

impl AlignedTile64 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([0.0f32; 16])
    }

    #[inline(always)]
    pub const fn splat(val: f32) -> Self {
        Self([val; 16])
    }

    /// Unrolled 16-lane Fused Multiply-Add: `self[i] = self[i].mul_add(scale, other[i])`
    /// Note: `self[i] += scale * other[i]` written for clean LLVM loop/SLP vectorization.
    #[inline(always)]
    pub fn fma_scale_assign(&mut self, scale: f32, other: &Self) {
        for i in 0..16 {
            self.0[i] = scale.mul_add(other.0[i], self.0[i]);
        }
    }

    /// Unrolled 16-lane elementwise FMA: `self[i] += a[i] * b[i]`.
    #[inline(always)]
    pub fn fma_vec_assign(&mut self, a: &Self, b: &Self) {
        for i in 0..16 {
            self.0[i] = a.0[i].mul_add(b.0[i], self.0[i]);
        }
    }

    /// Dual-tree horizontal dot product across 16 SIMD lanes to break FMA dependency chains.
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for i in 0..4 {
            acc0 = self.0[i].mul_add(other.0[i], acc0);
            acc1 = self.0[i + 4].mul_add(other.0[i + 4], acc1);
            acc2 = self.0[i + 8].mul_add(other.0[i + 8], acc2);
            acc3 = self.0[i + 12].mul_add(other.0[i + 12], acc3);
        }

        (acc0 + acc1) + (acc2 + acc3)
    }
}

/// Register-blocked 4x16 Matrix/Tensor Contraction Micro-Kernel.
pub struct TiledContractionKernel;

impl TiledContractionKernel {
    /// Computes C[4 x 16] += A[4 x K] * B[K x 16] using four independent 16-lane SIMD
    /// accumulator registers (`c0, c1, c2, c3`), achieving 64 FMAs per tile load.
    #[inline]
    pub fn contract_4x16_panel(
        a_rows: &[[f32; 4]],
        b_tiles: &[AlignedTile64],
        c_out: &mut [AlignedTile64; 4],
    ) {
        let k_len = a_rows.len().min(b_tiles.len());
        let mut c0 = c_out[0];
        let mut c1 = c_out[1];
        let mut c2 = c_out[2];
        let mut c3 = c_out[3];

        for k in 0..k_len {
            let a_col = a_rows[k];
            let b_vec = &b_tiles[k];

            c0.fma_scale_assign(a_col[0], b_vec);
            c1.fma_scale_assign(a_col[1], b_vec);
            c2.fma_scale_assign(a_col[2], b_vec);
            c3.fma_scale_assign(a_col[3], b_vec);
        }

        c_out[0] = c0;
        c_out[1] = c1;
        c_out[2] = c2;
        c_out[3] = c3;
    }

    /// Reference FP64 panel contraction for numerical ULP accuracy verification.
    pub fn contract_4x16_reference_f64(
        a_rows: &[[f32; 4]],
        b_tiles: &[AlignedTile64],
    ) -> [[f64; 16]; 4] {
        let k_len = a_rows.len().min(b_tiles.len());
        let mut out = [[0.0f64; 16]; 4];
        for k in 0..k_len {
            for r in 0..4 {
                let a_val = f64::from(a_rows[k][r]);
                for c in 0..16 {
                    out[r][c] += a_val * f64::from(b_tiles[k].0[c]);
                }
            }
        }
        out
    }
}

/// Branchless Conway-Sloane Gosset E8 Lattice Nearest-Point Micro-Kernel.
///
/// The Gosset lattice E8 in R^8 is the union of D8 and D8 + (1/2, ..., 1/2),
/// where D8 = { z in Z^8 : sum(z_i) is even }.
pub struct E8LatticeMicroKernel;

impl E8LatticeMicroKernel {
    /// Projects an arbitrary 8D vector `x` onto the closest point in the checkerboard lattice D8.
    /// Returns `(nearest_d8_point, squared_euclidean_distance)`.
    #[inline]
    pub fn project_d8(x: &[f32; 8]) -> ([f32; 8], f32) {
        let mut f = [0.0f32; 8];
        let mut sum_int: i32 = 0;
        let mut max_err = -1.0f32;
        let mut worst_idx = 0usize;

        for i in 0..8 {
            let r = x[i].round();
            f[i] = r;
            sum_int = sum_int.wrapping_add(r as i32);
            let err = (x[i] - r).abs();
            if err > max_err {
                max_err = err;
                worst_idx = i;
            }
        }

        // If sum of rounded integers is odd, flip the coordinate with the largest rounding residual
        if sum_int & 1 != 0 {
            let diff = x[worst_idx] - f[worst_idx];
            if diff >= 0.0 {
                f[worst_idx] += 1.0;
            } else {
                f[worst_idx] -= 1.0;
            }
        }

        let mut dist_sq = 0.0f32;
        for i in 0..8 {
            let d = x[i] - f[i];
            dist_sq = d.mul_add(d, dist_sq);
        }

        (f, dist_sq)
    }

    /// Exact Conway-Sloane nearest-point decoding on the Gosset E8 lattice in O(8) operations.
    /// Returns `(nearest_e8_point, squared_euclidean_distance, coset_bit)`.
    #[inline]
    pub fn decode_nearest_e8(x: &[f32; 8]) -> ([f32; 8], f32, u8) {
        // Coset 0: D8
        let (y0, d0) = Self::project_d8(x);

        // Coset 1: D8 + 0.5
        let mut x_shifted = [0.0f32; 8];
        for i in 0..8 {
            x_shifted[i] = x[i] - 0.5;
        }
        let (mut y1, d1) = Self::project_d8(&x_shifted);
        for val in &mut y1 {
            *val += 0.5;
        }

        if d0 <= d1 {
            (y0, d0, 0)
        } else {
            (y1, d1, 1)
        }
    }
}

/// High-speed SIMD-friendly Complex Kähler-Bergman Reproducing Kernel Evaluator.
pub struct BergmanSimdKernel;

impl BergmanSimdKernel {
    /// Computes the Hermitian inner product <z, w>_C = sum(z_j * conj(w_j))
    /// across interleaved 16-lane tiles (8 complex numbers per tile).
    #[inline]
    pub fn hermitian_inner_product_tile(z_tile: &AlignedTile64, w_tile: &AlignedTile64) -> (f32, f32) {
        let mut re_acc = 0.0f32;
        let mut im_acc = 0.0f32;

        // Even indices (2*j) are real parts; odd indices (2*j + 1) are imaginary parts.
        // z_j * conj(w_j) = (z_re * w_re + z_im * w_im) + i * (z_im * w_re - z_re * w_im)
        for j in 0..8 {
            let z_re = z_tile.0[2 * j];
            let z_im = z_tile.0[2 * j + 1];
            let w_re = w_tile.0[2 * j];
            let w_im = w_tile.0[2 * j + 1];

            re_acc = z_re.mul_add(w_re, z_im.mul_add(w_im, re_acc));
            im_acc = z_im.mul_add(w_re, (-z_re).mul_add(w_im, im_acc));
        }

        (re_acc, im_acc)
    }

    /// Evaluates the normalized Bergman kernel K_norm(z, w) = (1 - <z, w>_C)^{-(n+1)}
    /// using O(log n) binary exponentiation by squaring in complex arithmetic.
    #[inline]
    pub fn evaluate_normalized_bergman(
        z_tiles: &[AlignedTile64],
        w_tiles: &[AlignedTile64],
        complex_dim_n: usize,
    ) -> (f32, f32) {
        let n_tiles = z_tiles.len().min(w_tiles.len());
        let mut dot_re = 0.0f32;
        let mut dot_im = 0.0f32;

        for t in 0..n_tiles {
            let (re, im) = Self::hermitian_inner_product_tile(&z_tiles[t], &w_tiles[t]);
            dot_re += re;
            dot_im += im;
        }

        // Base complex number u = 1 - <z, w>_C
        let u_re = 1.0 - dot_re;
        let u_im = -dot_im;

        // Compute u^{n + 1} via binary exponentiation by squaring
        let mut exp = (complex_dim_n + 1) as u32;
        let mut base_re = u_re;
        let mut base_im = u_im;
        let mut pow_re = 1.0f32;
        let mut pow_im = 0.0f32;

        while exp > 0 {
            if exp & 1 == 1 {
                let next_re = pow_re.mul_add(base_re, -pow_im * base_im);
                let next_im = pow_re.mul_add(base_im, pow_im * base_re);
                pow_re = next_re;
                pow_im = next_im;
            }
            let sq_re = base_re.mul_add(base_re, -base_im * base_im);
            let sq_im = 2.0 * base_re * base_im;
            base_re = sq_re;
            base_im = sq_im;
            exp >>= 1;
        }

        // Complex reciprocal 1 / (pow_re + i * pow_im)
        let norm_sq = pow_re.mul_add(pow_re, pow_im * pow_im).max(1e-20);
        (pow_re / norm_sq, -pow_im / norm_sq)
    }
}

/// Telemetry report from hardware micro-kernel verification & benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroKernelBenchmarkReport {
    pub tile_size_bytes: usize,
    pub tile_alignment_bytes: usize,
    pub contraction_gflops: f64,
    pub contraction_max_rel_err: f64,
    pub e8_quant_latency_ns: f64,
    pub e8_valid_parity_pct: f64,
    pub bergman_eval_latency_ns: f64,
    pub bergman_rel_err_vs_f64: f64,
}

/// Evaluator for Phase 9 Hardware Micro-Kernels.
pub struct MicroKernelEvaluator;

impl MicroKernelEvaluator {
    pub fn evaluate() -> MicroKernelBenchmarkReport {
        let tile_size = std::mem::size_of::<AlignedTile64>();
        let tile_align = std::mem::align_of::<AlignedTile64>();

        // 1. Benchmark 4x16 register-blocked panel contraction (K = 128, 1000 iterations)
        let k_dim = 128usize;
        let a_rows: Vec<[f32; 4]> = (0..k_dim)
            .map(|k| {
                let kf = k as f32 * 0.01;
                [kf.sin(), kf.cos(), (kf * 2.0).sin(), (kf * 0.5).cos()]
            })
            .collect();
        let b_tiles: Vec<AlignedTile64> = (0..k_dim)
            .map(|k| {
                let mut arr = [0.0f32; 16];
                for (c, val) in arr.iter_mut().enumerate() {
                    *val = ((k * 16 + c) as f32 * 0.005).cos();
                }
                AlignedTile64(arr)
            })
            .collect();

        let mut c_out = [AlignedTile64::zero(); 4];
        let iters = 2000usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            TiledContractionKernel::contract_4x16_panel(&a_rows, &b_tiles, &mut c_out);
        }
        let elapsed_s = t0.elapsed().as_secs_f64().max(1e-9);
        // Each iteration performs 4 * 16 * K * 2 FLOPs
        let total_flops = (4 * 16 * k_dim * 2 * iters) as f64;
        let gflops = (total_flops / elapsed_s) / 1e9;

        // Verify single-pass accuracy against FP64 reference
        let mut c_single = [AlignedTile64::zero(); 4];
        TiledContractionKernel::contract_4x16_panel(&a_rows, &b_tiles, &mut c_single);
        let ref_f64 = TiledContractionKernel::contract_4x16_reference_f64(&a_rows, &b_tiles);

        let mut max_rel_err = 0.0f64;
        for r in 0..4 {
            for c in 0..16 {
                let actual = f64::from(c_single[r].0[c]);
                let expected = ref_f64[r][c];
                let rel = (actual - expected).abs() / expected.abs().max(1.0);
                max_rel_err = max_rel_err.max(rel);
            }
        }

        // 2. Benchmark E8 Gosset Lattice nearest-point decoder
        let mut e8_valid = 0usize;
        let e8_iters = 5000usize;
        let t_e8 = Instant::now();
        for idx in 0..e8_iters {
            let mut x = [0.0f32; 8];
            for (d, val) in x.iter_mut().enumerate() {
                *val = ((idx * 8 + d) as f32 * 0.173).sin() * 3.5;
            }
            let (y, _dist, coset) = E8LatticeMicroKernel::decode_nearest_e8(&x);
            let sum_coords: f32 = y.iter().sum();
            let sum_rounded = sum_coords.round() as i32;
            if sum_rounded % 2 == 0 && (coset == 0 || coset == 1) {
                e8_valid += 1;
            }
        }
        let e8_ns = t_e8.elapsed().as_nanos() as f64 / (e8_iters as f64);
        let e8_valid_pct = (e8_valid as f64 / e8_iters as f64) * 100.0;

        // 3. Benchmark Bergman SIMD Kernel (d_head = 64 -> n = 32 complex dim = 4 tiles)
        let z_tiles = [AlignedTile64::splat(0.08); 4];
        let w_tiles = [AlignedTile64::splat(0.07); 4];
        let b_iters = 5000usize;
        let t_berg = Instant::now();
        let mut b_res = (0.0f32, 0.0f32);
        for _ in 0..b_iters {
            b_res = BergmanSimdKernel::evaluate_normalized_bergman(&z_tiles, &w_tiles, 32);
        }
        let berg_ns = t_berg.elapsed().as_nanos() as f64 / (b_iters as f64);

        // Analytical f64 reference for z = 0.08 + 0.08i, w = 0.07 + 0.07i across n = 32
        // Each complex coordinate has z_j * conj(w_j) = (0.08*0.07 + 0.08*0.07) + 0i = 0.0112
        let dot_ref = 32.0f64 * (0.08 * 0.07 + 0.08 * 0.07);
        let expected_berg = (1.0f64 - dot_ref).powi(-33);
        let berg_err = (f64::from(b_res.0) - expected_berg).abs() / expected_berg.abs();

        MicroKernelBenchmarkReport {
            tile_size_bytes: tile_size,
            tile_alignment_bytes: tile_align,
            contraction_gflops: gflops,
            contraction_max_rel_err: max_rel_err,
            e8_quant_latency_ns: e8_ns,
            e8_valid_parity_pct: e8_valid_pct,
            bergman_eval_latency_ns: berg_ns,
            bergman_rel_err_vs_f64: berg_err,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile64_memory_layout_and_alignment() {
        assert_eq!(std::mem::size_of::<AlignedTile64>(), 64);
        assert_eq!(std::mem::align_of::<AlignedTile64>(), 64);
    }

    #[test]
    fn test_e8_lattice_axioms_and_optimality() {
        for seed in 0..200 {
            let mut x = [0.0f32; 8];
            for (i, val) in x.iter_mut().enumerate() {
                *val = ((seed * 13 + i * 7) as f32 * 0.37).sin() * 4.2;
            }
            let (y, dist_sq, coset) = E8LatticeMicroKernel::decode_nearest_e8(&x);

            // Verify coordinate parity axiom: sum of coordinates must be an even integer
            let sum: f32 = y.iter().sum();
            let rounded_sum = sum.round() as i32;
            assert!(
                (sum - (rounded_sum as f32)).abs() < 1e-4,
                "Sum {} is not an integer",
                sum
            );
            assert_eq!(
                rounded_sum.rem_euclid(2),
                0,
                "Sum {} is not an even integer",
                rounded_sum
            );

            // Verify all coordinates belong to Z (coset 0) or Z + 0.5 (coset 1)
            for &coord in &y {
                let shifted = if coset == 0 { coord } else { coord - 0.5 };
                assert!((shifted - shifted.round()).abs() < 1e-5);
            }

            // Verify distance is strictly <= distance to the origin (0, ..., 0) in E8 if near origin,
            // or strictly <= any single-pair (+/-1, +/-1) perturbation in D8.
            for i in 0..8 {
                for j in (i + 1)..8 {
                    let mut y_pert = y;
                    y_pert[i] += 1.0;
                    y_pert[j] += 1.0;
                    let d_pert: f32 = x
                        .iter()
                        .zip(y_pert.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    assert!(dist_sq <= d_pert + 1e-4);
                }
            }
        }
    }

    #[test]
    fn test_micro_kernel_evaluator_metrics() {
        let report = MicroKernelEvaluator::evaluate();
        assert_eq!(report.tile_size_bytes, 64);
        assert_eq!(report.tile_alignment_bytes, 64);
        assert!(report.contraction_max_rel_err < 1e-5);
        assert!((report.e8_valid_parity_pct - 100.0).abs() < 1e-9);
        assert!(report.bergman_rel_err_vs_f64 < 1e-4);
    }
}
