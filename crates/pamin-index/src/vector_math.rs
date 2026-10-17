//! The dot product used for exact rescoring after vector-index recall.
//!
//! One kernel per instruction set, chosen at run time on x86-64 so a single
//! binary uses AVX-512 or AVX2 where the CPU has them. NEON is part of every
//! aarch64 target, so it needs no detection; non-Apple aarch64 (Linux on
//! Graviton, Ampere, a Raspberry Pi 5) uses it, because it has no Accelerate.
//! Apple silicon keeps Accelerate's `cblas_sdot`: the library kernel stays the
//! default until a measurement justifies replacing it. The NEON kernel still
//! compiles and is tested there, so a clean measurement can switch to it. The
//! one comparison so far, an M4 screen taken under background load, favoured
//! NEON (5.25 µs against 6.00 µs per 100 rows, p50), but it is provisional and
//! covers no M1–M3.
//! Intel Macs take the same x86 kernels as Linux, so every x86-64 host runs
//! one implementation; nothing has timed them against Accelerate there.
//! Every other target gets a portable reduction over eight independent
//! lanes, which the compiler vectorizes for whatever SIMD the target has.
//!
//! The SIMD kernels sum in a different order from a sequential loop, so a
//! result can differ from one in the last bits, in either direction. On the
//! rescore benchmark's sample of 100 candidates × 1024 dimensions, the largest
//! cosine error against an f64 sum was 1.54e-8 for the AVX-512 kernel against
//! 7.75e-8 for the sequential loop, and 1.43e-8 for NEON against 1.18e-8 for
//! Accelerate on an M4 (`benchmarks/results/inference/rescore-simd-2026-10-12`).
//! The unit test holds every kernel within 1e-5 × √n of an f64 sum.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
}

/// The dot product of the dimensions `left` and `right` share.
pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    let count = left.len().min(right.len());
    let (left, right) = (&left[..count], &right[..count]);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let count = i32::try_from(count).expect("vector dimensions fit i32");
        // SAFETY: both slices contain `count` contiguous f32 values, and the
        // framework reads them only for this synchronous call.
        unsafe { cblas_sdot(count, left.as_ptr(), 1, right.as_ptr(), 1) }
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        neon::dot(left, right)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU reports AVX-512F.
            return unsafe { x86::dot_avx512(left, right) };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the CPU reports AVX2 and FMA.
            return unsafe { x86::dot_avx2(left, right) };
        }
        lanes(left, right)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        lanes(left, right)
    }
}

/// Eight running sums rather than one, so no addition waits on the one before
/// it and the compiler is free to keep them in a vector register.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn lanes(left: &[f32], right: &[f32]) -> f32 {
    let mut sums = [0.0f32; 8];
    let (left_chunks, left_tail) = left.as_chunks::<8>();
    let (right_chunks, right_tail) = right.as_chunks::<8>();
    for (a, b) in left_chunks.iter().zip(right_chunks) {
        for lane in 0..8 {
            sums[lane] += a[lane] * b[lane];
        }
    }
    let tail: f32 = left_tail.iter().zip(right_tail).map(|(a, b)| a * b).sum();
    sums.iter().sum::<f32>() + tail
}

/// Both kernels take slices of equal length and keep four accumulators, so
/// each fused multiply-add is independent of the previous three.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot_avx512(left: &[f32], right: &[f32]) -> f32 {
        let count = left.len();
        let (a, b) = (left.as_ptr(), right.as_ptr());
        let mut sums = [_mm512_setzero_ps(); 4];
        let mut at = 0;
        // SAFETY (all loads): every full load reads 16 values starting below
        // `count - 15`, and the masked load reads only the `count - at` left.
        unsafe {
            while at + 64 <= count {
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let offset = at + 16 * lane;
                    *sum = _mm512_fmadd_ps(
                        _mm512_loadu_ps(a.add(offset)),
                        _mm512_loadu_ps(b.add(offset)),
                        *sum,
                    );
                }
                at += 64;
            }
            while at + 16 <= count {
                sums[0] = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a.add(at)),
                    _mm512_loadu_ps(b.add(at)),
                    sums[0],
                );
                at += 16;
            }
            if at < count {
                let mask = (1u16 << (count - at)) - 1;
                sums[1] = _mm512_fmadd_ps(
                    _mm512_maskz_loadu_ps(mask, a.add(at)),
                    _mm512_maskz_loadu_ps(mask, b.add(at)),
                    sums[1],
                );
            }
        }
        _mm512_reduce_add_ps(_mm512_add_ps(
            _mm512_add_ps(sums[0], sums[1]),
            _mm512_add_ps(sums[2], sums[3]),
        ))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_avx2(left: &[f32], right: &[f32]) -> f32 {
        let count = left.len();
        let (a, b) = (left.as_ptr(), right.as_ptr());
        let mut sums = [_mm256_setzero_ps(); 4];
        let mut at = 0;
        // SAFETY (all loads): every load reads 8 values starting below
        // `count - 7`, and the tail reads single values below `count`.
        unsafe {
            while at + 32 <= count {
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let offset = at + 8 * lane;
                    *sum = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a.add(offset)),
                        _mm256_loadu_ps(b.add(offset)),
                        *sum,
                    );
                }
                at += 32;
            }
            while at + 8 <= count {
                sums[0] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.add(at)),
                    _mm256_loadu_ps(b.add(at)),
                    sums[0],
                );
                at += 8;
            }
            let sum = _mm256_add_ps(
                _mm256_add_ps(sums[0], sums[1]),
                _mm256_add_ps(sums[2], sums[3]),
            );
            let half = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
            let pair = _mm_add_ps(half, _mm_movehl_ps(half, half));
            let mut total = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
            while at < count {
                total += *a.add(at) * *b.add(at);
                at += 1;
            }
            total
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Takes slices of equal length; four accumulators, as on x86. On macOS
    /// only the tests call it, so it stays ready for a measured switch.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub(super) fn dot(left: &[f32], right: &[f32]) -> f32 {
        let count = left.len();
        let (a, b) = (left.as_ptr(), right.as_ptr());
        // SAFETY: NEON is a baseline feature of every aarch64 target; every
        // load reads 4 values starting below `count - 3`, and the tail reads
        // single values below `count`.
        unsafe {
            let mut sums = [vdupq_n_f32(0.0); 4];
            let mut at = 0;
            while at + 16 <= count {
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let offset = at + 4 * lane;
                    *sum = vfmaq_f32(*sum, vld1q_f32(a.add(offset)), vld1q_f32(b.add(offset)));
                }
                at += 16;
            }
            while at + 4 <= count {
                sums[0] = vfmaq_f32(sums[0], vld1q_f32(a.add(at)), vld1q_f32(b.add(at)));
                at += 4;
            }
            let mut total = vaddvq_f32(vaddq_f32(
                vaddq_f32(sums[0], sums[1]),
                vaddq_f32(sums[2], sums[3]),
            ));
            while at < count {
                total += *a.add(at) * *b.add(at);
                at += 1;
            }
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed xorshift sequence in [-1, 1), so failures reproduce.
    fn values(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn dot_reads_only_the_shared_prefix_and_handles_zero() {
        assert_eq!(dot(&[2.0, 3.0], &[4.0]), 8.0);
        assert_eq!(dot(&[0.0; 1024], &[1.0; 1024]), 0.0);
        assert_eq!(dot(&[], &[1.0]), 0.0);
    }

    /// Every length around each kernel's block and tail boundaries, against a
    /// sum in f64. Each kernel this CPU can run is checked, not only the one
    /// `dot` picks, so a machine with AVX-512 still tests the AVX2 path.
    #[test]
    fn every_kernel_matches_an_f64_sum() {
        for count in (0..=80).chain([255, 256, 257, 767, 768, 1023, 1024, 1025]) {
            let (left, right) = (
                values(1 + count as u64, count),
                values(7 + count as u64, count),
            );
            let exact: f64 = left
                .iter()
                .zip(&right)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let tolerance = 1e-5 * (count.max(1) as f64).sqrt();
            let mut kernels = vec![("dot", dot(&left, &right)), ("lanes", lanes(&left, &right))];
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx512f") {
                    // SAFETY: the CPU reports AVX-512F.
                    kernels.push(("avx512", unsafe { x86::dot_avx512(&left, &right) }));
                }
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    // SAFETY: the CPU reports AVX2 and FMA.
                    kernels.push(("avx2", unsafe { x86::dot_avx2(&left, &right) }));
                }
            }
            #[cfg(target_arch = "aarch64")]
            kernels.push(("neon", neon::dot(&left, &right)));
            for (name, value) in kernels {
                let error = (f64::from(value) - exact).abs();
                assert!(error <= tolerance, "{name} at {count}: {value} vs {exact}");
            }
        }
    }
}
