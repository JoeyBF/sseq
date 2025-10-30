use crate::{
    blas::block::{MatrixBlock, MatrixBlockSlice},
    limb::Limb,
};

mod generic;

#[cfg(target_arch = "x86_64")]
mod x86_64;

pub(crate) fn add_simd(target: &mut [Limb], source: &[Limb], min_limb: usize) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            x86_64::add_simd(target, source, min_limb)
        } else {
            generic::add_simd(target, source, min_limb)
        }
    }
}

pub(crate) fn gather_block_simd(slice: MatrixBlockSlice) -> MatrixBlock {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            x86_64::gather_block_simd(slice)
        } else {
            generic::gather_block_simd(slice)
        }
    }
}

#[inline]
pub(crate) fn gemm_block_simd(
    alpha: bool,
    a: MatrixBlock,
    b: MatrixBlock,
    beta: bool,
    c: MatrixBlock,
) -> MatrixBlock {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            x86_64::gemm_block_simd(alpha, a, b, beta, c)
        } else {
            generic::gemm_block_simd(alpha, a, b, beta, c)
        }
    }
}
