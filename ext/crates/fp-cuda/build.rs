//! Compile the CUDA C++ kernel to PTX via nvcc.
//!
//! The emitted `matmul_b1.ptx` is picked up by `src/lib.rs` via
//! `include_bytes!(concat!(env!("OUT_DIR"), "/matmul_b1.ptx"))`.
//!
//! A missing `nvcc` is a **hard error**. This used to degrade to a stub PTX carrying no kernel, so
//! that the crate stayed `cargo check`able on a machine with no CUDA Toolkit. That convenience cost
//! far more than it was worth: a stub build is indistinguishable from a real one at a glance, the
//! device banner still prints, and every row reduction silently falls back to CPU M4RI.
//!
//! It happened in production. The binary pinned for the stem-400 run was built without
//! `module load cuda/12.4`, so `nvcc` was absent, the stub was written behind nothing louder than a
//! `cargo:warning`, and 43,557 row reductions ran on the CPU while an entire H200 reserved by
//! `FP_CUDA_DEVICE` sat at 0% for hours. It had already invalidated an earlier measurement before
//! that.
//!
//! Build without CUDA by not enabling the feature that pulls this crate in — not by producing a
//! binary that claims a GPU backend it does not have.

use std::{env, path::PathBuf, process::Command};

const KERNEL_SRC: &str = "cuda_kernels/matmul_b1.cu";
const PTX_NAME: &str = "matmul_b1.ptx";
const ARCH: &str = "sm_90a";

fn main() {
    println!("cargo:rerun-if-changed={KERNEL_SRC}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let ptx_out = out_dir.join(PTX_NAME);

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let status = Command::new(&nvcc)
        .args([
            "-ptx",
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            &format!("-arch={ARCH}"),
            KERNEL_SRC,
            "-o",
        ])
        .arg(&ptx_out)
        .status();

    match status {
        // nvcc ran and compiled the kernel: the real PTX is in place.
        Ok(s) if s.success() => {}
        // nvcc exists but failed to compile: a real kernel error.
        Ok(s) => panic!(
            "nvcc failed to compile {KERNEL_SRC} (exit status: {s}).\nCheck that your CUDA \
             Toolkit supports {ARCH} (Hopper sm_90a)."
        ),
        // nvcc absent. Previously a stub; now fatal, for the reasons in the module docs.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => panic!(
            "nvcc not found ('{nvcc}': {e}).\nfp-cuda cannot be built without the CUDA Toolkit \
             (12.x+). On the cluster this means the build ran without `module load cuda/12.4`; \
             `nvcc` lives at /wsu/el7/cuda/12.4/bin/nvcc. Set NVCC to point at it, put it on PATH, \
             or build without the feature that enables this crate.\nThis used to emit a stub PTX \
             and continue, which silently disabled GPU row reduction at runtime."
        ),
        // nvcc is present but couldn't be executed (permissions, a broken exec, etc.).
        Err(e) => panic!(
            "failed to run nvcc ('{nvcc}': {e}). Set the NVCC env var to a working nvcc."
        ),
    }
}
