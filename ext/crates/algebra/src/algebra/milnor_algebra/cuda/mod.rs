//! [`multiply_batch`](super::batch::multiply_batch) on an NVIDIA GPU.
//!
//! Every (product, term) pair of a batch is one item of a persistent kernel, which enumerates the
//! terms of `P(R) P(S)` with the anti-diagonal walk of `antidiagonal.rs` and XORs each into its row
//! at the index the seqno tables give it (`multiply.cu`). The kernel needs no table of admissible
//! matrices and no state beyond the Milnor basis and the seqno tables, which are uploaded once and
//! extended when a batch reaches a higher degree.
//!
//! The kernel is compiled at runtime by NVRTC for the device it runs on, so any card the driver
//! supports works, and a binary built without a CUDA toolkit still runs; it needs `libnvrtc` at
//! runtime. `MILNOR_CUDA_DISABLE` turns the device off and `MILNOR_CUDA_DEVICE` (default 0) picks
//! it.
//!
//! Memory: a launch's output is split into blocks of whole rows of at most [`BLOCK_BYTES`], and
//! its per-item arrays into chunks of at most [`ARRAYS_BYTES`], all accumulating into the block's
//! output. Each block reserves its bytes from a per-device budget before allocating, so concurrent
//! callers wait for room instead of failing; a block that runs out of device memory anyway is
//! retried once with the device to itself.

use std::{
    borrow::Cow,
    ops::RangeInclusive,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, anyhow, bail};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig,
        PinnedHostSlice, PushKernelArg, sys,
    },
    nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts},
};

use super::{
    MilnorAlgebra, PPart,
    batch::{BatchOutput, BatchProduct, LimbBlock, num_limbs},
};
use crate::algebra::Algebra;

const SRC: &str = include_str!("multiply.cu");

/// The pair-algebra product's mod-4 terms; compiled after [`SRC`], whose helpers it uses.
const SIGMA_SRC: &str = include_str!("sigma.cu");

/// Walk widths the kernel is instantiated for: `multiply_antidiag_6` to `_10`.
const WIDTHS: RangeInclusive<usize> = 6..=PPart::MAX_LEN;

/// Threads per block. Measured on an H200 against 64 and 128.
const THREADS: u32 = 96;

/// Batches of at most this many terms run on the CPU. A launch costs ~0.1 ms however little it
/// carries, against ~0.15 ms for ~300 terms on one CPU thread (H200, stem-150 batches).
pub const CPU_TERMS: usize = 300;

/// Output bytes of one row block.
const BLOCK_BYTES: usize = 1 << 30;

/// Bytes of per-item arrays one kernel launch uploads.
const ARRAYS_BYTES: usize = 1 << 30;

/// Device memory the launch budget leaves to the tables, `fp` and anything else on the card.
const RESERVED_BYTES: usize = 6 << 30;

/// Kernel launches, process-wide. A test comparing the device with the CPU checks that this moved,
/// so that a batch which silently ran on the CPU cannot pass by agreeing with itself.
pub static LAUNCHES: AtomicU64 = AtomicU64::new(0);

/// Batches that failed on the device and were recomputed on the CPU, process-wide.
pub static FAILURES: AtomicU64 = AtomicU64::new(0);

/// The walk width of a product of `degree`: the number of $\xi_i$ that can occur, at least 6.
fn width_for_degree(degree: i32) -> usize {
    let mut n = 0;
    while n < PPart::MAX_LEN && (1i64 << (n + 1)) - 1 <= i64::from(degree) {
        n += 1;
    }
    n.max(*WIDTHS.start())
}

/// Shared memory of one block of the width-`n` kernel: the walk state (`STATE_BYTES`) and a
/// queue of 64 terms per warp (`QCAP`) with its count.
fn shared_bytes(n: usize) -> usize {
    let state = (n * (n - 1) + 2 * (n + 1)) * size_of::<u16>() * THREADS as usize;
    let warps = THREADS as usize / 32;
    state.next_multiple_of(16) + warps * (64 * 16 + 4)
}

/// Shared memory of one block of the width-`n` sigma kernel: the walk state (`SIG_STATE_BYTES`)
/// and the queue, as in [`shared_bytes`].
fn sigma_shared_bytes(n: usize) -> usize {
    let state = (n * (n - 1) + 2 * (n + 1)) * size_of::<u16>() * THREADS as usize;
    let warps = THREADS as usize / 32;
    state.next_multiple_of(16) + warps * (64 * 16 + 4)
}

/// Evaluate a batch on the device, or `None` if it should run on the CPU: no device, an algebra
/// the kernel does not support, or a batch too small to be worth a launch.
pub(super) fn multiply_batch(
    algebra: &MilnorAlgebra,
    num_cols: usize,
    col_map: Option<&[u32]>,
    num_rows: usize,
    products: &[BatchProduct],
) -> Option<anyhow::Result<BatchOutput>> {
    if !super::batch::batch_applicable(algebra) {
        return None;
    }
    let terms: usize = products.iter().map(|p| p.term_indices.len()).sum();
    if terms <= CPU_TERMS {
        return None;
    }
    let device = device()?;
    Some(device.run(algebra, num_cols, col_map, num_rows, products))
}

/// Whether a device was found (opening it on first use).
pub(super) fn available() -> bool {
    device().is_some()
}

/// Report a batch that failed on the device and is being recomputed on the CPU.
pub(super) fn report_failure(e: &anyhow::Error) {
    FAILURES.fetch_add(1, Ordering::Relaxed);
    eprintln!("[milnor_cuda] a batch failed on the device, computing it on the CPU: {e:#}");
}

/// The process's device, opened on first use. `None`, reported once, if there is none.
fn device() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            if std::env::var_os("MILNOR_CUDA_DISABLE").is_some() {
                return None;
            }
            let ordinal = match std::env::var("MILNOR_CUDA_DEVICE") {
                Ok(v) => v
                    .parse()
                    .unwrap_or_else(|_| panic!("MILNOR_CUDA_DEVICE={v:?} is not a device ordinal")),
                Err(_) => 0,
            };
            // cudarc panics, rather than returning an error, when the driver library is absent.
            match std::panic::catch_unwind(|| Device::open(ordinal)) {
                Ok(Ok(device)) => Some(device),
                Ok(Err(e)) => {
                    eprintln!("[milnor_cuda] no GPU multiply: {e:#}");
                    None
                }
                Err(_) => {
                    eprintln!("[milnor_cuda] no GPU multiply: the CUDA driver is not available");
                    None
                }
            }
        })
        .as_ref()
}

/// One width's kernel: the function, its shared memory, and the grid that fills the device.
struct Kernel {
    function: CudaFunction,
    shared_bytes: usize,
    grid: u32,
}

struct Device {
    ctx: Arc<CudaContext>,
    /// Indexed by width minus [`WIDTHS`]`.start()`.
    kernels: Vec<Kernel>,
    /// `sigma_step_6` to `_10` of `sigma.cu`, indexed by walk width minus the least.
    sigma_kernels: Vec<Kernel>,
    /// `sigma_pair` of `sigma.cu`.
    sigma_pair: CudaFunction,
    tables: Mutex<Option<Arc<Tables>>>,
    budget: Budget,
    pinned: PinnedPool,
}

impl Device {
    fn open(ordinal: usize) -> anyhow::Result<Self> {
        // SAFETY: only `dlopen`s the candidate library names. cudarc panics at the first missing
        // symbol, so this has to come before any other NVRTC call.
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            bail!("libnvrtc was not found, so the kernel cannot be compiled");
        }
        let ctx = CudaContext::new(ordinal)?;
        let attribute = |a| ctx.attribute(a).map(|v| v as u32);
        let major =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let sm_count =
            attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;

        let brace = |values: Vec<u32>| {
            let values: Vec<String> = values.iter().map(u32::to_string).collect();
            format!("{{{}}}", values.join(","))
        };
        let shifts = (0..PPart::MAX_LEN).map(PPart::shift).collect();
        let masks = (0..PPart::MAX_LEN).map(PPart::max_entry).collect();
        let opts = CompileOptions {
            arch: Some(Box::leak(
                format!("compute_{major}{minor}").into_boxed_str(),
            )),
            name: Some("multiply.cu".to_owned()),
            options: vec![
                format!("-DPPART_MAX_LEN={}", PPart::MAX_LEN),
                format!("-DPP_SHIFTS={}", brace(shifts)),
                format!("-DPP_MASKS={}", brace(masks)),
                format!("-DNT={THREADS}"),
            ],
            ..Default::default()
        };
        let src = format!("{SRC}\n{SIGMA_SRC}");
        let ptx = compile_ptx_with_opts(src, opts).map_err(|e| match &e {
            CompileError::CompileError { log, .. } => {
                anyhow!(
                    "NVRTC failed to compile multiply.cu:\n{}",
                    log.to_string_lossy()
                )
            }
            _ => anyhow!("NVRTC failed to compile multiply.cu: {e}"),
        })?;
        let module = ctx.load_module(ptx)?;

        let mut kernels = Vec::new();
        for n in WIDTHS {
            let name = format!("multiply_antidiag_{n}");
            let function = module.load_function(&name)?;
            let shared_bytes = shared_bytes(n);
            function.set_attribute(
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )?;
            let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
                THREADS,
                shared_bytes,
                None,
            )?;
            if per_sm == 0 {
                bail!("{name}: a block of {THREADS} threads does not fit on an SM");
            }
            kernels.push(Kernel {
                function,
                shared_bytes,
                grid: per_sm * sm_count,
            });
        }

        let sigma_pair = module.load_function("sigma_pair")?;
        let mut sigma_kernels = Vec::new();
        for n in WIDTHS {
            let name = format!("sigma_step_{n}");
            let function = module.load_function(&name)?;
            let shared_bytes = sigma_shared_bytes(n);
            function.set_attribute(
                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )?;
            let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
                THREADS,
                shared_bytes,
                None,
            )?;
            if per_sm == 0 {
                bail!("{name}: a block of {THREADS} threads does not fit on an SM");
            }
            sigma_kernels.push(Kernel {
                function,
                shared_bytes,
                grid: per_sm * sm_count,
            });
        }
        let total = ctx.total_mem()?;
        Ok(Self {
            sigma_pair,
            sigma_kernels,
            budget: Budget::new(total.saturating_sub(RESERVED_BYTES).max(1 << 30)),
            pinned: PinnedPool::new(ctx.clone()),
            tables: Mutex::new(None),
            kernels,
            ctx,
        })
    }

    /// Device tables covering products of degree up to `max_degree` with right factors up to
    /// `max_s_degree`, uploading new ones if the current ones fall short.
    fn tables(
        &self,
        algebra: &MilnorAlgebra,
        max_degree: i32,
        max_s_degree: i32,
    ) -> anyhow::Result<Arc<Tables>> {
        let mut current = self.tables.lock().unwrap();
        if let Some(t) = &*current
            && t.seqno_degree >= max_degree
            && t.basis_degree >= max_s_degree
        {
            return Ok(t.clone());
        }
        let tables = Arc::new(Tables::upload(&self.ctx.default_stream(), algebra)?);
        if tables.seqno_degree < max_degree || tables.basis_degree < max_s_degree {
            bail!(
                "the seqno tables reach degree {}, but the batch needs {max_degree}",
                tables.seqno_degree
            );
        }
        *current = Some(tables.clone());
        Ok(tables)
    }

    fn run(
        &'static self,
        algebra: &MilnorAlgebra,
        num_cols: usize,
        col_map: Option<&[u32]>,
        num_rows: usize,
        products: &[BatchProduct],
    ) -> anyhow::Result<BatchOutput> {
        let num_limbs = num_limbs(num_cols);
        let products: Cow<[BatchProduct]> = if products.is_sorted_by_key(|p| p.row) {
            Cow::Borrowed(products)
        } else {
            let mut v = products.to_vec();
            v.sort_by_key(|p| p.row);
            Cow::Owned(v)
        };
        let products = &*products;

        // The products that contribute, and each one's R, packed. Products arrive grouped by R, so
        // the previous one is usually reused.
        let live: Vec<bool> = products
            .iter()
            .map(|p| algebra.dimension(p.r_degree + p.s_degree) > 0 && !p.term_indices.is_empty())
            .collect();
        let mut r_bits = Vec::with_capacity(products.len());
        let mut last: Option<((i32, u32), u64)> = None;
        for p in products {
            let key = (p.r_degree, p.r_idx);
            let bits = match last {
                Some((k, b)) if k == key => b,
                _ => algebra.basis_element_from_index(key.0, key.1 as usize).p_part.bits(),
            };
            last = Some((key, bits));
            r_bits.push(bits);
        }
        let (max_degree, max_s_degree) = products
            .iter()
            .zip(&live)
            .filter(|(_, l)| **l)
            .fold((0, 0), |(o, s), (p, _)| {
                (o.max(p.r_degree + p.s_degree), s.max(p.s_degree))
            });
        let tables = self.tables(algebra, max_degree, max_s_degree)?;

        let stream = self.ctx.per_thread_stream();
        let col_map_dev = stream
            .clone_htod(col_map.unwrap_or(&[0]))
            .context("uploading the column map")?;
        let launch = Launch {
            device: self,
            tables: &tables,
            stream: &stream,
            products,
            r_bits: &r_bits,
            col_map: &col_map_dev,
            col_map_len: col_map.map_or(0, |m| m.len() as u32),
            use_col_map: u32::from(col_map.is_some()),
            num_limbs,
        };

        let mut blocks: Vec<Box<dyn LimbBlock>> = Vec::new();
        for (row0, row1, p0, p1) in row_blocks(products, num_rows, num_limbs) {
            let block_len = (row1 - row0) * num_limbs;
            // The live products of the block, by walk width, then cut into chunks of arrays.
            let mut groups: Vec<Vec<usize>> = WIDTHS.map(|_| Vec::new()).collect();
            for i in (p0..p1).filter(|&i| live[i]) {
                let p = &products[i];
                groups[width_for_degree(p.r_degree + p.s_degree) - WIDTHS.start()].push(i);
            }
            let chunks: Vec<Vec<std::ops::Range<usize>>> = groups
                .iter()
                .map(|g| array_chunks(g.iter().map(|&i| products[i].term_indices.len())))
                .collect();
            let max_chunk = chunks
                .iter()
                .zip(&groups)
                .flat_map(|(cs, g)| cs.iter().map(|c| launch.chunk_bytes(&g[c.clone()])))
                .max()
                .unwrap_or(0);
            let need =
                block_len * size_of::<u32>() + max_chunk + size_of_val(col_map.unwrap_or(&[]));

            let result = {
                let _permit = self.budget.acquire(need);
                launch.block(row0, block_len, &groups, &chunks)
            };
            let result = match result {
                Err(e) if is_out_of_memory(&e) => {
                    eprintln!(
                        "[milnor_cuda] out of device memory on a {:.2} GB block; retrying with \
                         the device to itself",
                        need as f64 / 1e9
                    );
                    let _ = stream.synchronize();
                    let _permit = self.budget.acquire_all();
                    trim_memory_pool(&self.ctx);
                    launch.block(row0, block_len, &groups, &chunks)
                }
                r => r,
            };
            blocks.push(Box::new(result?));
        }
        Ok(BatchOutput::from_blocks(blocks, num_limbs))
    }
}

/// The Milnor basis and the seqno tables, on the device.
struct Tables {
    /// The seqno tables cover products of degree at most this.
    seqno_degree: i32,
    /// The basis covers right factors of degree at most this.
    basis_degree: i32,
    /// Index into `pp` of element 0 of each degree.
    basis_start: Vec<u32>,
    /// Every basis element through `basis_degree`, as its packed p-part.
    pp: CudaSlice<u64>,
    /// `SeqnoTables::g` and `xi`, padded to [`PPart::MAX_LEN`] entries.
    g: CudaSlice<u32>,
    xi: CudaSlice<u32>,
    width: u32,
}

impl Tables {
    /// Upload the tables, through the degree the algebra's seqno tables reach.
    fn upload(stream: &Arc<CudaStream>, algebra: &MilnorAlgebra) -> anyhow::Result<Self> {
        let seqno = algebra.seqno_tables();
        let degree = seqno.max_degree();
        let mut basis_start = Vec::with_capacity(degree as usize + 1);
        let mut pp = Vec::new();
        for d in 0..=degree {
            basis_start.push(u32::try_from(pp.len()).context("the basis has 2^32 elements")?);
            pp.extend(algebra.ppart_table(d).iter().map(|p| p.bits()));
        }
        let g: Vec<u32> = seqno
            .g()
            .iter()
            .map(|&v| u32::try_from(v))
            .collect::<Result<_, _>>()
            .context("a seqno table entry exceeds u32")?;
        let mut xi: Vec<u32> = seqno.xi().iter().map(|&x| x as u32).collect();
        xi.resize(xi.len().max(PPart::MAX_LEN), 0);
        let tables = Self {
            seqno_degree: degree,
            basis_degree: degree,
            basis_start,
            pp: stream.clone_htod(&pp)?,
            g: stream.clone_htod(&g)?,
            xi: stream.clone_htod(&xi)?,
            width: seqno.width() as u32,
        };
        stream.synchronize()?;
        Ok(tables)
    }
}

/// What one batch shares across its blocks.
struct Launch<'a> {
    device: &'static Device,
    tables: &'a Tables,
    stream: &'a Arc<CudaStream>,
    products: &'a [BatchProduct],
    r_bits: &'a [u64],
    col_map: &'a CudaSlice<u32>,
    col_map_len: u32,
    use_col_map: u32,
    num_limbs: usize,
}

/// The kernel's per-item and per-product arrays for one chunk of products.
struct Arrays {
    /// Per item: the term's index into the basis table, and its product.
    term_gei: Vec<u32>,
    term_prod: Vec<u32>,
    /// Per product: R, the first limb of its row in the block, and its column offset.
    prod_r: Vec<u64>,
    prod_row_base: Vec<u32>,
    prod_out_offset: Vec<u32>,
}

impl Launch<'_> {
    /// Device bytes of the arrays of `chunk`.
    fn chunk_bytes(&self, chunk: &[usize]) -> usize {
        chunk
            .iter()
            .map(|&i| 8 * self.products[i].term_indices.len() + 16)
            .sum()
    }

    fn arrays(&self, chunk: &[usize], row0: usize) -> anyhow::Result<Arrays> {
        let terms = chunk
            .iter()
            .map(|&i| self.products[i].term_indices.len())
            .sum();
        let mut a = Arrays {
            term_gei: Vec::with_capacity(terms),
            term_prod: Vec::with_capacity(terms),
            prod_r: Vec::with_capacity(chunk.len()),
            prod_row_base: Vec::with_capacity(chunk.len()),
            prod_out_offset: Vec::with_capacity(chunk.len()),
        };
        for &i in chunk {
            let p = &self.products[i];
            let local = a.prod_r.len() as u32;
            let base = self.tables.basis_start[p.s_degree as usize];
            a.term_gei
                .reserve(p.term_indices.len());
            p.term_indices.for_each(|t| a.term_gei.push(base + t as u32));
            a.term_prod
                .extend(std::iter::repeat_n(local, p.term_indices.len()));
            a.prod_r.push(self.r_bits[i]);
            a.prod_row_base
                .push(((p.row as usize - row0) * self.num_limbs) as u32);
            a.prod_out_offset
                .push(p.out_offset);
        }
        Ok(a)
    }

    /// Run the products of one row block, starting at row `row0` and `len` limbs long, and read
    /// its output back.
    fn block(
        &self,
        row0: usize,
        len: usize,
        groups: &[Vec<usize>],
        chunks: &[Vec<std::ops::Range<usize>>],
    ) -> anyhow::Result<Pinned> {
        let stream = self.stream;
        let t = self.tables;
        let mut out = stream
            .alloc_zeros::<u32>(len.max(1))
            .context("allocating the output")?;
        let mut counter = stream
            .alloc_zeros::<u32>(1)
            .context("allocating the counter")?;
        let num_limbs = self.num_limbs as u32;
        let out_len = len as u64;
        for (gi, (group, chunks)) in groups.iter().zip(chunks).enumerate() {
            let kernel = &self.device.kernels[gi];
            for chunk in chunks {
                let a = self.arrays(&group[chunk.clone()], row0)?;
                let num_items = a.term_gei.len() as u32;
                if num_items == 0 {
                    continue;
                }
                let upload = "uploading the arrays";
                let term_gei = stream.clone_htod(&a.term_gei).context(upload)?;
                let term_prod = stream.clone_htod(&a.term_prod).context(upload)?;
                let prod_r = stream.clone_htod(&a.prod_r).context(upload)?;
                let prod_row_base = stream.clone_htod(&a.prod_row_base).context(upload)?;
                let prod_out_offset = stream.clone_htod(&a.prod_out_offset).context(upload)?;
                stream.memset_zeros(&mut counter).context(upload)?;
                let cfg = LaunchConfig {
                    grid_dim: (kernel.grid.min(num_items.div_ceil(THREADS)), 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: kernel.shared_bytes as u32,
                };
                let mut builder = stream.launch_builder(&kernel.function);
                builder
                    .arg(&t.pp)
                    .arg(&term_gei)
                    .arg(&term_prod)
                    .arg(&prod_r)
                    .arg(&prod_row_base)
                    .arg(&prod_out_offset)
                    .arg(&t.g)
                    .arg(&t.xi)
                    .arg(&mut out)
                    .arg(self.col_map)
                    .arg(&self.col_map_len)
                    .arg(&self.use_col_map)
                    .arg(&num_items)
                    .arg(&mut counter)
                    .arg(&t.width)
                    .arg(&num_limbs)
                    .arg(&out_len);
                // SAFETY: the arguments match the kernel's parameters in order and type, and every
                // array is as long as the kernel indexes it: items and products by construction,
                // `out` checked against `out_len` and `num_limbs`, `pp` by the basis covering
                // every term's degree, `g` and `xi` by the seqno tables covering every product's.
                unsafe { builder.launch(cfg) }
                    .with_context(|| format!("launching {:?}", cfg.grid_dim))?;
                LAUNCHES.fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut host = self
            .device
            .pinned
            .take(len)
            .context("allocating page-locked memory")?;
        // Exactly `len` limbs: the copy takes its size from the destination, and a pooled
        // buffer may be longer than this block.
        let dst = &mut host.buf.as_mut().unwrap().0.as_mut_slice()?[..len];
        stream
            .memcpy_dtoh(&out.slice(..len), dst)
            .context("reading the output back")?;
        stream.synchronize().context("completing the block")?;
        Ok(host)
    }
}

/// Split the products, sorted by row, into blocks of whole rows whose output fits
/// [`BLOCK_BYTES`]: `(row0, row1, p0, p1)` per block, the products of rows `row0..row1` being
/// `p0..p1`. A single row larger than that is a block of its own.
fn row_blocks(
    products: &[BatchProduct],
    num_rows: usize,
    num_limbs: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let max_rows = (BLOCK_BYTES / (num_limbs * size_of::<u32>())).max(1);
    let mut blocks = Vec::new();
    let (mut row0, mut p0) = (0, 0);
    for (i, p) in products.iter().enumerate() {
        let row = p.row as usize;
        if row >= row0 + max_rows {
            blocks.push((row0, row, p0, i));
            (row0, p0) = (row, i);
        }
    }
    blocks.push((row0, num_rows, p0, products.len()));
    blocks
}

/// Cut products with the given term counts into consecutive chunks whose arrays fit
/// [`ARRAYS_BYTES`] and whose items fit a `u32`, at least one product each.
fn array_chunks(terms: impl Iterator<Item = usize>) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let (mut start, mut bytes, mut items, mut n) = (0, 0, 0, 0);
    for t in terms {
        let b = 8 * t + 16;
        if n > start && (bytes + b > ARRAYS_BYTES || items + t > u32::MAX as usize) {
            chunks.push(start..n);
            (start, bytes, items) = (n, 0, 0);
        }
        bytes += b;
        items += t;
        n += 1;
    }
    if n > start {
        chunks.push(start..n);
    }
    chunks
}

fn is_out_of_memory(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<DriverError>()
            .is_some_and(|d| d.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
    })
}

/// Return the memory the stream-ordered allocator keeps cached to the device.
fn trim_memory_pool(ctx: &Arc<CudaContext>) {
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    let _ = ctx.bind_to_thread();
    // SAFETY: an out-pointer to a local and the context's own device; trimming releases only
    // memory no allocation holds.
    unsafe {
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS
        {
            sys::cuMemPoolTrimTo(pool, 0);
        }
    }
}

/// Device memory the launches in flight may hold together. A block takes its bytes before
/// allocating, so callers wait for room rather than failing; a request larger than the whole
/// budget is clamped to it and runs alone.
struct Budget {
    cap: usize,
    avail: Mutex<usize>,
    /// Callers waiting for the whole budget. Ordinary requests wait while there are any, so the
    /// budget drains for them instead of being refilled by smaller requests.
    draining: AtomicU64,
    freed: Condvar,
}

struct Permit<'a> {
    budget: &'a Budget,
    bytes: usize,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.budget.avail.lock().unwrap() += self.bytes;
        self.budget.freed.notify_all();
    }
}

impl Budget {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            avail: Mutex::new(cap),
            draining: AtomicU64::new(0),
            freed: Condvar::new(),
        }
    }

    fn acquire(&self, bytes: usize) -> Permit<'_> {
        let bytes = bytes.min(self.cap);
        let mut avail = self.avail.lock().unwrap();
        while *avail < bytes || self.draining.load(Ordering::Relaxed) > 0 {
            avail = self.freed.wait(avail).unwrap();
        }
        *avail -= bytes;
        Permit {
            budget: self,
            bytes,
        }
    }

    /// The whole budget, once every other holder has let go.
    fn acquire_all(&self) -> Permit<'_> {
        let mut avail = self.avail.lock().unwrap();
        self.draining.fetch_add(1, Ordering::Relaxed);
        while *avail < self.cap {
            avail = self.freed.wait(avail).unwrap();
        }
        *avail = 0;
        self.draining.fetch_sub(1, Ordering::Relaxed);
        self.freed.notify_all();
        Permit {
            budget: self,
            bytes: self.cap,
        }
    }
}

/// Page-locked host buffers for reading outputs back, reused because locking pages costs about
/// ten times the copy itself (69 ms against 7 ms for a 338 MB output). At most
/// [`PinnedPool::MAX`] are kept, the smallest being freed first.
struct PinnedPool {
    ctx: Arc<CudaContext>,
    free: Mutex<Vec<PinnedSlice>>,
}

/// A pooled buffer: `PinnedHostSlice` behind a newtype, so the pool can hold it.
struct PinnedSlice(PinnedHostSlice<u32>);

impl PinnedSlice {
    fn len(&self) -> usize {
        cudarc::driver::HostSlice::len(&self.0)
    }
}

/// An output block in a pooled buffer, returned to the pool when the output is dropped.
struct Pinned {
    buf: Option<PinnedSlice>,
    len: usize,
    pool: &'static PinnedPool,
}

impl PinnedPool {
    const MAX: usize = 4;

    fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            free: Mutex::new(Vec::new()),
        }
    }

    fn take(&'static self, len: usize) -> anyhow::Result<Pinned> {
        let reused = {
            let mut free = self.free.lock().unwrap();
            free.iter()
                .position(|b| b.len() >= len)
                .map(|i| free.swap_remove(i))
        };
        let buf = match reused {
            Some(b) => b,
            // SAFETY: the memory is uninitialized until the device writes the output into it,
            // which happens before anything reads it. Flags 0: cached, because the host reads it.
            None => PinnedSlice(unsafe { self.ctx.alloc_pinned_with_flags(len.max(1), 0) }?),
        };
        Ok(Pinned {
            buf: Some(buf),
            len,
            pool: self,
        })
    }

    fn put(&self, buf: PinnedSlice) {
        let mut free = self.free.lock().unwrap();
        free.push(buf);
        if free.len() > Self::MAX {
            let smallest = (0..free.len()).min_by_key(|&i| free[i].len()).unwrap();
            free.swap_remove(smallest);
        }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.put(buf);
        }
    }
}

impl LimbBlock for Pinned {
    fn limbs(&self) -> &[u32] {
        let buf = self.buf.as_ref().unwrap();
        // The copy into it completed before the block was returned (`Launch::block` synchronizes),
        // so this does not wait.
        &buf.0.as_slice().expect("pinned output buffer")[..self.len]
    }
}

/// Row stride of [`PairLayout::yoff`]; mirrored by `Y_STRIDE` in `sigma.cu`.
const Y_STRIDE: usize = 16;

/// A `ys[i][j]` a pair element of that degree does not have; mirrored by `Y_ABSENT`.
const Y_ABSENT: u32 = u32::MAX;

/// The product terms of sigma, over Z/4; mirrored by `PART_PRODUCT`.
pub(crate) const PART_PRODUCT: u32 = 1;

/// The Y terms of sigma, mod 2; mirrored by `PART_Y`.
pub(crate) const PART_Y: u32 = 2;

/// Where the parts of a pair element of each degree sit in a row of a [`sigma_batch`] output: its
/// product terms from bit 0, then each `ys[i][j]` of
/// [`new_pair_element`](crate::pair_algebra::PairAlgebra::new_pair_element), in its order.
pub(crate) struct PairLayout {
    /// The first bit of `ys[i][j]` of degree `D` at `(D * Y_STRIDE + i) * Y_STRIDE + j`, or
    /// [`Y_ABSENT`].
    yoff: Vec<u32>,
    /// The bits a pair element of each degree takes.
    bits: Vec<usize>,
}

impl PairLayout {
    /// The layout of pair elements of every degree through `max_degree`.
    pub(crate) fn new(algebra: &MilnorAlgebra, max_degree: i32) -> Self {
        let n = max_degree as usize + 1;
        let mut yoff = vec![Y_ABSENT; n * Y_STRIDE * Y_STRIDE];
        let mut bits = Vec::with_capacity(n);
        for d in 0..n {
            let mut off = algebra.dimension(d as i32);
            if d > 0 {
                for k in 0..=fp::prime::log2(d) {
                    let rem = d + 1 - (1 << k);
                    for l in 0..=fp::prime::log2(rem) {
                        yoff[(d * Y_STRIDE + k) * Y_STRIDE + l] =
                            u32::try_from(off).expect("a pair element exceeds u32 bits");
                        off += algebra.dimension((rem - (1 << l)) as i32);
                    }
                }
            }
            bits.push(off);
        }
        Self { yoff, bits }
    }

    /// The bits a pair element of `degree` takes.
    pub(crate) fn bits(&self, degree: i32) -> usize {
        self.bits[degree as usize]
    }

    /// The first bit of `ys[i][j]` of a pair element of `degree`.
    pub(crate) fn y_offset(&self, degree: i32, i: usize, j: usize) -> usize {
        let off = self.yoff[(degree as usize * Y_STRIDE + i) * Y_STRIDE + j];
        assert_ne!(off, Y_ABSENT, "no ys[{i}][{j}] in degree {degree}");
        off as usize
    }
}

/// One product of a [`sigma_batch`]: `sigma(P(r), P(s))`, of total degree `degree`, added into
/// row `row` from bit `col` on, laid out as [`PairLayout`] says.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SigmaItem {
    pub r: PPart,
    pub s: PPart,
    pub degree: i32,
    pub row: u32,
    pub col: u32,
}

/// The parts `parts` ([`PART_PRODUCT`], [`PART_Y`]) of the pair-algebra product of every item,
/// summed into `num_rows` rows of `num_cols` bits as the bit planes `(ones, twos)`. The product
/// terms are summed mod 4 across both planes, the Y terms mod 2 in `ones`.
pub(crate) fn sigma_batch(
    algebra: &MilnorAlgebra,
    layout: &PairLayout,
    parts: u32,
    items: &[SigmaItem],
    num_rows: usize,
    num_cols: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let device = device().context("no CUDA device")?;
    anyhow::ensure!(
        items.iter().all(|i| (i.row as usize) < num_rows),
        "an item's row is out of range"
    );
    let max_degree = items.iter().map(|i| i.degree).max().unwrap_or(0);
    anyhow::ensure!(
        (max_degree as usize) < layout.bits.len(),
        "the layout stops below degree {max_degree}"
    );
    let tables = device.tables(algebra, max_degree, 0)?;
    let stream = device.ctx.per_thread_stream();
    let num_limbs = num_limbs(num_cols);
    let r: Vec<u64> = items.iter().map(|i| i.r.bits()).collect();
    let s: Vec<u64> = items.iter().map(|i| i.s.bits()).collect();
    let degree: Vec<i32> = items.iter().map(|i| i.degree).collect();
    let row: Vec<u32> = items.iter().map(|i| i.row).collect();
    let col: Vec<u32> = items.iter().map(|i| i.col).collect();
    let (r, s) = (stream.clone_htod(&r)?, stream.clone_htod(&s)?);
    let degree = stream.clone_htod(&degree)?;
    let (row, col) = (stream.clone_htod(&row)?, stream.clone_htod(&col)?);
    let yoff = stream.clone_htod(&layout.yoff)?;
    let mut ones = stream.alloc_zeros::<u32>((num_rows * num_limbs).max(1))?;
    let mut twos = stream.alloc_zeros::<u32>((num_rows * num_limbs).max(1))?;
    let num_items = u32::try_from(items.len()).context("too many items")?;
    let num_limbs_u32 = u32::try_from(num_limbs).context("rows too long")?;
    anyhow::ensure!(
        num_rows * num_limbs < 1 << 32 && num_cols < 1 << 29,
        "the output is too large for the kernel's u32 words and 29-bit columns"
    );
    if std::env::var("SIGMA_KERNEL").is_ok_and(|v| v == "pair") {
        const BLOCK: u32 = 128;
        let cfg = LaunchConfig {
            grid_dim: (num_items.div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&device.sigma_pair);
        builder
            .arg(&r)
            .arg(&s)
            .arg(&degree)
            .arg(&row)
            .arg(&col)
            .arg(&num_items)
            .arg(&tables.g)
            .arg(&tables.xi)
            .arg(&tables.width)
            .arg(&yoff)
            .arg(&parts)
            .arg(&mut ones)
            .arg(&mut twos)
            .arg(&num_limbs_u32);
        // SAFETY: the arguments match `sigma_pair`'s parameters in order and type. The item
        // arrays hold `num_items` entries, every row is below `num_rows` (checked above), the
        // kernel drops columns past `num_limbs`, `yoff` and the tables `g` and `xi` cover every
        // item's degree (checked above, and by `tables`), and the kernel skips a `ys[i][j]` with
        // `i` or `j` past `Y_STRIDE`.
        unsafe { builder.launch(cfg) }.context("launching sigma_pair")?;
    } else {
        let kernel = &device.sigma_kernels[width_for_degree(max_degree) - WIDTHS.start()];
        let mut counter = stream.alloc_zeros::<u32>(1)?;
        let cfg = LaunchConfig {
            grid_dim: (kernel.grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: kernel.shared_bytes as u32,
        };
        let mut builder = stream.launch_builder(&kernel.function);
        builder
            .arg(&r)
            .arg(&s)
            .arg(&degree)
            .arg(&row)
            .arg(&col)
            .arg(&num_items)
            .arg(&mut counter)
            .arg(&tables.g)
            .arg(&tables.xi)
            .arg(&tables.width)
            .arg(&yoff)
            .arg(&parts)
            .arg(&mut ones)
            .arg(&mut twos)
            .arg(&num_limbs_u32);
        // SAFETY: the arguments match `sigma_step_N`'s parameters in order and type, and the
        // block's dynamic shared memory is the `sigma_shared_bytes(N)` the kernel lays out. The
        // item arrays hold `num_items` entries, handed out once each by `counter`, which starts at
        // zero; every row is below `num_rows` and the row words and columns fit the kernel's
        // `u32`s and 29 bits (checked above); the kernel drops columns past `num_limbs`; `yoff`
        // and the tables `g` and `xi` cover every item's degree, and the width `N` every entry of
        // a product of degree at most `max_degree`.
        unsafe { builder.launch(cfg) }.context("launching sigma_step")?;
    }
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    let ones = stream.clone_dtoh(&ones)?;
    let twos = stream.clone_dtoh(&twos)?;
    stream.synchronize()?;
    Ok((ones, twos))
}

#[cfg(test)]
mod sigma_tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{algebra::milnor_algebra::MilnorBasisElement, pair_algebra::PairAlgebra};

    /// A basis element of `degree` from its `PPart`.
    fn elt(algebra: &MilnorAlgebra, p_part: PPart, degree: i32) -> (i32, usize) {
        let idx = algebra.basis_element_to_index(&MilnorBasisElement {
            q_part: 0,
            p_part,
            degree,
        });
        (degree, idx)
    }

    /// Every `stride`-th pair of basis elements of total degree `total`, the stride chosen to
    /// keep at most `max_pairs`.
    fn pairs(algebra: &MilnorAlgebra, total: i32, max_pairs: usize) -> Vec<(i32, PPart, PPart)> {
        let count: usize = (0..=total)
            .map(|a| algebra.dimension(a) * algebra.dimension(total - a))
            .sum();
        let stride = count.div_ceil(max_pairs).max(1);
        let mut kept = Vec::new();
        let mut n = 0;
        for a in 0..=total {
            for &r in algebra.ppart_table(a) {
                for &s in algebra.ppart_table(total - a) {
                    if n % stride == 0 {
                        kept.push((a, r, s));
                    }
                    n += 1;
                }
            }
        }
        kept
    }

    /// Runs `sigma(r, s)` of `pairs` of total degree `total` on the device and, unless `check` is
    /// false, on one CPU thread with `sigma_multiply_basis_direct`, and compares every bit. The
    /// pairs are summed into `rows` rows, so carries between pairs are exercised. Returns the
    /// device's seconds, the CPU's and the number of pairs.
    fn run(
        algebra: &MilnorAlgebra,
        total: i32,
        max_pairs: usize,
        rows: usize,
        parts: u32,
        check: bool,
    ) -> (f64, f64, usize) {
        let MilnorAlgebra::Polynomial(inner) = algebra else {
            unreachable!()
        };
        let pairs = pairs(algebra, total, max_pairs);
        let num_rows = rows.min(pairs.len()).max(1);
        let layout = PairLayout::new(algebra, total);
        let num_cols = layout.bits(total);
        let items: Vec<SigmaItem> = pairs
            .iter()
            .enumerate()
            .map(|(n, &(_, r, s))| SigmaItem {
                r,
                s,
                degree: total,
                row: (n % num_rows) as u32,
                col: 0,
            })
            .collect();

        let start = std::time::Instant::now();
        let before = LAUNCHES.load(Ordering::Relaxed);
        let (ones, twos) =
            sigma_batch(algebra, &layout, parts, &items, num_rows, num_cols).unwrap();
        let gpu_s = start.elapsed().as_secs_f64();
        assert!(
            LAUNCHES.load(Ordering::Relaxed) > before,
            "nothing was launched"
        );
        if !check {
            return (gpu_s, 0.0, pairs.len());
        }

        let start = std::time::Instant::now();
        let mut want: Vec<_> = (0..num_rows)
            .map(|_| inner.new_pair_element(total))
            .collect();
        for (n, &(a, r, s)) in pairs.iter().enumerate() {
            let (rd, ri) = elt(algebra, r, a);
            let (sd, si) = elt(algebra, s, total - a);
            inner.sigma_multiply_basis_direct(&mut want[n % num_rows], 1, rd, ri, sd, si);
        }
        let cpu_s = start.elapsed().as_secs_f64();

        let num_limbs = num_limbs(num_cols);
        let bit =
            |v: &[u32], row: usize, col: usize| (v[row * num_limbs + col / 32] >> (col % 32)) & 1;
        for (row, w) in want.iter().enumerate() {
            if parts & PART_PRODUCT != 0 {
                for col in 0..w.ones.len() {
                    let got = bit(&ones, row, col) + 2 * bit(&twos, row, col);
                    let want = w.ones.entry(col) + 2 * w.twos.entry(col);
                    assert_eq!(got, want, "degree {total}, row {row}, product term {col}");
                }
            }
            if parts & PART_Y != 0 {
                for (i, ys) in w.ys.iter().enumerate() {
                    for (j, y) in ys.iter().enumerate() {
                        let off = layout.y_offset(total, i, j);
                        for x in 0..y.len() {
                            assert_eq!(
                                bit(&ones, row, off + x),
                                y.entry(x),
                                "degree {total}, row {row}, ys[{i}][{j}] term {x}"
                            );
                        }
                    }
                }
            }
        }
        (gpu_s, cpu_s, pairs.len())
    }

    #[test]
    fn sigma_matches_cpu() {
        let algebra = MilnorAlgebra::new(fp::prime::TWO, false);
        algebra.compute_basis(130);
        for total in [1, 2, 7, 15, 31, 40, 63, 80, 130] {
            run(&algebra, total, 200_000, 64, PART_PRODUCT | PART_Y, true);
        }
    }

    /// Device against one CPU thread on products of real d2 composites, as `d2_dump` writes them:
    /// `SIGMA_DUMP` the records, `SIGMA_DUMP_INDEX` the index. Each generator's records are one
    /// launch, one row per g2. `SIGMA_DUMP_CPU_MAX` (default 200000) caps the records the CPU
    /// times, and the device's output is checked against it when it covers the generator. Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn sigma_dump_bench() {
        let path = std::env::var("SIGMA_DUMP").expect("SIGMA_DUMP");
        let index =
            std::fs::read_to_string(std::env::var("SIGMA_DUMP_INDEX").expect("index")).unwrap();
        let cpu_max: usize =
            std::env::var("SIGMA_DUMP_CPU_MAX").map_or(200_000, |v| v.parse().unwrap());
        let bytes = std::fs::read(path).unwrap();
        let record = |k: usize| {
            let b = &bytes[k * 24..][..24];
            let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            (
                PPart::from_bits(u64_at(0)),
                PPart::from_bits(u64_at(8)),
                u32_at(16) as i32,
                u32_at(20),
            )
        };
        let gens: Vec<Vec<usize>> = index
            .lines()
            .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
            .collect();
        let max_degree = gens.iter().map(|g| g[0] + g[1]).max().unwrap() as i32;
        let algebra = MilnorAlgebra::new(fp::prime::TWO, false);
        algebra.compute_basis(max_degree + 1);
        let MilnorAlgebra::Polynomial(inner) = &algebra else {
            unreachable!()
        };
        let layout = PairLayout::new(&algebra, max_degree);
        run(&algebra, 10, usize::MAX, 64, PART_PRODUCT | PART_Y, false);
        let (mut gpu_total, mut cpu_total, mut cpu_records_total, mut records_total) =
            (0.0, 0.0, 0usize, 0usize);
        for g in &gens {
            let (n, s, idx, records, first, num_rows) = (g[0], g[1], g[2], g[3], g[4], g[5]);
            if records == 0 {
                continue;
            }
            let recs: Vec<_> = (first..first + records).map(record).collect();
            let row_degree: HashMap<u32, i32> = recs.iter().map(|r| (r.3, r.2)).collect();
            let row_bits = row_degree.values().map(|&d| layout.bits(d)).max().unwrap();
            let items: Vec<SigmaItem> = recs
                .iter()
                .map(|&(r, s, degree, row)| SigmaItem {
                    r,
                    s,
                    degree,
                    row,
                    col: 0,
                })
                .collect();
            let start = std::time::Instant::now();
            let (ones, twos) = sigma_batch(
                &algebra,
                &layout,
                PART_PRODUCT | PART_Y,
                &items,
                num_rows,
                row_bits,
            )
            .unwrap();
            let gpu_s = start.elapsed().as_secs_f64();

            let degree_of = |p: PPart| -> i32 {
                p.iter()
                    .enumerate()
                    .map(|(i, e)| ((1 << (i + 1)) - 1) * e as i32)
                    .sum()
            };
            let timed = records.min(cpu_max);
            let mut want: HashMap<u32, _> = HashMap::new();
            let start = std::time::Instant::now();
            for &(r, s, degree, row) in &recs[..timed] {
                let w = want
                    .entry(row)
                    .or_insert_with(|| inner.new_pair_element(degree));
                let (rd, sd) = (degree_of(r), degree_of(s));
                let ri = algebra.basis_element_to_index(&MilnorBasisElement {
                    q_part: 0,
                    p_part: r,
                    degree: rd,
                });
                let si = algebra.basis_element_to_index(&MilnorBasisElement {
                    q_part: 0,
                    p_part: s,
                    degree: sd,
                });
                inner.sigma_multiply_basis_direct(w, 1, rd, ri, sd, si);
            }
            let cpu_s = start.elapsed().as_secs_f64();
            if timed == records {
                let num_limbs = num_limbs(row_bits);
                let bit = |v: &[u32], row: usize, col: usize| {
                    (v[row * num_limbs + col / 32] >> (col % 32)) & 1
                };
                for (&row, w) in &want {
                    let row = row as usize;
                    for col in 0..w.ones.len() {
                        let got = bit(&ones, row, col) + 2 * bit(&twos, row, col);
                        assert_eq!(got, w.ones.entry(col) + 2 * w.twos.entry(col));
                    }
                    let degree = row_degree[&(row as u32)];
                    for (i, ys) in w.ys.iter().enumerate() {
                        for (j, y) in ys.iter().enumerate() {
                            let off = layout.y_offset(degree, i, j);
                            for x in 0..y.len() {
                                assert_eq!(bit(&ones, row, off + x), y.entry(x));
                            }
                        }
                    }
                }
            }
            let cpu_per = cpu_s / timed as f64;
            println!(
                "SIGMADUMP ({n},{s},{idx}) records={records} rows={num_rows} gpu_s={gpu_s:.4} \
                 gpu_ns_per_pair={:.1} cpu1_ns_per_pair={:.1} gpu_over_cpu1={:.1} checked={}",
                gpu_s / records as f64 * 1e9,
                cpu_per * 1e9,
                cpu_per * records as f64 / gpu_s,
                timed == records
            );
            gpu_total += gpu_s;
            cpu_total += cpu_s;
            cpu_records_total += timed;
            records_total += records;
        }
        let cpu_per = cpu_total / cpu_records_total as f64;
        println!(
            "SIGMADUMP total records={records_total} gpu_ns_per_pair={:.1} \
             cpu1_ns_per_pair={:.1} gpu_over_cpu1={:.1}",
            gpu_total / records_total as f64 * 1e9,
            cpu_per * 1e9,
            cpu_per * records_total as f64 / gpu_total
        );
    }

    /// Device against one CPU thread on pairs of each total degree `SIGMA_BENCH_DEGREES` (default
    /// 100..=300 by 100), at most `SIGMA_BENCH_MAX_PAIRS` of them, for the product terms, the Y
    /// terms and both. `SIGMA_BENCH_CHECK=0` skips the CPU. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn sigma_bench() {
        let degrees: Vec<i32> = std::env::var("SIGMA_BENCH_DEGREES")
            .map(|v| v.split(',').map(|d| d.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![100, 200, 300]);
        let max_pairs =
            std::env::var("SIGMA_BENCH_MAX_PAIRS").map_or(1_000_000, |v| v.parse().unwrap());
        let check = std::env::var("SIGMA_BENCH_CHECK").map_or(true, |v| v != "0");
        let algebra = MilnorAlgebra::new(fp::prime::TWO, false);
        algebra.compute_basis(*degrees.iter().max().unwrap() + 1);
        // Warm the device and its compiled module.
        run(&algebra, 10, usize::MAX, 64, PART_PRODUCT | PART_Y, false);
        for total in degrees {
            for (name, parts) in [("product", PART_PRODUCT), ("y", PART_Y), ("both", 3)] {
                let (gpu_s, cpu_s, n) =
                    run(&algebra, total, max_pairs, 4096, parts, check && parts == 3);
                println!(
                    "SIGMABENCH degree={total} parts={name} pairs={n} gpu_s={gpu_s:.4} \
                     gpu_ns_per_pair={:.1} cpu1_s={cpu_s:.4} gpu_over_cpu1={:.1}",
                    gpu_s / n as f64 * 1e9,
                    cpu_s / gpu_s
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::batch::tests::*, *};
    use crate::algebra::milnor_algebra::batch::{COL_MAP_DROP, cpu_multiply_batch};

    fn device_or_fail() -> &'static Device {
        device().expect("these tests need a CUDA device")
    }

    /// Run on the device and compare with the CPU, checking the device was used.
    fn check(
        algebra: &MilnorAlgebra,
        num_cols: usize,
        col_map: Option<&[u32]>,
        num_rows: usize,
        products: &[BatchProduct],
    ) {
        let before = LAUNCHES.load(Ordering::Relaxed);
        let got = device_or_fail()
            .run(algebra, num_cols, col_map, num_rows, products)
            .unwrap();
        assert!(
            LAUNCHES.load(Ordering::Relaxed) > before,
            "nothing was launched"
        );
        let want = cpu_multiply_batch(algebra, num_cols, col_map, num_rows, products);
        assert_eq!(got, want);
    }

    #[test]
    fn device_matches_cpu() {
        let max_degree = 300;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(7);
        for (degree, rows, blocks) in [(40, 50, 4), (130, 40, 3), (300, 12, 2)] {
            let (products, width) = random_batch(&algebra, &mut rng, degree, rows, blocks);
            check(&algebra, width, None, rows, &products);
        }
    }

    #[test]
    fn device_matches_cpu_col_map() {
        let max_degree = 160;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(8);
        let (products, width) = random_batch(&algebra, &mut rng, max_degree, 30, 3);
        let kept: Vec<usize> = (0..width).filter(|_| rng.chance(0.2)).collect();
        let mut map = vec![COL_MAP_DROP; width];
        for (j, &c) in kept.iter().enumerate() {
            map[c] = j as u32;
        }
        check(&algebra, kept.len(), Some(&map), 30, &products);
    }

    /// A prefix narrower than the products' blocks keeps only its columns.
    #[test]
    fn device_matches_cpu_prefix() {
        let max_degree = 120;
        let algebra = algebra(max_degree);
        let mut rng = TestRng::new(9);
        let (products, width) = random_batch(&algebra, &mut rng, max_degree, 20, 3);
        check(&algebra, width / 3 + 5, None, 20, &products);
    }

    /// A smaller batch after a larger one reads back into the larger one's pooled buffer.
    #[test]
    fn device_reuses_a_larger_buffer() {
        let algebra = algebra(100);
        let mut rng = TestRng::new(11);
        for rows in [60, 20, 5] {
            let (products, width) = random_batch(&algebra, &mut rng, 100, rows, 3);
            check(&algebra, width, None, rows, &products);
        }
    }

    #[test]
    fn row_blocks_partition() {
        let algebra = algebra(60);
        let mut rng = TestRng::new(10);
        let (products, _) = random_batch(&algebra, &mut rng, 60, 40, 2);
        // An absurdly wide row makes every row a block of its own.
        let wide = BLOCK_BYTES / 4;
        let blocks = row_blocks(&products, 40, wide);
        assert_eq!(blocks.len(), 40);
        let mut next = (0, 0);
        for &(r0, r1, p0, p1) in &blocks {
            assert_eq!((r0, p0), next);
            assert!(products[p0..p1].iter().all(|p| (r0..r1).contains(&(p.row as usize))));
            next = (r1, p1);
        }
        assert_eq!(next, (40, products.len()));
    }

    #[test]
    fn array_chunks_respect_the_budget() {
        // Each of the first four takes a third of the budget: two fit, a third does not.
        let per = ARRAYS_BYTES / 8 / 3;
        let chunks = array_chunks([per, per, per, per, 1, 1].into_iter());
        assert_eq!(chunks, vec![0..2, 2..6]);
        // One product larger than the budget still runs, alone.
        let chunks = array_chunks([ARRAYS_BYTES, 1].into_iter());
        assert_eq!(chunks, vec![0..1, 1..2]);
    }
}
