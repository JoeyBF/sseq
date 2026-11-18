//! Writes a sample of the products the d2 composites of chosen generators take, for benchmarks.

use std::{collections::HashMap, io::Write, sync::Arc, time::Instant};

use ext::chain_complex::ChainComplex;
use sseq::coordinates::Bidegree;

/// Usage: `d2_dump <store> <max_n> <max_s> <gens> <out> <index>`.
///
/// `gens` has lines `n s idx p`. The composite of generator `idx` at `(n, s)` multiplies each
/// nonzero entry `r` of the operation from it to a generator g1 of d(g) by each nonzero entry `s` of
/// d(g1), into the row of the generator g2 of row `s - 2` that entry belongs to. Each `r` is kept
/// with probability `p` (seeded, per generator), with all of its products, so the sample keeps the
/// grouping the composite has. Each kept product is a 24-byte record of `out`: the packed `PPart`s
/// of `r` and `s` (`u64` each), their total degree (`i32`) and the row (`u32`, numbering the g2 of
/// the generator in the order they are first met), little-endian. `index` gets one line per
/// generator: `n s idx records first_record rows`.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 7,
        "usage: d2_dump <store> <max_n> <max_s> <gens> <out> <index>"
    );
    let store = std::path::PathBuf::from(&args[1]);
    let max = Bidegree::n_s(args[2].parse()?, args[3].parse()?);
    let gens: Vec<(i32, i32, usize, f64)> = std::fs::read_to_string(&args[4])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            Ok((f[0].parse()?, f[1].parse()?, f[2].parse()?, f[3].parse()?))
        })
        .collect::<anyhow::Result<_>>()?;

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", Some(store))?);
    res.compute_through_stem(max);
    eprintln!("DUMP open_s={:.1}", start.elapsed().as_secs_f64());
    let algebra = res.algebra();
    let algebra: &algebra::MilnorAlgebra = (&*algebra).try_into()?;

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[5])?);
    let mut index = std::io::BufWriter::new(std::fs::File::create(&args[6])?);
    let mut first_record = 0u64;
    for (n, s, idx, p) in gens {
        let t = n + s;
        let (d1, d0) = (res.differential(s), res.differential(s - 1));
        let (below, below2) = (res.module(s - 1), res.module(s - 2));
        let dg = d1.output(t, idx);
        let mut rows: HashMap<(i32, usize), u32> = HashMap::new();
        let mut records = 0u64;
        let mut seed =
            0x9e37_79b9_7f4a_7c15_u64 ^ ((n as u64) << 40 | (s as u64) << 20 | idx as u64);
        for (g1_deg, g1_idx, _, op) in below.iter_slices(t, dg.as_slice()) {
            let dg1 = d0.output(g1_deg, g1_idx);
            for r_idx in op.iter_nonzero().map(|(i, _)| i) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if (seed >> 11) as f64 / (1u64 << 53) as f64 >= p {
                    continue;
                }
                let r = algebra.basis_element_from_index(t - g1_deg, r_idx).p_part;
                for (g2_deg, g2_idx, _, ent) in below2.iter_slices(g1_deg, dg1.as_slice()) {
                    let next = rows.len() as u32;
                    let row = *rows.entry((g2_deg, g2_idx)).or_insert(next);
                    for s_idx in ent.iter_nonzero().map(|(i, _)| i) {
                        let sp = algebra
                            .basis_element_from_index(g1_deg - g2_deg, s_idx)
                            .p_part;
                        out.write_all(&r.bits().to_le_bytes())?;
                        out.write_all(&sp.bits().to_le_bytes())?;
                        out.write_all(&(t - g2_deg).to_le_bytes())?;
                        out.write_all(&row.to_le_bytes())?;
                        records += 1;
                    }
                }
            }
        }
        writeln!(
            index,
            "{n} {s} {idx} {records} {first_record} {}",
            rows.len()
        )?;
        index.flush()?;
        eprintln!(
            "DUMP ({n},{s},{idx}) records={records} rows={} at {:.1}s",
            rows.len(),
            start.elapsed().as_secs_f64()
        );
        first_record += records;
    }
    out.flush()?;
    Ok(())
}
