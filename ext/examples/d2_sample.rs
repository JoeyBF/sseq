//! Times the d2 composite of a sample of generators of a saved resolution, to size a d2 run.

use std::{
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
};

use ext::{chain_complex::ChainComplex, secondary::SecondaryComposite};
use maybe_rayon::prelude::*;
use sseq::coordinates::Bidegree;

/// The stem band a stem falls in, for stratifying the sample.
const STEM_BAND: i32 = 25;

/// The lower ends of the s bands, for stratifying the sample.
const S_BANDS: [i32; 8] = [2, 4, 7, 11, 16, 24, 40, 80];

/// The s band of `s`, as an index into [`S_BANDS`].
fn s_band(s: i32) -> usize {
    S_BANDS.iter().rposition(|&lo| s >= lo).unwrap()
}

/// Usage: `d2_sample <store> <max_n> <max_s> <counts_out> <samples_out> <per_stratum>
/// <min_stem> <max_stem> [seed]`.
///
/// Opens the resolution saved in `store` through `(max_n, max_s)`, writes the number of
/// generators of every bidegree to `counts_out`, then builds the d2 composite of up to
/// `per_stratum` random generators of each stratum (a [`STEM_BAND`] of stems by an s band of
/// [`S_BANDS`]) with stem in `min_stem..=max_stem`. Each composite is one line of `samples_out`
/// as soon as it is done: `n s idx secs dg_terms dg_gens`, where `dg_terms` is the number of
/// nonzero entries of d(g) and `dg_gens` the number of generators they involve.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 9, "see the usage in the source");
    let store = std::path::PathBuf::from(&args[1]);
    let max = Bidegree::n_s(args[2].parse()?, args[3].parse()?);
    let per_stratum: usize = args[6].parse()?;
    let (min_stem, max_stem): (i32, i32) = (args[7].parse()?, args[8].parse()?);
    let mut seed: u64 = args.get(9).map_or(Ok(1), |s| s.parse())?;

    let start = Instant::now();
    let res = Arc::new(ext::utils::construct("S_2@milnor", Some(store))?);
    res.compute_through_stem(max);
    eprintln!("SAMPLE open_s={:.1}", start.elapsed().as_secs_f64());

    // Every generator a composite exists for (s >= 2), grouped by stratum.
    let mut counts = std::io::BufWriter::new(std::fs::File::create(&args[4])?);
    let mut strata: std::collections::BTreeMap<(i32, usize), Vec<(Bidegree, usize)>> =
        Default::default();
    for s in 2..max.s() {
        for n in 0..=max.n() {
            let b = Bidegree::n_s(n, s);
            if !res.has_computed_bidegree(b) {
                continue;
            }
            let gens = res.module(s).number_of_gens_in_degree(b.t());
            if gens == 0 {
                continue;
            }
            writeln!(counts, "{n} {s} {gens}")?;
            if (min_stem..=max_stem).contains(&n) {
                let stratum = strata.entry((n / STEM_BAND, s_band(s))).or_default();
                stratum.extend((0..gens).map(|i| (b, i)));
            }
        }
    }
    counts.flush()?;

    // A seeded partial Fisher-Yates shuffle picks each stratum's sample.
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut samples = Vec::new();
    for gens in strata.values_mut() {
        let k = per_stratum.min(gens.len());
        for i in 0..k {
            let j = i + (next() % (gens.len() - i) as u64) as usize;
            gens.swap(i, j);
        }
        samples.extend_from_slice(&gens[..k]);
    }
    samples.sort_by_key(|(b, _)| (b.n(), b.s()));
    eprintln!("SAMPLE strata={} samples={}", strata.len(), samples.len());

    let out = Mutex::new(std::fs::File::create(&args[5])?);
    samples.into_maybe_par_iter().for_each(|(b, idx)| {
        let (s, t) = (b.s(), b.t());
        let d1 = res.differential(s);
        let d0 = res.differential(s - 1);
        let dg = d1.output(t, idx);
        let dg_terms = dg.iter_nonzero().count();
        let dg_gens = res
            .module(s - 1)
            .iter_slices(t, dg.as_slice())
            .filter(|(.., v)| !v.is_zero())
            .count();

        let start = Instant::now();
        let mut composite = SecondaryComposite::new(res.module(s - 2), t, false);
        composite.add_composite(1, t, idx, &d1, &d0);
        composite.finalize();
        let secs = start.elapsed().as_secs_f64();

        let mut out = out.lock().unwrap();
        writeln!(
            out,
            "{} {s} {idx} {secs:.4} {dg_terms} {dg_gens}",
            b.n()
        )
        .unwrap();
        out.flush().unwrap();
    });
    Ok(())
}
