//! The backbone with each differential coefficient replaced by its antipode.

use std::{
    collections::BTreeMap,
    io::{Read, Write},
    path::Path,
    sync::Mutex,
    time::Instant,
};

use algebra::{Algebra, MilnorAlgebra, MilnorAntipode, milnor_algebra::PPart};
use maybe_rayon::prelude::*;
use rustc_hash::FxHashMap;
use sseq::coordinates::Bidegree;

use crate::chain_complex::FreeChainComplex;

/// The coefficients of one generator's differential: for each generator $x$ with
/// $\chi(a_{zx}) \neq 0$, its degree, its index and the $R$ of the $P(R)$ in $\chi(a_{zx})$.
pub type ConjugatedOutput = Vec<(i32, usize, Vec<PPart>)>;

/// A free resolution over the Milnor algebra at $p = 2$ with each differential coefficient
/// $a_{zx}$, the element of $A$ multiplying generator $x$ in $dz$, replaced by $\chi(a_{zx})$.
///
/// With the action $x \cdot a = \chi(a) x$ this is a resolution by free right modules, so a
/// comodule can be paired with it through the side of its coaction that needs no antipode (see
/// `algebra::comodule::ConjugateAction`). The antipode is applied to whole coefficients, so the
/// terms of $\chi$ of different $P(R)$ in one coefficient cancel before anything is stored.
pub struct ConjugatedBackbone {
    /// `slices[s][t][l]` conjugates the differential of generator `l` of degree `t` in filtration
    /// `s + 1`, landing in filtration `s`.
    pub(crate) slices: Vec<Vec<Vec<ConjugatedOutput>>>,
}

/// What [`ConjugatedBackbone::compute`] did, by the stem of the generator whose differential
/// was conjugated.
#[derive(Debug, Default, Clone)]
pub struct ConjugationStats {
    /// `(Milnor terms read, antipode terms summed before cancelling, terms kept, seconds)`.
    pub by_stem: BTreeMap<i32, (usize, usize, usize, f64)>,
}

impl ConjugationStats {
    /// The totals over every stem.
    pub fn total(&self) -> (usize, usize, usize, f64) {
        self.by_stem.values().fold((0, 0, 0, 0.0), |a, b| {
            (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3)
        })
    }
}

impl ConjugatedBackbone {
    /// Conjugates the differentials out of filtrations `1..=max_s + 1` from generators of degree
    /// at most `max_t`, skipping bidegrees the backbone has not computed. Bidegrees run in
    /// parallel and share one memo of the antipode.
    pub fn compute<CC>(backbone: &CC, max_s: i32, max_t: i32) -> (Self, ConjugationStats)
    where
        CC: FreeChainComplex<Algebra = MilnorAlgebra> + Sync,
    {
        let algebra = backbone.algebra();
        assert_eq!(algebra.prime(), 2, "the conjugation is written for p = 2");
        let chi = MilnorAntipode::new(std::sync::Arc::clone(&algebra));
        let stats = Mutex::new(ConjugationStats::default());
        let bidegrees: Vec<(i32, i32)> = (0..=max_s)
            .flat_map(|s| (0..=max_t).map(move |t| (s, t)))
            .filter(|&(s, t)| backbone.has_computed_bidegree(Bidegree::s_t(s + 1, t)))
            .collect();
        let done: Vec<((i32, i32), Vec<ConjugatedOutput>)> = bidegrees
            .into_maybe_par_iter()
            .map(|(s, t)| {
                let start = Instant::now();
                let d = backbone.differential(s + 1);
                let target = backbone.module(s);
                let (mut read, mut summed, mut kept) = (0, 0, 0);
                let gens = backbone.number_of_gens_in_bidegree(Bidegree::s_t(s + 1, t));
                let outputs: Vec<ConjugatedOutput> = (0..gens)
                    .map(|l| {
                        let mut odd: FxHashMap<(i32, usize, PPart), bool> = FxHashMap::default();
                        for (idx, _) in d.output(t, l).iter_nonzero() {
                            let og = target.index_to_op_gen(t, idx);
                            let r = algebra
                                .basis_element_from_index(og.operation_degree, og.operation_index)
                                .p_part;
                            let chi_r = chi.of(r);
                            read += 1;
                            summed += chi_r.len();
                            for &(m, _) in chi_r.iter() {
                                let e = odd
                                    .entry((og.generator_degree, og.generator_index, m))
                                    .or_insert(false);
                                *e = !*e;
                            }
                        }
                        let mut by_x: BTreeMap<(i32, usize), Vec<PPart>> = BTreeMap::new();
                        for ((tx, x, m), keep) in odd {
                            if keep {
                                by_x.entry((tx, x)).or_default().push(m);
                            }
                        }
                        by_x.into_iter()
                            .map(|((tx, x), mut ms)| {
                                ms.sort_unstable();
                                kept += ms.len();
                                (tx, x, ms)
                            })
                            .collect()
                    })
                    .collect();
                let mut st = stats.lock().unwrap();
                let e = st.by_stem.entry(t - s - 1).or_default();
                e.0 += read;
                e.1 += summed;
                e.2 += kept;
                e.3 += start.elapsed().as_secs_f64();
                ((s, t), outputs)
            })
            .collect();
        let mut slices: Vec<Vec<Vec<ConjugatedOutput>>> = (0..=max_s)
            .map(|_| (0..=max_t).map(|_| Vec::new()).collect())
            .collect();
        for ((s, t), outputs) in done {
            slices[s as usize][t as usize] = outputs;
        }
        (Self { slices }, stats.into_inner().unwrap())
    }

    /// The number of terms stored.
    pub fn terms(&self) -> usize {
        self.slices
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|x| x.2.len())
            .sum()
    }

    /// The conjugated differential of generator `l` of degree `t` in filtration `s + 1`.
    pub fn output(&self, s: i32, t: i32, l: usize) -> &[(i32, usize, Vec<PPart>)] {
        self.slices
            .get(s as usize)
            .and_then(|x| x.get(t as usize))
            .and_then(|x| x.get(l))
            .map_or(&[], Vec::as_slice)
    }

    /// The number of generators of degree `t` in filtration `s + 1` with a stored differential.
    pub fn output_len(&self, s: i32, t: i32) -> usize {
        self.slices
            .get(s as usize)
            .and_then(|x| x.get(t as usize))
            .map_or(0, Vec::len)
    }

    /// Writes the conjugated backbone to `dir`, one file per filtration.
    ///
    /// Each file is a sequence of little-endian `u32`s: per degree, the number of generators;
    /// per generator, the number of $x$; per $x$, its degree, its index and the number of $R$;
    /// per $R$, its length and entries.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (s, by_t) in self.slices.iter().enumerate() {
            let file = std::fs::File::create(dir.join(format!("conjugated_{s}")))?;
            let mut w = std::io::BufWriter::new(file);
            let mut put = |x: u32| w.write_all(&x.to_le_bytes());
            put(by_t.len() as u32)?;
            for gens in by_t {
                put(gens.len() as u32)?;
                for outputs in gens {
                    put(outputs.len() as u32)?;
                    for (tx, x, ms) in outputs {
                        put(*tx as u32)?;
                        put(*x as u32)?;
                        put(ms.len() as u32)?;
                        for m in ms {
                            put(m.len() as u32)?;
                            for e in m.iter() {
                                put(e)?;
                            }
                        }
                    }
                }
            }
            w.flush()?;
        }
        Ok(())
    }

    /// Reads a conjugated backbone [`Self::save`] wrote, for filtrations `0..=max_s`.
    pub fn load(dir: &Path, max_s: i32) -> std::io::Result<Self> {
        let mut slices = Vec::new();
        for s in 0..=max_s {
            let mut bytes = Vec::new();
            std::fs::File::open(dir.join(format!("conjugated_{s}")))?.read_to_end(&mut bytes)?;
            let mut pos = 0;
            let mut get = || {
                let x = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                pos += 4;
                x
            };
            let degrees = get();
            let mut by_t = Vec::with_capacity(degrees as usize);
            for _ in 0..degrees {
                let gens = get();
                let mut by_l = Vec::with_capacity(gens as usize);
                for _ in 0..gens {
                    let n = get();
                    let mut outputs = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        let (tx, x, k) = (get() as i32, get() as usize, get());
                        let ms = (0..k)
                            .map(|_| {
                                let len = get();
                                let entries: Vec<u32> = (0..len).map(|_| get()).collect();
                                PPart::from_slice(&entries)
                            })
                            .collect();
                        outputs.push((tx, x, ms));
                    }
                    by_l.push(outputs);
                }
                by_t.push(by_l);
            }
            slices.push(by_t);
        }
        Ok(Self { slices })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::algebraic_novikov::unit_module;

    /// Saving and loading give back the same conjugated backbone, and every coefficient is
    /// nonzero.
    #[test]
    fn save_round_trip() {
        let res = crate::nassau::Resolution::new(Arc::new(unit_module()));
        res.compute_through_stem(Bidegree::n_s(20, 10));
        let (conj, stats) = ConjugatedBackbone::compute(&res, 9, 30);
        assert!(conj.terms() > 0);
        assert_eq!(stats.total().2, conj.terms());
        let dir = std::env::temp_dir().join(format!("conjugated-{}", std::process::id()));
        conj.save(&dir).unwrap();
        let back = ConjugatedBackbone::load(&dir, 9).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(back.slices, conj.slices);
        for x in conj.slices.iter().flatten().flatten().flatten() {
            assert!(!x.2.is_empty());
        }
    }
}
