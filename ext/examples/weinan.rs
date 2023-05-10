//! This script converts between our basis and Bruner's basis. At the moment, most inputs are
//! hardcoded, and this only works for the sphere.
//!
//! The script performs the following procedure:
//!
//! 1. Compute our own resolution with the Milnor basis
//! 2. Create Bruner's resolution as a
//!    [`FiniteChainComplex`](ext::chain_complex::FiniteChainComplex) object
//! 3. Use a [`ResolutionHomomorphism`](ext::resolution_homomorphism::ResolutionHomomorphism) to
//!    lift the identity to a chain map from Bruner's resolution
//!    to our resolution. We should do it in this direction because we have stored the
//!    quasi-inverses for our resolution, but not Bruner's.
//! 4. Read off the transformation matrix we need
//!
//! The main extra work to put in is step (2), where we have to parse Bruner's differentials and
//! interpret it as a chain complex. Bruner's resolution can be found at
//! <https://archive.sigma2.no/pages/public/datasetDetail.jsf?id=10.11582/2022.00015>
//! while the descirption of his save file is at <https://arxiv.org/abs/2109.13117>.

use algebra::{
    milnor_algebra::MilnorBasisElement, module::homomorphism::FreeModuleHomomorphism as FMH,
    module::FreeModule as FM, module::Module, Algebra, MilnorAlgebra,
};
use anyhow::Result;
use ext::{
    chain_complex::{ChainComplex, FiniteChainComplex as FCC},
    resolution_homomorphism::ResolutionHomomorphism,
};
use fp::{matrix::Matrix, prime::TWO, vector::FpVector};
use itertools::Itertools;
use std::{fs::File, io::Read, path::PathBuf, sync::Arc};

#[cfg(feature = "nassau")]
type FreeModule = FM<MilnorAlgebra>;
#[cfg(not(feature = "nassau"))]
type FreeModule = FM<algebra::SteenrodAlgebra>;

type FreeModuleHomomorphism = FMH<FreeModule>;
type FiniteChainComplex = FCC<FreeModule, FreeModuleHomomorphism>;

#[derive(Debug, serde::Deserialize)]
pub struct Entry {
    pub id: u32,
    pub s: u32,
    pub t: i32,
    pub index: u32,
    pub diff: String,
}

impl Entry {
    pub fn diff_vec(&self, a: &MilnorAlgebra, m: &FreeModule) -> FpVector {
        a.compute_basis(self.t);
        m.compute_basis(self.t);

        let mut result = FpVector::new(TWO, m.dimension(self.t));

        for monomial in self.diff.split('+') {
            if let Some((coeff, gen_idx)) = monomial.split_once(':') {
                let gen_idx: usize = gen_idx.parse().expect("Malformed generator");
                let gen_offset = m.internal_generator_offset(self.t, gen_idx);

                let p_part: Vec<u32> = coeff.split(',').map(|x| x.parse().unwrap()).collect();
                let degree = p_part
                    .iter()
                    .enumerate()
                    .map(|(idx, k)| ((1 << (idx + 1)) - 1) * k)
                    .sum::<u32>() as i32;
                let elt = MilnorBasisElement {
                    q_part: 0,
                    p_part,
                    degree,
                };
                let op_idx = a.basis_element_to_index(&elt);

                result.add_basis_element(gen_offset + op_idx, 1);
            } else {
                // ensure we're dealing with the identity
                assert_eq!(self.s, 0);
                assert_eq!(self.t, 0);
                assert_eq!(self.diff, "");
                return result;
            }
        }
        result
    }
}

/// Create a new `FiniteChainComplex` with `num_s` many non-zero modules.
fn create_chain_complex(num_s: usize) -> FiniteChainComplex {
    #[cfg(feature = "nassau")]
    let algebra: Arc<MilnorAlgebra> = Arc::new(MilnorAlgebra::new(TWO, false));

    #[cfg(not(feature = "nassau"))]
    let algebra: Arc<algebra::SteenrodAlgebra> = Arc::new(algebra::SteenrodAlgebra::MilnorAlgebra(
        MilnorAlgebra::new(TWO, false),
    ));

    let mut modules: Vec<Arc<FreeModule>> = Vec::with_capacity(num_s);
    let mut differentials: Vec<Arc<FreeModuleHomomorphism>> = Vec::with_capacity(num_s - 1);
    for _ in 0..num_s {
        modules.push(Arc::new(FreeModule::new(
            Arc::clone(&algebra),
            String::new(),
            0,
        )));
    }
    for s in 1..num_s {
        differentials.push(Arc::new(FreeModuleHomomorphism::new(
            Arc::clone(&modules[s]),
            Arc::clone(&modules[s - 1]),
            0,
        )));
    }
    FiniteChainComplex::new(modules, differentials)
}

/// Parse the csv file containing the differentials.
fn read_weinan_resolution<R: Read>(
    mut resolution_csv: csv::Reader<R>,
    max_n: i32,
) -> Result<(u32, FiniteChainComplex)> {
    let records = {
        let mut r = resolution_csv
            .deserialize()
            .filter_ok(|e: &Entry| (e.t - e.s as i32) <= max_n)
            .collect::<Result<Vec<_>, _>>()?;
        r.sort_by(|e1, e2| (e1.s, e1.t).cmp(&(e2.s, e2.t)));
        r
    };

    if records.is_empty() {
        return Err(anyhow::anyhow!("Empty resolution provided"));
    }

    let max_s = records.iter().map(|r| r.s).max().unwrap();
    let max_t = records.iter().map(|r| r.t).max().unwrap();
    let cc = create_chain_complex(max_s as usize);
    let algebra = cc.algebra();
    let algebra: &MilnorAlgebra = algebra.as_ref().try_into()?;

    algebra.compute_basis(max_t + 1);
    // Handle s = 0
    {
        // TODO: actually parse file
        let m = cc.module(0);
        m.add_generators(0, 1, None);
        m.extend_by_zero(max_n + 1);
    }

    for (s, s_group) in records
        .iter()
        .group_by(|e| e.s)
        .into_iter()
        .filter(|(s, _)| *s > 0)
    {
        let m = cc.module(s);
        let d = cc.differential(s);

        for (t, entries) in s_group.group_by(|e| e.t).into_iter() {
            m.extend_by_zero(i32::max(t - 1, 0));
            d.extend_by_zero(i32::max(t - 1, 0));

            m.compute_basis(t);
            let entries = entries
                .map(|e| e.diff_vec(&algebra, &cc.module(s - 1)))
                .collect::<Vec<_>>();
            m.add_generators(t, entries.len(), None);
            d.add_generators_from_rows(t, entries);
        }
        m.extend_by_zero(max_n + s as i32 + 1);
        d.extend_by_zero(max_n + s as i32);
    }

    Ok((max_s, cc))
}

fn main() {
    let resolution_csv: csv::Reader<File> = query::raw("Weinan's csv file", |path| {
        csv::Reader::from_path(PathBuf::from(path))
    });

    let max_n: i32 = query::with_default("Max n", "20", str::parse);

    // Read in Weinan's resolution
    let (max_s, cc) = read_weinan_resolution(resolution_csv, max_n).unwrap();
    let cc = Arc::new(cc);

    let save_dir = query::optional("Save directory", |x| {
        core::result::Result::<PathBuf, std::convert::Infallible>::Ok(PathBuf::from(x))
    });

    #[cfg(feature = "nassau")]
    if save_dir.is_none() {
        panic!(
            "A save directory is required for comparison between Weinan and Nassau resolutions."
        );
    }

    let resolution = ext::utils::construct("S_2@milnor", save_dir).unwrap();
    resolution.compute_through_stem(max_s, max_n);
    let resolution = Arc::new(resolution);

    // Create a ResolutionHomomorphism object
    let hom = ResolutionHomomorphism::new(String::new(), cc, resolution, 0, 0);

    // We have to explicitly tell it what to do at (0, 0)
    hom.extend_step(0, 0, Some(&Matrix::from_vec(TWO, &[vec![1]])));
    hom.extend_all();

    // Now print the results
    println!("sseq_basis | weinan_basis");
    for (s, n, t) in hom.target.iter_stem() {
        let matrix = hom.get_map(s).hom_k(t);

        for (i, row) in matrix.into_iter().enumerate() {
            println!("x_({n},{s},{i}) = {row:?}");
        }
    }
}
