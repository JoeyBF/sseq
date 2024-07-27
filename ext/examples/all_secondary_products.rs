//! Computes all products in $\Mod_{C\tau^2}$.

use std::{sync::Arc, vec};

use algebra::module::Module;
use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    resolution_homomorphism::ResolutionHomomorphism,
    secondary::*,
    utils::{query_module, QueryModuleResolution},
};
use fp::{matrix::Matrix, prime::ValidPrime, vector::FpVector};
use itertools::Itertools;
use maybe_rayon::prelude::*;
use sseq::{
    coordinates::{Bidegree, BidegreeElement, BidegreeGenerator},
    Sseq,
};

#[derive(Clone)]
struct ProductComputationData {
    p: ValidPrime,
    resolution: Arc<QueryModuleResolution>,
    unit: Arc<QueryModuleResolution>,
    is_unit: bool,
    res_lift: Arc<SecondaryResolution<QueryModuleResolution>>,
    unit_lift: Arc<SecondaryResolution<QueryModuleResolution>>,
    res_sseq: Arc<Sseq>,
    unit_sseq: Arc<Sseq>,
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();

    let resolution = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);

    let (is_unit, unit) = ext::utils::get_unit(Arc::clone(&resolution))?;

    let p = resolution.prime();

    let res_lift = SecondaryResolution::new(Arc::clone(&resolution));
    res_lift.extend_all();

    let res_lift = Arc::new(res_lift);

    let unit_lift = if is_unit {
        Arc::clone(&res_lift)
    } else {
        let lift = SecondaryResolution::new(Arc::clone(&unit));
        lift.extend_all();
        Arc::new(lift)
    };

    // Compute E3 page
    let res_sseq = Arc::new(res_lift.e3_page());
    let unit_sseq = if is_unit {
        Arc::clone(&res_sseq)
    } else {
        Arc::new(unit_lift.e3_page())
    };

    let data = ProductComputationData {
        p,
        resolution: Arc::clone(&resolution),
        unit,
        is_unit,
        res_lift,
        unit_lift,
        res_sseq,
        unit_sseq,
    };

    let span = tracing::Span::current();
    resolution
        .iter_stem()
        .skip(1)
        .maybe_par_bridge()
        .try_for_each(|b| -> anyhow::Result<()> {
            let _tracing_guard = span.enter();
            let dim = resolution.number_of_gens_in_bidegree(b);
            for i in 0..dim {
                let g = BidegreeGenerator::new(b, i);
                compute_products(g, data.clone())?;
            }
            Ok(())
        })?;

    Ok(())
}

fn get_page_data(sseq: &sseq::Sseq<sseq::Adams>, b: Bidegree) -> &fp::matrix::Subquotient {
    let d = sseq.page_data(b.n(), b.s() as i32);
    &d[std::cmp::min(3, d.len() - 1)]
}

fn compute_products(
    generator: BidegreeGenerator,
    data: ProductComputationData,
) -> anyhow::Result<()> {
    let name = format!("x_{}", generator);
    let shift = generator.degree();

    if data.resolution.next_homological_degree() <= shift.s() + 2 {
        eprintln!("Adams filtration of {name} too large, skipping");
        return Ok(());
    }

    let hom = Arc::new(ResolutionHomomorphism::new(
        name.clone(),
        Arc::clone(&data.resolution),
        Arc::clone(&data.unit),
        shift,
    ));

    let mut matrix = Matrix::new(data.p, hom.source.number_of_gens_in_bidegree(shift), 1);

    if matrix.rows() == 0 || matrix.columns() == 0 {
        panic!("No classes in this bidegree");
    }

    let mut v = vec![0; matrix.rows()];
    v[generator.idx()] = 1;
    matrix[generator.idx()].set_entry(0, 1);

    hom.extend_step(shift, Some(&matrix));
    let extend_max = shift + generator.degree() + TAU_BIDEGREE;
    let res_max = Bidegree::n_s(
        data.resolution.module(0).max_computed_degree(),
        data.resolution.next_homological_degree() - 1,
    );
    // This is the maximum bidegree of the region containing the entire stem of `generator` and
    // everything to its left.
    let extend_max = Bidegree::n_s(std::cmp::min(extend_max.n(), res_max.n()), res_max.s());
    hom.extend_through_stem(extend_max);

    if !data.is_unit {
        let res_max = Bidegree::n_s(
            data.resolution.module(0).max_computed_degree(),
            data.resolution.next_homological_degree() - 1,
        );
        data.unit.compute_through_stem(res_max - shift);
    }

    // Check that class survives to E3.
    {
        let m = data
            .res_lift
            .homotopy(shift.s() + 2)
            .homotopies
            .hom_k(shift.t());
        assert_eq!(m.len(), v.len());
        let mut sum = vec![0; m[0].len()];
        for (x, d2) in v.iter().zip_eq(&m) {
            sum.iter_mut().zip_eq(d2).for_each(|(a, b)| *a += x * b);
        }
        if sum.iter().any(|x| *x % data.p != 0) {
            eprintln!("{name} supports a non-zero d2, skipping");
            return Ok(());
        }
    }

    let hom_lift = SecondaryResolutionHomomorphism::new(
        Arc::clone(&data.res_lift),
        Arc::clone(&data.unit_lift),
        Arc::clone(&hom),
    );

    hom_lift.extend_all();

    let name = hom_lift.name();
    // Iterate through the multiplicand
    for b in data.unit.iter_stem().skip(1) {
        if (b.n(), b.s()) > (generator.n(), generator.s()) {
            // By symmetry, it's enough to compute products with cycles in degrees at most as large
            // as the generator (in the lexicographic order). The `iter_stem` iterator returns
            // bidegrees that are lexicographically increasing.
            return Ok(());
        }

        // The potential target has to be hit, and we need to have computed (the data need for) the
        // d2 that hits the potential target.
        if !data
            .resolution
            .has_computed_bidegree(b + shift + TAU_BIDEGREE)
        {
            continue;
        }
        if !data
            .resolution
            .has_computed_bidegree(b + shift - Bidegree::s_t(1, 0))
        {
            continue;
        }

        if data.unit.number_of_gens_in_bidegree(b) == 0 {
            continue;
        }

        let page_data = get_page_data(data.unit_sseq.as_ref(), b);

        let target_num_gens = data.resolution.number_of_gens_in_bidegree(b + shift);
        let tau_num_gens = data
            .resolution
            .number_of_gens_in_bidegree(b + shift + TAU_BIDEGREE);

        if target_num_gens == 0 && tau_num_gens == 0 {
            continue;
        }

        // First print the products with non-surviving classes
        if target_num_gens > 0 {
            let hom_k = hom.get_map((b + shift).s()).hom_k(b.t());
            for i in page_data
                .complement_pivots()
                .filter(|i| hom_k[*i].iter().any(|c| *c != 0))
            {
                let gen = BidegreeGenerator::new(b, i);
                println!("{name} τ x_{gen} = τ {:?}", &hom_k[i]);
            }
        }

        // Now print the secondary products
        if page_data.subspace_dimension() == 0 {
            continue;
        }

        let mut outputs = vec![
            FpVector::new(data.p, target_num_gens + tau_num_gens);
            page_data.subspace_dimension()
        ];

        hom_lift.hom_k(
            Some(&data.res_sseq),
            b,
            page_data.subspace_gens(),
            outputs.iter_mut().map(FpVector::as_slice_mut),
        );
        for (gen, output) in page_data
            .subspace_gens()
            .zip_eq(outputs)
            .filter(|(_, v)| v.iter().any(|x| x != 0))
        {
            println!(
                "{name} [{basis_string}] = {} + τ {}",
                output.slice(0, target_num_gens),
                output.slice(target_num_gens, target_num_gens + tau_num_gens),
                basis_string = BidegreeElement::new(b, gen.to_owned()).to_basis_string(),
            );
        }
    }
    Ok(())
}
