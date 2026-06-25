//! Computes massey products in $\Mod_{C\lambda^2}$.
//!
//! # Usage
//! This computes all Massey products of the form $\langle -, b, a\rangle$, where $a \in \Ext^{\*,
//! \*}(M, k)$ and $b, (-) \in \Ext^{\*, \*}(k, k)$. It does not verify that the Massey product is
//! valid, i.e. $a$ and $b$ both lift to $\Mod_{C\lambda^2}$ and have trivial product.
//!
//! Since we must choose $a$ and $b$ to have trivial product, it is necessary to be able to specify
//! the $\lambda$ part of them, and not insist that they are standard lifts of the $\Ext$ classes.
//! Thus, the user is first prompted for the $\Ext$ part, then the $\lambda$ part of each class. To
//! set a part to zero, supply an empty name. Note that if the bidegree right above the class is
//! empty, the user is not prompted for the $\lambda$ part.
//!
//! # Output
//! This computes the Massey products up to a sign. We write our output in the category
//! $\Mod_{C\lambda^2}$, so the format is $\langle a, b, -\rangle$ instead of $\langle -, b,
//! a\rangle$. Brave souls are encouraged to figure out the correct sign for the products.

use std::sync::Arc;

use ext::{
    chain_complex::{ChainComplex, FreeChainComplex},
    ext_algebra::{ExtAlgebra, secondary::SecondaryExtAlgebra},
    secondary::LAMBDA_BIDEGREE,
    utils::{QueryModuleResolution, query_module},
};
use fp::{prime::ValidPrime, vector::FpVector};
use itertools::Itertools;
use sseq::coordinates::{Bidegree, BidegreeElement};

/// The result of prompting for one input class: its display name, the $\Ext$ part, and an optional
/// $\lambda$ part (one filtration up).
struct InputClass {
    name: String,
    ext: BidegreeElement,
    lambda: Option<BidegreeElement>,
}

/// Prompt for an input class as in the original `get_hom`: the bidegree, the $\Ext$-part name and
/// coordinates, then (if the bidegree above is nonempty) the $\lambda$-part name and coordinates.
/// `source` is the resolution the class lives over.
fn query_class(name: &str, source: &Arc<QueryModuleResolution>, p: ValidPrime) -> InputClass {
    let shift = Bidegree::n_s(
        query::raw(&format!("n of {name}"), str::parse),
        query::raw(&format!("s of {name}"), str::parse),
    );

    let ext_name: String = query::raw(&format!("Name of Ext part of {name}"), str::parse);

    source.compute_through_stem(shift + LAMBDA_BIDEGREE);

    let num_gens = source.number_of_gens_in_bidegree(shift);
    let num_lambda_gens = source.number_of_gens_in_bidegree(shift + LAMBDA_BIDEGREE);

    let mut ext = FpVector::new(p, num_gens);
    if !ext_name.is_empty() {
        if num_gens == 0 {
            eprintln!("No classes in this bidegree");
        } else {
            let v: Vec<u32> = query::vector(&format!("Input Ext class {ext_name}"), num_gens);
            for (i, &x) in v.iter().enumerate() {
                ext.set_entry(i, x);
            }
        }
    }

    let (lambda_name, lambda) = if num_lambda_gens > 0 {
        let lambda_name: String = query::raw(&format!("Name of λ part of {name}"), str::parse);
        if lambda_name.is_empty() {
            (String::new(), None)
        } else {
            let v = query::vector(&format!("Input Ext class {lambda_name}"), num_lambda_gens);
            let mut lambda = FpVector::new(p, num_lambda_gens);
            for (i, &x) in v.iter().enumerate() {
                lambda.set_entry(i, x);
            }
            (
                lambda_name,
                Some(BidegreeElement::new(shift + LAMBDA_BIDEGREE, lambda)),
            )
        }
    } else {
        (String::new(), None)
    };

    let name = match (&*ext_name, &*lambda_name) {
        ("", "") => panic!("Do not compute zero Massey product"),
        ("", x) => format!("λ{x}"),
        (x, "") => format!("[{x}]"),
        (x, y) => format!("[{x}] + λ{y}"),
    };

    InputClass {
        name,
        ext: BidegreeElement::new(shift, ext),
        lambda,
    }
}

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    eprintln!(
        "We are going to compute <-, b, a> for all (-), where a is an element in Ext(M, k) and b \
         and (-) are elements in Ext(k, k)."
    );

    let resolution = Arc::new(query_module(Some(algebra::AlgebraType::Milnor), true)?);

    let (is_unit, unit) = ext::utils::get_unit(Arc::clone(&resolution))?;

    let p = resolution.prime();

    let alg = Arc::new(ExtAlgebra::new(
        Arc::clone(&resolution),
        Arc::clone(&unit),
        is_unit,
    ));
    let sec = SecondaryExtAlgebra::new(Arc::clone(&alg));

    // `a` lives in Ext(M, k); `b` and the third factor live in Ext(k, k).
    let a = query_class("a", &resolution, p);
    let b = query_class("b", &unit, p);

    let a_name = &a.name;
    let b_name = &b.name;

    let massey = sec.secondary_massey(&a.ext, a.lambda.as_ref(), &b.ext, b.lambda.as_ref());

    if let Some(s) = ext::utils::secondary_job() {
        massey.compute_partial(s);
        return Ok(());
    }

    massey.extend();

    for r in massey.family() {
        print!("<{a_name}, {b_name}, ");
        let c = r.third_factor.degree();
        let has_ext = {
            if r.third_factor.vec().iter_nonzero().count() > 0 {
                print!(
                    "[{basis_string}]",
                    basis_string = r.third_factor.to_basis_string()
                );
                true
            } else {
                false
            }
        };

        let num_entries = r.third_factor_lambda.iter_nonzero().count();
        if num_entries > 0 {
            if has_ext {
                print!(" + ");
            }
            print!("λ");

            let basis_string =
                BidegreeElement::new(c + LAMBDA_BIDEGREE, r.third_factor_lambda.clone())
                    .to_basis_string();
            if num_entries == 1 {
                print!("{basis_string}");
            } else {
                print!("({basis_string})");
            }
        }
        print!("> = ±");

        print!("[{}]", r.representative.vec().iter().format(", "));
        println!(" + λ{}", r.representative_lambda);
    }
    Ok(())
}
