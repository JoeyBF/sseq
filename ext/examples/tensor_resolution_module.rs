//! Computes `Ext_A(M, k)` of a finite module `M` from a resolution of `k` and prints the chart.
//!
//! `A` is the algebra of the module spec, so a profiled spec such as `tmf-mod-eta` gives Ext over
//! `A(2)`. Only the base field `k` is resolved (by Nassau's algorithm when the `nassau` feature is
//! on), and `M` enters through the untwisted tensor resolution
//! [`tensor_resolution_ext`](ext::ext_algebra::tensor_resolution_ext). One save directory for `k`
//! therefore serves every module over the same algebra.
//!
//! The chart has the layout of `resolve_through_stem`, so the two outputs can be compared with
//! `diff`: rows are `s` from the top, columns are `n` from the bottom degree of `M`, and leading empty rows are dropped.

use std::{path::PathBuf, sync::Arc};

use algebra::module::{FDModule, Module};
use ext::{
    chain_complex::ChainComplex,
    ext_algebra::tensor_resolution_ext,
    utils::{construct, parse_module_name, unicode_num},
};
use serde_json::{Value, json};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let (name, module_json): (String, Value) = query::with_default("Module", "C2", |s| {
        Result::<_, anyhow::Error>::Ok((s.to_owned(), parse_module_name(s)?))
    });
    let save_dir = query::optional("Base field save directory", |x| {
        core::result::Result::<PathBuf, std::convert::Infallible>::Ok(PathBuf::from(x))
    });
    let max_n: i32 = query::with_default("Max n", "30", str::parse);
    let max_s: i32 = query::with_default("Max s", "15", str::parse);

    // The base field over the same algebra as `M`: the spec of `M` with its cells replaced by one.
    let mut k_json = json!({
        "p": module_json["p"],
        "type": "finite dimensional module",
        "gens": { "x0": 0 },
        "actions": [],
    });
    if let Some(algebra) = module_json.get("algebra") {
        k_json["algebra"] = algebra.clone();
    }
    let mut k_res = construct((k_json, "milnor"), save_dir)?;
    k_res.set_name(format!("k for {name}"));
    let k_res = Arc::new(k_res);

    // The cohomology at `(n, s)` reads the coboundaries out of `s - 1` and `s` at `t = n + s`.
    k_res.compute_through_bidegree(Bidegree::s_t(max_s + 1, max_n + max_s));

    let module = Arc::new(FDModule::from_json(k_res.algebra(), &module_json)?);
    let min_n = module.min_degree();
    let ext = tensor_resolution_ext(Arc::clone(&k_res), module);

    let mut chart = String::new();
    for s in (0..=max_s).rev() {
        for n in min_n..=max_n {
            let dim = ext
                .cohomology_dimension(Bidegree::n_s(n, s))
                .expect("bidegree is in the resolved range");
            chart.push(unicode_num(dim));
            chart.push(' ');
        }
        chart.push('\n');
        if chart.trim_start().is_empty() {
            chart.clear();
        }
    }
    println!("{chart}");

    Ok(())
}
