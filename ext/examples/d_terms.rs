//! Resolves a module through (Max n, Max s) and prints, per bidegree (s, t) with t <= Max t, the
//! number of generators and the number of nonzero terms of d(g) over the target free module's F2
//! basis (Milnor basis element x target generator pairs).
//!
//! Output CSV on stdout: `s,t,ngens,total_terms,max_terms,target_dim`.
use algebra::module::Module;
use ext::chain_complex::ChainComplex;
use sseq::coordinates::Bidegree;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let res = ext::utils::query_module_only("Module", None, false)?;
    let max = Bidegree::n_s(
        query::with_default("Max n", "130", str::parse),
        query::with_default("Max s", "20", str::parse),
    );
    let max_t: i32 = query::with_default("Max t", "150", str::parse);

    let start = Instant::now();
    res.compute_through_stem(max);
    eprintln!("resolve_time_s={:.2}", start.elapsed().as_secs_f64());

    println!("s,t,ngens,total_terms,max_terms,target_dim");
    for s in 1..=max.s() {
        let d = res.differential(s);
        let source = res.module(s);
        let target = res.module(s - 1);
        for t in s..=std::cmp::min(max_t, max.n() + s) {
            let ngens = source.number_of_gens_in_degree(t);
            if ngens == 0 {
                continue;
            }
            let mut total = 0usize;
            let mut mx = 0usize;
            for i in 0..ngens {
                let c = d.output(t, i).iter_nonzero().count();
                total += c;
                mx = mx.max(c);
            }
            let tdim = if t <= target.max_computed_degree() {
                target.dimension(t) as i64
            } else {
                -1
            };
            println!("{s},{t},{ngens},{total},{mx},{tdim}");
        }
    }
    Ok(())
}
