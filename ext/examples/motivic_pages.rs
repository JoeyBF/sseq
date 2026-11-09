//! Prints the pages of the deformation spectral sequence of a module over $A_C/\tau$, as CSV.

use std::sync::Arc;

use ext::motivic::MotivicResolution;
use sseq::coordinates::degree::MultiDegree;

/// Print `n,s,w,dim E_1,dim E_2,…` for every `(n, s, w)` in the box with classes on $E_1$.
///
/// Page $r$ runs up to `Max s`, past which no $d_r$ in the box can be nonzero. Dimensions are
/// independent of the minimal resolution chosen, so runs on different engines compare line by
/// line.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let module: String = query::with_default("Module", "Ctau", |s| {
        Ok::<_, std::convert::Infallible>(s.to_owned())
    });
    let max = sseq::coordinates::Bidegree::n_s(
        query::with_default("Max n", "30", str::parse),
        query::with_default("Max s", "17", str::parse),
    );
    // Nassau's resolution keeps its quasi-inverses only on disk, so it needs a save directory.
    let save_dir = query::optional("Save directory", |s| {
        Ok::<_, std::convert::Infallible>(std::path::PathBuf::from(s))
    });
    #[cfg(feature = "nassau")]
    let resolution = {
        anyhow::ensure!(
            save_dir.is_some(),
            "Nassau's algorithm needs a save directory for the lift's quasi-inverses"
        );
        ext::utils::construct_nassau(module.as_str(), save_dir)?
    };
    #[cfg(not(feature = "nassau"))]
    let resolution = ext::utils::construct_standard::<false, _, _>(module.as_str(), save_dir)?;
    let res = MotivicResolution::new(Arc::new(resolution), max)?;
    let sseq = res.deformation_sseq();

    // Ext at the top filtration would read generators one filtration higher.
    let (max_n, max_s) = (res.max().n(), res.max().s() - 1);
    let mut rows = Vec::new();
    for b in sseq.iter_degrees() {
        let [n, s, w] = b.coords();
        if !(0..=max_n).contains(&n) || !(0..=max_s).contains(&s) {
            continue;
        }
        let deg = MultiDegree::from([n, s, w]);
        let dims: Vec<usize> = (1..=max_s + 1)
            .map(|r| sseq.page_data(deg).get_max(r).dimension())
            .collect();
        if dims[0] > 0 {
            rows.push(((n, s, w), dims));
        }
    }
    rows.sort();
    for ((n, s, w), dims) in rows {
        let dims: Vec<String> = dims.iter().map(ToString::to_string).collect();
        println!("{n},{s},{w},{}", dims.join(","));
    }
    Ok(())
}
