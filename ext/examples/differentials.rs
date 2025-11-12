//! This prints all the differentials in the resolution.

use ext::{chain_complex::ChainComplex, utils::query_module};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;

    let resolution = query_module(None, false)?;

    for s in 0..resolution.next_homological_degree() {
        println!("{}", resolution.differential(s));
    }

    Ok(())
}
