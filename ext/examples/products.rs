use std::sync::Arc;

use ext::{products::ProductStructure, utils::query_module};

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging();
    let resolution = query_module(None, true)?;

    let products = ProductStructure::new(Arc::new(resolution));
    products.compute_all_products();
    products.compute_all_massey_products();

    Ok(())
}
