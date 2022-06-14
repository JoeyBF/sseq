use std::sync::Arc;

use ext::{products::ProductStructure, utils::query_module};

fn main() -> anyhow::Result<()> {
    let resolution = query_module(None, false)?;

    let products = ProductStructure::new(Arc::new(resolution));
    products.compute_all_products();

    Ok(())
}
