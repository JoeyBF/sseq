//! Shared module-loading front end for the motivic examples — the motivic analogue
//! of [`crate::utils::load_module_json`] / [`crate::utils::query_module_only`].
//!
//! A motivic module is a `.json` descriptor over $A_C/\tau$ in the **native**
//! notation the motivic Steenrod algebra prints and parses (`Q_i`, `P(R)`), *not*
//! the admissible `Sq^k` of the classical `steenrod_modules/` library — a classical
//! module is not motivic (it carries no weights), so the motivic library is a
//! separate, deliberately duplicated set of descriptors in `steenrod_modules_motivic/`.
//! The bundled names are `S_2` (the sphere), `C2`, `Ceta`, `Cnu`, `Csigma` (the
//! Hopf-map cofibers); add more by dropping a descriptor into that directory or the
//! working directory.
//!
//! Every motivic example acquires its module and save directory through
//! [`query_motivic_module`], exactly as the classical examples call
//! [`crate::utils::query_module_only`], so they stay ~15-line wrappers over the
//! deformation pipeline ([`MotivicResolution::with_module`]).

use std::{path::PathBuf, sync::Arc};

use algebra::{CTauAlgebra, module::FDModule};
use serde_json::Value;

use super::MotivicResolution;

/// The bundled motivic module library, resolved relative to the crate at build time
/// (mirrors [`crate::utils`]'s `STATIC_MODULES_PATH`).
const STATIC_MOTIVIC_MODULES_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../ext/steenrod_modules_motivic");

/// Load a motivic module descriptor by `name`, searching (in order) the current
/// directory, `$CWD/steenrod_modules_motivic`, and the bundled library. Returns the
/// parsed JSON; see [`query_motivic_module`] to also build the [`FDModule`].
pub fn load_motivic_module_json(name: &str) -> anyhow::Result<Value> {
    let current_dir = std::env::current_dir()?;
    let relative_dir = current_dir.join("steenrod_modules_motivic");

    for dir in [
        current_dir,
        relative_dir,
        PathBuf::from(STATIC_MOTIVIC_MODULES_PATH),
    ] {
        let mut path = dir;
        path.push(name);
        path.set_extension("json");
        if let Ok(s) = std::fs::read_to_string(&path) {
            return serde_json::from_str(&s)
                .map_err(|e| anyhow::Error::new(e).context(format!("parsing module json {path:?}")));
        }
    }
    anyhow::bail!(
        "motivic module '{name}' not found (searched ., ./steenrod_modules_motivic, and the \
         bundled library). Available bundled: S_2, C2, Ceta, Cnu, Csigma."
    )
}

/// Build a motivic module from a named descriptor (see [`load_motivic_module_json`])
/// or, if `name` ends in `.json`, from that descriptor file directly.
pub fn motivic_module_by_name(name: &str) -> anyhow::Result<Arc<FDModule<CTauAlgebra>>> {
    let json = if name.ends_with(".json") {
        let s = std::fs::read_to_string(name)
            .map_err(|e| anyhow::Error::new(e).context(format!("reading module file {name}")))?;
        serde_json::from_str(&s)?
    } else {
        load_motivic_module_json(name)?
    };
    MotivicResolution::module_from_json(&json)
}

/// Query the user for a motivic module and its save directory — the motivic
/// [`crate::utils::query_module_only`]. Prompts `"{prompt}"` for the module name
/// (defaulting to `default`, e.g. `"S_2"` for the sphere) and `"{prompt} save
/// directory"` for an optional ZarrV3 store to cache the resolution + lift.
///
/// Returns the built module and the save directory; the caller supplies the box and
/// calls [`MotivicResolution::with_module`] (the box is a per-example prompt, so it
/// is not read here — mirroring the classical split between module choice and the
/// `Max n`/`Max s` prompts).
pub fn query_motivic_module(
    prompt: &str,
    default: &str,
) -> anyhow::Result<(Arc<FDModule<CTauAlgebra>>, Option<PathBuf>)> {
    let module = query::with_default(prompt, default, |name: &str| motivic_module_by_name(name));
    let save_dir = query::optional(&format!("{prompt} save directory"), |s: &str| {
        Ok::<_, std::convert::Infallible>(PathBuf::from(s))
    });
    Ok((module, save_dir))
}
