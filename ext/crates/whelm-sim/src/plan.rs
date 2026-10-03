//! Speed-aware placement and machine models, as the simulators' plans name them.

use whelm::{ScoreTerm, SpeedConfig, Timing};

/// Speed-aware placement, as a plan names it.
///
/// Whether workers are ranked by speed, and the scheduler's [`SpeedConfig`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpeedPlan {
    /// Rank the workers that admit a job fastest first ([`ScoreTerm::Speed`]).
    ///
    /// [`SpeedConfig::defer`] and [`SpeedConfig::speculate`] read speed either way.
    pub fast: bool,
    /// The machine model, deferral and speculation.
    pub config: SpeedConfig,
}

impl SpeedPlan {
    /// The scheduler's score: `[Speed, Preferred, Load]` when `fast`, else `[Preferred, Load]`.
    pub fn score(&self) -> Vec<ScoreTerm> {
        let mut s = vec![ScoreTerm::Preferred, ScoreTerm::Load];
        if self.fast {
            s.insert(0, ScoreTerm::Speed);
        }
        s
    }

    /// Whether speeds are learned.
    ///
    /// Simulated workers then report speed 1, so that learning starts from no knowledge.
    pub fn learned(&self) -> bool {
        self.config.timing.learn().is_some()
    }

    /// A plan-name suffix describing it (empty for speed-oblivious placement on reported speeds).
    pub fn name(&self) -> String {
        let mut s = match (self.fast, self.config.defer) {
            (_, Some(d)) => format!(
                ", earliest finish (wait <= {:.0}s)",
                d.max_wait.as_secs_f64()
            ),
            (true, None) => ", fast first".into(),
            (false, None) => String::new(),
        };
        if self.config.speculate.is_some() {
            s += ", speculative";
        }
        s += match self.config.timing {
            Timing::Identical => ", identical machines",
            Timing::Related { learn: None } => "",
            Timing::Related { learn: Some(_) } => ", learned speeds",
            Timing::Unrelated { .. } => ", learned speeds per kind",
        };
        s
    }
}

/// The machine model a command-line name stands for.
///
/// `p` (identical: speed ignored), `q` (related, at the reported speeds), `q-learn` (related,
/// learned) or `r` (unrelated: learned per job kind and worker class).
pub fn timing_named(name: &str) -> Result<Timing, String> {
    match name {
        "p" => Ok(Timing::Identical),
        "q" => Ok(Timing::default()),
        "q-learn" => Ok(Timing::learned()),
        "r" => Ok(Timing::unrelated()),
        other => Err(format!("unknown timing {other} (p, q, q-learn or r)")),
    }
}
