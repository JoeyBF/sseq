pub use degree::MultiDegree;
pub use element::MultiDegreeElement;
pub use generator::MultiDegreeGenerator;
use maybe_rayon::prelude::*;
use ordered::OrderedMultiDegree;
pub use range::BidegreeRange;

pub mod degree;
pub mod element;
pub mod generator;
pub mod ordered;
pub mod range;

/// Outcome of a single computation step in a bidegree iteration.
///
/// This makes the intent of the computation explicit, replacing the previous approach of
/// encoding control flow in return ranges.
///
/// # Variants
///
/// - `Computed(range)`: Successfully computed data. The range indicates which `t` values
///   in `s+1` are now contiguous and should be processed next.
/// - `AlreadyProcessed`: The step had nothing to do; it was already computed.
/// - `Skipped`: Deliberately skipped this step (e.g., error condition met, unreachable case).
#[derive(Debug, Clone)]
pub enum ComputeOutcome {
    /// Successfully computed. The range indicates newly-contiguous t-values for the next s.
    Computed(std::ops::Range<i32>),
    /// The step was already processed; no new computation was done.
    AlreadyProcessed,
    /// Deliberately skipped this step (e.g., error condition met, unreachable).
    Skipped,
}

pub type Bidegree = MultiDegree<2>;
pub type BidegreeElement = MultiDegreeElement<2>;
pub type BidegreeGenerator = MultiDegreeGenerator<2>;
pub type OrderedBidegree<O> = OrderedMultiDegree<2, O>;

impl Bidegree {
    pub const fn n_s(n: i32, s: i32) -> Self {
        Self::new([n, s])
    }

    pub const fn s_t(s: i32, t: i32) -> Self {
        Self::n_s(t - s, s)
    }

    pub const fn x_y(x: i32, y: i32) -> Self {
        Self::n_s(x, y)
    }
}

impl BidegreeGenerator {
    pub fn s_t(s: i32, t: i32, idx: usize) -> Self {
        Self::new(Bidegree::s_t(s, t), idx)
    }

    pub fn n_s(n: i32, s: i32, idx: usize) -> Self {
        Self::new(Bidegree::n_s(n, s), idx)
    }
}

/// Execute a function on a range of bidegrees, possibly in parallel.
///
/// Given a function `f(s, t)`, compute it for every `s` in `[min_s, max_s]` and every `t` in
/// `[min_t, max_t(s)]`.  Further, we only compute `f(s, t)` when `f(s - 1, t')` has been computed
/// for all `t' < t`.
///
/// The function `f` should return a [`ComputeOutcome`] indicating what happened:
/// - `ComputeOutcome::Computed(range)`: The computation succeeded and produced data. The range
///   indicates which `t` values in `s+1` should be processed next.
/// - `ComputeOutcome::Skipped`: The step was skipped (e.g., already computed, error condition).
///   No further steps will be spawned for this branch.
///
/// This uses [`maybe_rayon`] under the hood, and `f` should feel free to use further parallelism.
///
/// # Arguments:
///  - `max_s`: This is exclusive
///  - `max_t`: This is exclusive
pub fn iter_s_t<T: Sync>(
    f: &(impl Fn(Bidegree) -> ComputeOutcome + Sync),
    min: Bidegree,
    max: BidegreeRange<T>,
) {
    // Track `tracing` spans correctly
    let tracing_span = tracing::Span::current();
    let f = &|b| {
        let _tracing_guard = tracing_span.enter();
        f(b)
    };

    maybe_rayon::scope(|scope| {
        // Rust does not support recursive closures, so we have to pass everything along as
        // arguments.
        fn run<'a, S: Sync>(
            scope: &maybe_rayon::Scope<'a>,
            f: &'a (impl Fn(Bidegree) -> ComputeOutcome + Sync + 'a),
            max: BidegreeRange<'a, S>,
            current: Bidegree,
        ) {
            let outcome = f(current);
            let mut ret = match outcome {
                ComputeOutcome::Computed(range) => range,
                ComputeOutcome::AlreadyProcessed => {
                    current.t()..current.t() + 1
                }
                ComputeOutcome::Skipped => {
                    current.t()..current.t()
                }
            };

            if current.s() + 1 < max.s() {
                ret.start += 1;
                ret.end = std::cmp::min(ret.end + 1, max.t(current.s() + 1));

                if !ret.is_empty() {
                    // We spawn a new scope to avoid recursion, which may blow the stack
                    scope.spawn(move |scope| {
                        ret.into_maybe_par_iter()
                            .for_each(|t| run(scope, f, max, Bidegree::s_t(current.s() + 1, t)));
                    });
                }
            }
        }

        maybe_rayon::join(
            || {
                (min.t()..max.t(min.s()))
                    .into_maybe_par_iter()
                    .for_each(|t| run(scope, f, max, Bidegree::s_t(min.s(), t)))
            },
            || {
                (min.s() + 1..max.s())
                    .into_maybe_par_iter()
                    .for_each(|s| run(scope, f, max, Bidegree::s_t(s, min.t())))
            },
        );
    });
}

#[cfg(test)]
mod tests {
    use fp::{prime::ValidPrime, vector::FpVector};

    use super::{Bidegree, BidegreeElement, BidegreeGenerator};

    #[test]
    fn test_bidegree_generator_try_from_element() {
        let b = Bidegree::n_s(23, 9);
        let mut vec = FpVector::new(ValidPrime::new(2), 2);
        vec.set_entry(1, 1);
        let h1_pd0 = BidegreeElement::new(b, vec.clone());
        assert_eq!(Ok(BidegreeGenerator::new(b, 1)), h1_pd0.try_into());
        vec.set_entry(0, 1);
        let h0_squared_i = BidegreeElement::new(b, vec.clone());
        assert_eq!(
            Result::<BidegreeGenerator, ()>::Err(()),
            h0_squared_i.try_into()
        );
    }
}
