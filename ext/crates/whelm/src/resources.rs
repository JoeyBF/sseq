//! Resource vectors and their dimensions.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Admission, JobSpec, ProductionAdmission, WorkerLoad, WorkerState, WorkerView};

/// Number of resource dimensions in a [`Resources`] vector.
///
/// # Examples
///
/// Per-dimension values, such as [`WorkerLoad::headroom`], are arrays indexed by dimension.
///
/// ```
/// use whelm::{DIMS, Resources};
///
/// let r = Resources::mem(10).with_dev(20);
/// let total: u64 = (0..DIMS).map(|d| r[d]).sum();
/// assert_eq!(total, 30);
/// ```
pub const DIMS: usize = 3;

/// The host-memory dimension of a [`Resources`] vector, in bytes.
pub const MEM: usize = 0;

/// The device-memory dimension of a [`Resources`] vector, in bytes.
pub const DEV: usize = 1;

/// The execution-slot dimension of a [`Resources`] vector.
///
/// Every job demands one slot: the scheduler sets that component of [`JobSpec::demand`] at
/// submission, whatever the caller wrote. A worker's slot capacity is [`WorkerState::slots`].
///
/// # Examples
///
/// The demands placed on a worker count its jobs in this dimension.
///
/// ```
/// use whelm::{Config, Input, JobSpec, Policy, Resources, SLOTS, Scheduler, Time, WorkerState};
///
/// let mut p = Scheduler::new(Config::default());
/// p.handle(
///     Input::Worker(WorkerState {
///         id: 1,
///         class: "cpu".into(),
///         slots: 4,
///         ..Default::default()
///     }),
///     Time::ZERO,
/// );
/// for id in 1..=3 {
///     p.handle(
///         Input::Submit(JobSpec {
///             id,
///             demand: Resources::mem(10),
///             ..Default::default()
///         }),
///         Time::ZERO,
///     );
/// }
/// p.poll(Time::ZERO);
/// let placed = p.stats().workers[0].placed;
/// assert_eq!((placed[SLOTS], placed[whelm::MEM]), (3, 30));
/// ```
pub const SLOTS: usize = 2;

/// Which dimensions are hard.
///
/// A hard dimension is always enforced, a zero capacity included, and has no escape hatch; a soft
/// one is enforced only where its capacity is known (nonzero), and a job alone on a worker ignores
/// it (see [`ProductionAdmission`]). Slots are counted exactly, so they are hard; memory figures
/// are estimates, so they are soft.
///
/// # Examples
///
/// ```
/// use whelm::{DEV, HARD, MEM, SLOTS};
///
/// assert_eq!((HARD[MEM], HARD[DEV], HARD[SLOTS]), (false, false, true));
/// ```
pub const HARD: [bool; DIMS] = {
    let mut hard = [false; DIMS];
    hard[SLOTS] = true;
    hard
};

/// An additive resource vector, one component per dimension ([`MEM`], [`DEV`], [`SLOTS`]).
///
/// Comparisons between vectors are component-wise ([`Resources::fits_within`]). As a capacity
/// ([`WorkerState::budget`]), a zero component of a soft dimension means its capacity is unknown
/// and not enforced, and a zero component of a [`HARD`] one means none (see
/// [`ProductionAdmission`]); as a demand, a zero component means none.
///
/// # Examples
///
/// Build vectors from host memory up, index them by dimension, and do saturating arithmetic.
///
/// ```
/// use whelm::{DEV, MEM, Resources, SLOTS};
///
/// let job = Resources::mem_gb(4.0).with_dev_gb(1.5);
/// assert_eq!(
///     (job[MEM], job[DEV], job[SLOTS]),
///     (4_000_000_000, 1_500_000_000, 0)
/// );
///
/// let two = job + job;
/// assert_eq!(two, job.saturating_mul(2));
/// assert_eq!(two - job, job);
/// assert_eq!(job - two, Resources::ZERO); // never below zero
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resources(pub [u64; DIMS]);

/// Bytes in a gigabyte (10^9), rounding to the nearest byte.
fn gb_bytes(gb: f64) -> u64 {
    (gb.max(0.0) * 1e9).round() as u64
}

impl Resources {
    /// The largest representable vector (an "unbounded" capacity).
    pub const MAX: Self = Self([u64::MAX; DIMS]);
    /// No resources.
    pub const ZERO: Self = Self([0; DIMS]);

    /// A vector with `bytes` of host memory and nothing else.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{MEM, Resources};
    ///
    /// assert_eq!(Resources::mem(1 << 30)[MEM], 1 << 30);
    /// ```
    pub const fn mem(bytes: u64) -> Self {
        let mut r = Self::ZERO;
        r.0[MEM] = bytes;
        r
    }

    /// A vector with `gb` gigabytes (10^9 bytes) of host memory, rounded to the nearest byte.
    ///
    /// Negative sizes count as zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// assert_eq!(Resources::mem_gb(1.5), Resources::mem(1_500_000_000));
    /// assert_eq!(Resources::mem_gb(-1.0), Resources::ZERO);
    /// ```
    pub fn mem_gb(gb: f64) -> Self {
        Self::mem(gb_bytes(gb))
    }

    /// This vector with `bytes` of device memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{DEV, MEM, Resources};
    ///
    /// let r = Resources::mem(8).with_dev(2);
    /// assert_eq!((r[MEM], r[DEV]), (8, 2));
    /// ```
    pub const fn with_dev(mut self, bytes: u64) -> Self {
        self.0[DEV] = bytes;
        self
    }

    /// This vector with `gb` gigabytes of device memory.
    ///
    /// # Examples
    ///
    /// A device-only vector, as for a worker's [`per_task`](WorkerState::per_task) floor.
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// assert_eq!(
    ///     Resources::ZERO.with_dev_gb(2.0),
    ///     Resources::ZERO.with_dev(2_000_000_000)
    /// );
    /// ```
    pub fn with_dev_gb(self, gb: f64) -> Self {
        self.with_dev(gb_bytes(gb))
    }

    /// This vector with `n` [`SLOTS`].
    ///
    /// The scheduler sets a job's slot component itself, so this is for working with demands as
    /// it sees them, such as testing an [`Admission`] rule against a [`WorkerView`].
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{MEM, Resources, SLOTS};
    ///
    /// let r = Resources::mem(8).with_slots(1);
    /// assert_eq!((r[MEM], r[SLOTS]), (8, 1));
    /// ```
    pub const fn with_slots(mut self, n: u64) -> Self {
        self.0[SLOTS] = n;
        self
    }

    /// Whether every component of `self` is at most the matching component of `cap`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// let cap = Resources::mem(10).with_dev(4);
    /// assert!(Resources::mem(10).fits_within(&cap));
    /// assert!(!Resources::mem(1).with_dev(5).fits_within(&cap));
    /// ```
    pub fn fits_within(&self, cap: &Self) -> bool {
        (0..DIMS).all(|d| self[d] <= cap[d])
    }

    /// The vector `f(self[d], other[d])` for every dimension `d`.
    fn zip(self, other: Self, f: impl Fn(u64, u64) -> u64) -> Self {
        Self(std::array::from_fn(|d| f(self[d], other[d])))
    }

    /// Component-wise maximum.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// let a = Resources::mem(10).with_dev(1);
    /// let b = Resources::mem(2).with_dev(5);
    /// assert_eq!(a.max(b), Resources::mem(10).with_dev(5));
    /// ```
    pub fn max(self, other: Self) -> Self {
        self.zip(other, u64::max)
    }

    /// Component-wise saturating addition; `+` is the same.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// assert_eq!(
    ///     Resources::mem(u64::MAX).saturating_add(Resources::mem(1)),
    ///     Resources::mem(u64::MAX)
    /// );
    /// ```
    pub fn saturating_add(self, other: Self) -> Self {
        self.zip(other, u64::saturating_add)
    }

    /// Component-wise saturating subtraction; `-` is the same.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// let r = Resources::mem(3)
    ///     .with_dev(9)
    ///     .saturating_sub(Resources::mem(5).with_dev(4));
    /// assert_eq!(r, Resources::ZERO.with_dev(5));
    /// ```
    pub fn saturating_sub(self, other: Self) -> Self {
        self.zip(other, u64::saturating_sub)
    }

    /// Every component multiplied by `n`, saturating.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// assert_eq!(Resources::mem(3).saturating_mul(4), Resources::mem(12));
    /// ```
    pub fn saturating_mul(self, n: u64) -> Self {
        Self(self.0.map(|x| x.saturating_mul(n)))
    }
}

impl std::ops::Index<usize> for Resources {
    type Output = u64;

    /// The component of dimension `d` ([`MEM`], [`DEV`], [`SLOTS`]).
    fn index(&self, d: usize) -> &u64 {
        &self.0[d]
    }
}

impl std::ops::IndexMut<usize> for Resources {
    /// The component of dimension `d` ([`MEM`], [`DEV`], [`SLOTS`]).
    fn index_mut(&mut self, d: usize) -> &mut u64 {
        &mut self.0[d]
    }
}

impl std::ops::Add for Resources {
    type Output = Self;

    /// Saturating, like [`Resources::saturating_add`].
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl std::ops::AddAssign for Resources {
    /// Saturating, like [`Resources::saturating_add`].
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::Sub for Resources {
    type Output = Self;

    /// Saturating: bookkeeping never goes below zero.
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl std::ops::SubAssign for Resources {
    /// Saturating, like [`Resources::saturating_sub`].
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}
