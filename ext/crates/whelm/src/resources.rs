//! Resource declarations, and the vectors of amounts indexed by them.
//!
//! A [`Config`] declares the resources its workers have and its jobs demand, as a list of
//! [`Resource`]s; a [`ResourceId`] names one by its position in that list, and a [`Resources`]
//! vector holds one amount per declared resource. The [default declaration](Config::resources)
//! is host memory, device memory and execution slots, and [`Resources`] has shorthands for it.
//!
//! Here a pool of four software licenses per worker joins the default declaration: a hard
//! resource no job ever exceeds, of which jobs take none unless they say so.
//!
//! ```
//! use whelm::{Config, Resource, ResourceId, Resources};
//!
//! let mut config = Config::default();
//! let licenses = ResourceId(config.resources.len());
//! config.resources.push(Resource {
//!     name: "licenses".into(),
//!     hard: true,
//!     ..Default::default()
//! });
//!
//! let capacity = Resources::mem_gb(64.0).with_slots(8).with(licenses, 4);
//! let demand = Resources::mem_gb(2.0).with(licenses, 1);
//! assert_eq!((capacity[licenses], demand[licenses]), (4, 1));
//! assert_eq!(config.resources[licenses.0].name, "licenses");
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    Admission, Config, Explanation, JobSpec, ProductionAdmission, ScoreTerm, WorkerLoad,
    WorkerState, WorkerView,
};

/// How a [`Resource`]'s amounts read when an [`Explanation`] is displayed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ResourceUnit {
    /// A count of things, shown as it is.
    #[default]
    Count,
    /// Bytes, shown in gigabytes (10^9 bytes).
    Bytes,
}

/// One resource dimension, as [`Config::resources`] declares it.
///
/// A **hard** resource is never exceeded: a worker whose capacity of it is zero admits no job that
/// demands any. A **soft** one holds estimates: a zero capacity means unknown and is not
/// enforced, and a job alone on a worker runs whatever its demand (the escape hatch, so that
/// every job can run somewhere). [`ProductionAdmission`] has the rule.
///
/// Nothing checks that a declaration makes sense. A hard resource that a worker leaves at zero
/// capacity keeps every job that demands it off that worker, and one with a nonzero
/// [`default_demand`](Self::default_demand) keeps every job off it: under the default
/// declaration, a worker without slots runs nothing.
///
/// # Examples
///
/// The default declaration, from the named constructors:
///
/// ```
/// use whelm::{Config, Resource};
///
/// let slots = Resource::slots();
/// assert!(slots.hard && slots.default_demand == 1);
/// let memory = Resource::memory("memory");
/// assert!(!memory.hard && memory.default_demand == 0);
/// assert_eq!(
///     Config::default().resources,
///     [memory, Resource::memory("device memory"), slots]
/// );
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resource {
    /// What an [`Explanation`] calls it.
    pub name: String,
    /// Whether it is hard rather than soft.
    pub hard: bool,
    /// The demand of a job whose [`JobSpec::demand`] leaves this component at zero, filled in at
    /// submission. One for slots, so that every job takes one.
    pub default_demand: u64,
    /// How its amounts are shown.
    pub unit: ResourceUnit,
}

impl Resource {
    /// Execution slots: hard, one per job unless the job demands more.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resource;
    ///
    /// assert_eq!(Resource::slots().name, "slots");
    /// ```
    pub fn slots() -> Self {
        Self {
            name: "slots".into(),
            hard: true,
            default_demand: 1,
            unit: ResourceUnit::Count,
        }
    }

    /// A soft memory pool called `name`, in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{Resource, ResourceUnit};
    ///
    /// let scratch = Resource::memory("scratch disk");
    /// assert_eq!((scratch.hard, scratch.unit), (false, ResourceUnit::Bytes));
    /// ```
    pub fn memory(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hard: false,
            default_demand: 0,
            unit: ResourceUnit::Bytes,
        }
    }
}

/// A resource, by its position in [`Config::resources`].
///
/// The associated constants name the default declaration's resources.
///
/// # Examples
///
/// ```
/// use whelm::{Config, ResourceId};
///
/// let config = Config::default();
/// assert_eq!(config.resources[ResourceId::DEV.0].name, "device memory");
/// assert!(config.resources[ResourceId::SLOTS.0].hard);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResourceId(pub usize);

impl ResourceId {
    /// Device memory in the default declaration, in bytes: the pool jobs' device allocations come
    /// from.
    pub const DEV: Self = Self(1);
    /// Host memory in the default declaration, in bytes.
    pub const MEM: Self = Self(0);
    /// Execution slots in the default declaration.
    pub const SLOTS: Self = Self(2);
}

/// An additive vector of resource amounts, indexed by [`ResourceId`].
///
/// Components beyond the vector's length read as zero, so a vector built for the first few
/// resources of a declaration is valid for all of it, and equality ignores trailing zeros.
/// Comparisons and arithmetic are component-wise, and arithmetic saturates. As a capacity
/// ([`WorkerState::capacity`]) a zero component means none of a hard resource and an unknown
/// amount of a soft one; as a demand, it means none, until the scheduler fills in the
/// [`default_demand`](Resource::default_demand).
///
/// The shorthands [`mem`](Self::mem), [`mem_gb`](Self::mem_gb), [`with_dev`](Self::with_dev),
/// [`with_dev_gb`](Self::with_dev_gb) and [`with_slots`](Self::with_slots) set the components of
/// the default declaration; [`of`](Self::of) and [`with`](Self::with) set any.
///
/// # Examples
///
/// Build vectors from host memory up, index them by resource, and do saturating arithmetic.
///
/// ```
/// use whelm::{ResourceId, Resources};
///
/// let job = Resources::mem_gb(4.0).with_dev_gb(1.5);
/// assert_eq!(
///     (
///         job[ResourceId::MEM],
///         job[ResourceId::DEV],
///         job[ResourceId::SLOTS]
///     ),
///     (4_000_000_000, 1_500_000_000, 0)
/// );
///
/// let two = job.clone() + &job;
/// assert_eq!(two, job.clone().saturating_mul(2));
/// assert_eq!(two.clone() - &job, job);
/// assert_eq!(job - two, Resources::ZERO); // never below zero
/// ```
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resources(Vec<u64>);

/// Bytes in a gigabyte (10^9), rounding to the nearest byte.
fn gb_bytes(gb: f64) -> u64 {
    (gb.max(0.0) * 1e9).round() as u64
}

impl Resources {
    /// No resources.
    pub const ZERO: Self = Self(Vec::new());

    /// The vector with `amount` in each of the first `n` resources.
    ///
    /// # Examples
    ///
    /// An unbounded capacity for a declaration of three resources, as
    /// [`Admission::bound`] returns by default:
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// let all = Resources::repeat(u64::MAX, 3);
    /// assert!(Resources::mem(1 << 40).with_slots(9).fits_within(&all));
    /// assert_eq!(all[ResourceId(3)], 0);
    /// ```
    pub fn repeat(amount: u64, n: usize) -> Self {
        Self(vec![amount; n])
    }

    /// The vector with these amounts, zero elsewhere; a resource named twice takes its last
    /// amount.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// let gpus = ResourceId(3);
    /// let r = Resources::of([(ResourceId::SLOTS, 2), (gpus, 1)]);
    /// assert_eq!(r, Resources::ZERO.with_slots(2).with(gpus, 1));
    /// ```
    pub fn of(amounts: impl IntoIterator<Item = (ResourceId, u64)>) -> Self {
        amounts
            .into_iter()
            .fold(Self::ZERO, |r, (id, amount)| r.with(id, amount))
    }

    /// This vector with `amount` of resource `id`.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// let licenses = ResourceId(3);
    /// assert_eq!(Resources::mem(8).with(licenses, 2)[licenses], 2);
    /// ```
    pub fn with(mut self, id: ResourceId, amount: u64) -> Self {
        self[id] = amount;
        self
    }

    /// A vector with `bytes` of host memory and nothing else, for the default declaration.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// assert_eq!(Resources::mem(1 << 30)[ResourceId::MEM], 1 << 30);
    /// ```
    pub fn mem(bytes: u64) -> Self {
        Self::ZERO.with(ResourceId::MEM, bytes)
    }

    /// A vector with `gb` gigabytes (10^9 bytes) of host memory, rounded to the nearest byte, for
    /// the default declaration.
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

    /// This vector with `bytes` of device memory, for the default declaration.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// let r = Resources::mem(8).with_dev(2);
    /// assert_eq!((r[ResourceId::MEM], r[ResourceId::DEV]), (8, 2));
    /// ```
    pub fn with_dev(self, bytes: u64) -> Self {
        self.with(ResourceId::DEV, bytes)
    }

    /// This vector with `gb` gigabytes of device memory, for the default declaration.
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

    /// This vector with `n` slots, for the default declaration.
    ///
    /// A worker's [`capacity`](WorkerState::capacity) needs slots to run anything. A job's demand
    /// takes one slot unless it says otherwise, so a demand needs them only to take several, or
    /// to test an [`Admission`] rule against a [`WorkerView`] with demands as the scheduler sees
    /// them.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::{ResourceId, Resources};
    ///
    /// let r = Resources::mem(8).with_slots(1);
    /// assert_eq!((r[ResourceId::MEM], r[ResourceId::SLOTS]), (8, 1));
    /// ```
    pub fn with_slots(self, n: u64) -> Self {
        self.with(ResourceId::SLOTS, n)
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
        (0..self.0.len()).all(|d| self[ResourceId(d)] <= cap[ResourceId(d)])
    }

    /// `self[d] = f(self[d], other[d])` for every resource `d` of either.
    fn zip(mut self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (d, x) in self.0.iter_mut().enumerate() {
            *x = f(*x, other[ResourceId(d)]);
        }
        self
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
        self.zip(&other, u64::max)
    }

    /// Component-wise saturating addition; `+` is the same.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Resources;
    ///
    /// assert_eq!(
    ///     Resources::mem(u64::MAX).saturating_add(&Resources::mem(1)),
    ///     Resources::mem(u64::MAX)
    /// );
    /// ```
    pub fn saturating_add(self, other: &Self) -> Self {
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
    ///     .saturating_sub(&Resources::mem(5).with_dev(4));
    /// assert_eq!(r, Resources::ZERO.with_dev(5));
    /// ```
    pub fn saturating_sub(self, other: &Self) -> Self {
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
    pub fn saturating_mul(mut self, n: u64) -> Self {
        for x in &mut self.0 {
            *x = x.saturating_mul(n);
        }
        self
    }

    /// Drop the components of resource `n` and beyond.
    pub(crate) fn truncate(&mut self, n: usize) {
        self.0.truncate(n);
    }

    /// The components up to the last nonzero one.
    fn trimmed(&self) -> &[u64] {
        let len = self.0.iter().rposition(|&x| x != 0).map_or(0, |d| d + 1);
        &self.0[..len]
    }
}

impl PartialEq for Resources {
    /// Component-wise, so trailing zeros do not matter.
    fn eq(&self, other: &Self) -> bool {
        self.trimmed() == other.trimmed()
    }
}

impl Eq for Resources {}

impl std::hash::Hash for Resources {
    /// The components up to the last nonzero one, as [`PartialEq`] compares them.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.trimmed().hash(state);
    }
}

impl std::ops::Index<ResourceId> for Resources {
    type Output = u64;

    /// The amount of resource `d`; zero beyond the vector's length.
    fn index(&self, d: ResourceId) -> &u64 {
        self.0.get(d.0).unwrap_or(&0)
    }
}

impl std::ops::IndexMut<ResourceId> for Resources {
    /// The amount of resource `d`, lengthening the vector with zeros to reach it.
    fn index_mut(&mut self, d: ResourceId) -> &mut u64 {
        if self.0.len() <= d.0 {
            self.0.resize(d.0 + 1, 0);
        }
        &mut self.0[d.0]
    }
}

impl std::ops::Add for Resources {
    type Output = Self;

    /// Saturating, like [`Resources::saturating_add`].
    fn add(self, other: Self) -> Self {
        self.saturating_add(&other)
    }
}

impl std::ops::Add<&Resources> for Resources {
    type Output = Self;

    /// Saturating, like [`Resources::saturating_add`].
    fn add(self, other: &Self) -> Self {
        self.saturating_add(other)
    }
}

impl std::ops::AddAssign<&Resources> for Resources {
    /// Saturating, like [`Resources::saturating_add`].
    fn add_assign(&mut self, other: &Self) {
        *self = std::mem::take(self).saturating_add(other);
    }
}

impl std::ops::AddAssign for Resources {
    /// Saturating, like [`Resources::saturating_add`].
    fn add_assign(&mut self, other: Self) {
        *self += &other;
    }
}

impl std::ops::Sub for Resources {
    type Output = Self;

    /// Saturating: bookkeeping never goes below zero.
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(&other)
    }
}

impl std::ops::Sub<&Resources> for Resources {
    type Output = Self;

    /// Saturating, like [`Resources::saturating_sub`].
    fn sub(self, other: &Self) -> Self {
        self.saturating_sub(other)
    }
}

impl std::ops::SubAssign<&Resources> for Resources {
    /// Saturating, like [`Resources::saturating_sub`].
    fn sub_assign(&mut self, other: &Self) {
        *self = std::mem::take(self).saturating_sub(other);
    }
}

impl std::ops::SubAssign for Resources {
    /// Saturating, like [`Resources::saturating_sub`].
    fn sub_assign(&mut self, other: Self) {
        *self -= &other;
    }
}
