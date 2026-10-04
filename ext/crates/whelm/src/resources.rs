//! Resource kinds, and the amounts of them that jobs demand and workers have.
//!
//! A [`Resource`] describes one kind of resource: its name, whether it is hard or soft, what a job
//! takes of it by default, and how its amounts read. [`Config::resources`] declares the ones a
//! scheduler knows, and a [`Resources`] value holds amounts of them keyed by name. The default
//! declaration is [`MEMORY`], [`DEVICE_MEMORY`] and [`SLOTS`].
//!
//! Here workers count their GPUs: a hard resource that jobs take none of unless they say so.
//!
//! ```
//! use whelm::{Config, MEMORY, Resource, Resources, SLOTS, gb};
//!
//! const GPUS: Resource = Resource::new("gpus").hard();
//!
//! let mut config = Config::default();
//! config.resources.push(GPUS);
//!
//! let capacity = Resources::new()
//!     .with(MEMORY, gb(64.0))
//!     .with(SLOTS, 8)
//!     .with(GPUS, 4);
//! let demand = Resources::new().with(MEMORY, gb(2.0)).with(GPUS, 1);
//! assert_eq!((capacity.get(GPUS), demand.get(GPUS)), (4, 1));
//! assert_eq!(demand.get(SLOTS), 0); // the scheduler fills in the default demand
//! ```

use std::{borrow::Cow, fmt};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{
    Admission, Config, Explanation, JobSpec, Output, ProductionAdmission, Scheduler, WorkerState,
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

/// One kind of resource: what [`Config::resources`] declares, and what [`Resources`] amounts are
/// keyed by.
///
/// A resource is identified by its [`name`](Self::name) alone. Its other fields are rules, and a
/// scheduler takes them from its own declaration only: a [`Resources`] value keeps the name of the
/// resource it was built with and nothing else, so two values with the same name and different
/// rules are the same resource, under the rules [`Config::resources`] gives it.
///
/// A **hard** resource is never exceeded: a worker whose capacity of it is zero admits no job that
/// demands any. A **soft** one holds estimates: a zero capacity means unknown and is not
/// enforced, and a job alone on a worker runs whatever its demand (the escape hatch, so that
/// every job can run somewhere). [`ProductionAdmission`] has the rule.
///
/// Nothing checks that a declaration makes sense. A hard resource that a worker leaves at zero
/// capacity keeps every job that demands it off that worker, and one with a nonzero
/// [`default_demand`](Self::default_demand) keeps every job off it: under the default
/// declaration, a worker without [`SLOTS`] runs nothing.
///
/// # Examples
///
/// A resource is usually a constant, built with the `const` builder methods. A name known only at
/// run time goes in the [`name`](Self::name) field.
///
/// ```
/// use whelm::{Resource, ResourceUnit};
///
/// const LICENSES: Resource = Resource::new("licenses").hard();
/// const SCRATCH: Resource = Resource::new("scratch").unit(ResourceUnit::Bytes);
/// assert!(LICENSES.hard && LICENSES.default_demand == 0);
/// assert!(!SCRATCH.hard);
///
/// let pool = Resource {
///     name: format!("pool {}", 3).into(),
///     ..Resource::new("")
/// };
/// assert_eq!(pool.name, "pool 3");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resource {
    /// The resource's identity, and what an [`Explanation`] calls it.
    pub name: Cow<'static, str>,
    /// Whether it is hard rather than soft.
    pub hard: bool,
    /// What a job whose [`JobSpec::demand`] leaves this resource out takes of it, filled in at
    /// submission.
    pub default_demand: u64,
    /// How its amounts are shown.
    pub unit: ResourceUnit,
}

impl Resource {
    /// A soft resource called `name`, counted, of which jobs take none unless they say so.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            hard: false,
            default_demand: 0,
            unit: ResourceUnit::Count,
        }
    }

    /// This resource, made hard.
    pub const fn hard(mut self) -> Self {
        self.hard = true;
        self
    }

    /// This resource, of which a job that says nothing takes `amount`.
    pub const fn default_demand(mut self, amount: u64) -> Self {
        self.default_demand = amount;
        self
    }

    /// This resource, with its amounts shown in `unit`.
    pub const fn unit(mut self, unit: ResourceUnit) -> Self {
        self.unit = unit;
        self
    }
}

impl AsRef<Resource> for Resource {
    /// The resource itself, so that methods taking `impl AsRef<Resource>` accept a [`Resource`]
    /// constant as well as a reference.
    fn as_ref(&self) -> &Resource {
        self
    }
}

/// Host memory, in bytes: soft.
pub const MEMORY: Resource = Resource::new("memory").unit(ResourceUnit::Bytes);

/// Device memory, in bytes: soft. The pool jobs' device allocations come from.
pub const DEVICE_MEMORY: Resource = Resource::new("device memory").unit(ResourceUnit::Bytes);

/// Execution slots: hard, one per job unless the job demands more.
///
/// A worker's [`capacity`](WorkerState::capacity) needs slots to run anything, under a declaration
/// that has them.
pub const SLOTS: Resource = Resource::new("slots").hard().default_demand(1);

/// `x` gigabytes (10^9 bytes) in bytes, rounded to the nearest byte; negative sizes count as zero.
///
/// # Examples
///
/// ```
/// use whelm::gb;
///
/// assert_eq!(gb(1.5), 1_500_000_000);
/// assert_eq!(gb(-1.0), 0);
/// ```
pub fn gb(x: f64) -> u64 {
    (x.max(0.0) * 1e9).round() as u64
}

/// Amounts of resources, keyed by resource name.
///
/// A resource it leaves out has amount zero, and setting an amount to zero leaves it out, so
/// equality compares the nonzero amounts. As a capacity ([`WorkerState::capacity`]) a zero amount
/// means none of a hard resource and an unknown amount of a soft one; as a demand
/// ([`JobSpec::demand`]), it means the resource's [`default_demand`](Resource::default_demand).
///
/// Names must be declared in [`Config::resources`]: a [`Scheduler`] rejects a job whose demand
/// names another ([`Output::Rejected`]) and panics on a worker whose state does.
///
/// It serialises (feature `serde`) as a map from name to amount.
///
/// # Examples
///
/// ```
/// use whelm::{DEVICE_MEMORY, MEMORY, Resources, SLOTS, gb};
///
/// let demand = Resources::new()
///     .with(MEMORY, gb(4.0))
///     .with(DEVICE_MEMORY, gb(1.5));
/// assert_eq!(
///     (
///         demand.get(MEMORY),
///         demand.get(&DEVICE_MEMORY),
///         demand.get(SLOTS)
///     ),
///     (4_000_000_000, 1_500_000_000, 0)
/// );
///
/// // The last amount set wins, and zero is the same as nothing.
/// let r = Resources::new()
///     .with(SLOTS, 2)
///     .with(SLOTS, 3)
///     .with(MEMORY, 0);
/// assert_eq!(r, Resources::new().with(SLOTS, 3));
/// assert_eq!(r.iter().collect::<Vec<_>>(), [("slots", 3)]);
/// ```
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Resources(Vec<(Cow<'static, str>, u64)>);

impl Resources {
    /// No resources.
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// These amounts with `amount` of `resource`, replacing any amount it had.
    pub fn with(mut self, resource: impl AsRef<Resource>, amount: u64) -> Self {
        self.set(resource.as_ref().name.clone(), amount);
        self
    }

    /// The amount of `resource`; zero if it is left out.
    pub fn get(&self, resource: impl AsRef<Resource>) -> u64 {
        let name = &resource.as_ref().name;
        self.find(name).map_or(0, |i| self.0[i].1)
    }

    /// Each resource's name and nonzero amount, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(name, x)| (&**name, *x))
    }

    /// Whether every amount is zero.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Where `name` is among the amounts, or else where it would go.
    fn find(&self, name: &str) -> Result<usize, usize> {
        self.0.binary_search_by(|(n, _)| (**n).cmp(name))
    }

    /// Set the amount of the resource called `name`.
    pub(crate) fn set(&mut self, name: Cow<'static, str>, amount: u64) {
        match (self.find(&name), amount) {
            (Ok(i), 0) => {
                self.0.remove(i);
            }
            (Ok(i), _) => self.0[i].1 = amount,
            (Err(_), 0) => {}
            (Err(i), _) => self.0.insert(i, (name, amount)),
        }
    }

    /// The names and amounts, in name order.
    pub(crate) fn entries(&self) -> &[(Cow<'static, str>, u64)] {
        &self.0
    }
}

impl FromIterator<(Resource, u64)> for Resources {
    /// The amounts as [`with`](Resources::with) sets them one by one: a resource named twice takes
    /// its last amount.
    fn from_iter<I: IntoIterator<Item = (Resource, u64)>>(iter: I) -> Self {
        let mut r = Self::new();
        for (resource, amount) in iter {
            r.set(resource.name, amount);
        }
        r
    }
}

impl fmt::Debug for Resources {
    /// As a map from name to amount.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(feature = "serde")]
impl Serialize for Resources {
    /// As a map from name to amount.
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_map(self.iter())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Resources {
    /// From a map from name to amount.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let map = std::collections::BTreeMap::<String, u64>::deserialize(d)?;
        let mut r = Self::new();
        for (name, amount) in map {
            r.set(name.into(), amount);
        }
        Ok(r)
    }
}

/// Amounts of the resources of a declaration, by position in it: the scheduler's working form of
/// a [`Resources`] value, converted once on the way in.
pub(crate) type Dense = Vec<u64>;

/// The position of the resource called `name` in `resources`.
pub(crate) fn position(resources: &[Resource], name: &str) -> Option<usize> {
    resources.iter().position(|r| r.name == name)
}

/// `r` over the declaration `resources`, or the first name (in name order) that it does not
/// declare.
pub(crate) fn dense(resources: &[Resource], r: &Resources) -> Result<Dense, Cow<'static, str>> {
    let mut out = vec![0; resources.len()];
    for (name, amount) in r.entries() {
        let d = position(resources, name).ok_or_else(|| name.clone())?;
        out[d] = *amount;
    }
    Ok(out)
}

/// The amounts `v` over the declaration `resources`, keyed by name again.
pub(crate) fn named(resources: &[Resource], v: &[u64]) -> Resources {
    let mut r = Resources::new();
    for (res, &amount) in resources.iter().zip(v) {
        r.set(res.name.clone(), amount);
    }
    r
}

/// `a[d] += b[d]` for every `d`, saturating.
pub(crate) fn add(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x = x.saturating_add(*y);
    }
}

/// `a[d] -= b[d]` for every `d`, saturating: bookkeeping never goes below zero.
pub(crate) fn sub(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x = x.saturating_sub(*y);
    }
}

/// Whether `a[d] <= b[d]` for every `d`.
pub(crate) fn fits_within(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Setting, overwriting and clearing keep the amounts sorted by name and free of zeros.
    #[test]
    fn set_get_iter() {
        const B: Resource = Resource::new("b");
        let r = Resources::new()
            .with(SLOTS, 1)
            .with(B, 2)
            .with(MEMORY, 3)
            .with(B, 4)
            .with(SLOTS, 0);
        assert_eq!(r.iter().collect::<Vec<_>>(), [("b", 4), ("memory", 3)]);
        assert_eq!((r.get(B), r.get(&SLOTS)), (4, 0));
        let collected: Resources = [(B, 4), (MEMORY, 9), (MEMORY, 3)].into_iter().collect();
        assert_eq!(collected, r);
        assert_eq!(format!("{r:?}"), r#"{"b": 4, "memory": 3}"#);
    }

    /// The same name with other rules is the same resource.
    #[test]
    fn identity_is_the_name() {
        let other = Resource::new("slots").default_demand(7);
        assert_eq!(Resources::new().with(SLOTS, 2).get(other), 2);
    }

    /// Conversion to and from the dense form.
    #[test]
    fn dense_round_trip() {
        let decl = [MEMORY, DEVICE_MEMORY, SLOTS];
        let r = Resources::new().with(SLOTS, 2).with(MEMORY, 5);
        let v = dense(&decl, &r).unwrap();
        assert_eq!(v, [5, 0, 2]);
        assert_eq!(named(&decl, &v), r);
        let stray = r
            .with(Resource::new("gpus"), 1)
            .with(Resource::new("zz"), 1);
        assert_eq!(dense(&decl, &stray), Err("gpus".into()));
    }

    /// A map in name order, zeros dropped on the way in.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_map() {
        let r = Resources::new().with(SLOTS, 1).with(MEMORY, 5);
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(json, r#"{"memory":5,"slots":1}"#);
        let back: Resources = serde_json::from_str(r#"{"slots":1,"memory":5,"gpus":0}"#).unwrap();
        assert_eq!(back, r);
    }
}
