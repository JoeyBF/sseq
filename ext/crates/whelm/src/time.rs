//! Points in time on the caller's clock, [`Time`].

use std::{
    fmt,
    ops::{Add, AddAssign, Sub, SubAssign},
    time::Duration,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(doc)]
use crate::Policy;

/// A point in time: the span since an origin the caller picks, at nanosecond resolution.
///
/// A [`Policy`] never reads a clock; every call hands it the current `now`. Any origin works --
/// the start of the run, the Unix epoch -- as long as the caller keeps to it and `now` never
/// decreases. Unlike [`std::time::Instant`], a `Time` can be built from a number, so tests and
/// replays name the moments they mean, and a run fed the same times behaves the same.
///
/// Spans are [`Duration`]s. Arithmetic saturates rather than panics: adding past [`Time::MAX`]
/// gives `Time::MAX`, subtracting past the origin gives [`Time::ZERO`], and the difference of two
/// times is the span from the earlier to the later, or zero if the first is the earlier (as
/// [`Instant::duration_since`](std::time::Instant::duration_since)). The `checked_*` methods say
/// when that happened.
///
/// With feature `serde` it serialises as an integer number of nanoseconds, so a logged time reads
/// back exactly.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use whelm::Time;
///
/// let start = Time::from_secs(30);
/// let end = start + Duration::from_millis(1500);
/// assert_eq!(end, Time::from_millis(31_500));
/// assert_eq!(end - start, Duration::from_millis(1500));
/// assert_eq!(start - end, Duration::ZERO);
/// assert_eq!(end.as_secs_f64(), 31.5);
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(Duration);

impl Time {
    /// The latest representable time, about 584 years after the origin: the largest whose
    /// nanosecond count fits a `u64`, so that every `Time` serialises as one.
    pub const MAX: Self = Self(Duration::from_nanos(u64::MAX));
    /// The origin.
    pub const ZERO: Self = Self(Duration::ZERO);

    /// `secs` seconds after the origin, saturating at [`Time::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_secs(2), Time::from_millis(2000));
    /// assert_eq!(Time::from_secs(u64::MAX), Time::MAX);
    /// ```
    pub const fn from_secs(secs: u64) -> Self {
        Self::from_nanos_u128(secs as u128 * 1_000_000_000)
    }

    /// `millis` milliseconds after the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_millis(2500).as_secs_f64(), 2.5);
    /// ```
    pub const fn from_millis(millis: u64) -> Self {
        Self::from_nanos_u128(millis as u128 * 1_000_000)
    }

    /// `nanos` nanoseconds after the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_nanos(1_500_000_000), Time::from_millis(1500));
    /// assert_eq!(Time::from_nanos(u64::MAX), Time::MAX);
    /// ```
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(Duration::from_nanos(nanos))
    }

    /// `secs` seconds after the origin, rounded to the nanosecond and saturating at
    /// [`Time::MAX`] (so `f64::INFINITY` gives `Time::MAX`).
    ///
    /// # Panics
    ///
    /// If `secs` is negative or NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_secs_f64(0.25), Time::from_millis(250));
    /// assert_eq!(Time::from_secs_f64(f64::INFINITY), Time::MAX);
    /// ```
    pub fn from_secs_f64(secs: f64) -> Self {
        assert!(
            secs >= 0.0,
            "Time::from_secs_f64: {secs} is negative or NaN"
        );
        let nanos = (secs * 1e9).round();
        if nanos >= u64::MAX as f64 {
            Self::MAX
        } else {
            Self::from_nanos(nanos as u64)
        }
    }

    /// `nanos` nanoseconds after the origin, saturating at [`Time::MAX`].
    const fn from_nanos_u128(nanos: u128) -> Self {
        if nanos > u64::MAX as u128 {
            Self::MAX
        } else {
            Self::from_nanos(nanos as u64)
        }
    }

    /// The span since the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_secs(3).since_origin(), Duration::from_secs(3));
    /// ```
    pub const fn since_origin(self) -> Duration {
        self.0
    }

    /// Seconds since the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_millis(1250).as_secs_f64(), 1.25);
    /// ```
    pub const fn as_secs_f64(self) -> f64 {
        self.0.as_secs_f64()
    }

    /// Nanoseconds since the origin (what a `Time` serialises as).
    ///
    /// # Examples
    ///
    /// ```
    /// use whelm::Time;
    ///
    /// assert_eq!(Time::from_millis(3).as_nanos(), 3_000_000);
    /// ```
    pub const fn as_nanos(self) -> u64 {
        // `MAX` bounds every `Time`, so this does not truncate.
        self.0.as_nanos() as u64
    }

    /// The span from `earlier` to `self`, or zero if `earlier` is later (also `self - earlier`).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let (a, b) = (Time::from_secs(5), Time::from_secs(8));
    /// assert_eq!(b.since(a), Duration::from_secs(3));
    /// assert_eq!(a.since(b), Duration::ZERO);
    /// ```
    pub fn since(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    /// The span from `earlier` to `self`, or `None` if `earlier` is later.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let (a, b) = (Time::from_secs(5), Time::from_secs(8));
    /// assert_eq!(b.checked_since(a), Some(Duration::from_secs(3)));
    /// assert_eq!(a.checked_since(b), None);
    /// ```
    pub fn checked_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0)
    }

    /// `d` after `self`, or `None` past [`Time::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let t = Time::from_secs(1);
    /// assert_eq!(
    ///     t.checked_add(Duration::from_secs(1)),
    ///     Some(Time::from_secs(2))
    /// );
    /// assert_eq!(Time::MAX.checked_add(Duration::from_nanos(1)), None);
    /// ```
    pub fn checked_add(self, d: Duration) -> Option<Self> {
        let nanos = self.0.as_nanos() + d.as_nanos();
        (nanos <= u64::MAX as u128).then(|| Self::from_nanos(nanos as u64))
    }

    /// `d` before `self`, or `None` before the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let t = Time::from_secs(1);
    /// assert_eq!(t.checked_sub(Duration::from_secs(1)), Some(Time::ZERO));
    /// assert_eq!(t.checked_sub(Duration::from_secs(2)), None);
    /// ```
    pub fn checked_sub(self, d: Duration) -> Option<Self> {
        self.0.checked_sub(d).map(Self)
    }
}

impl fmt::Debug for Time {
    /// The span since the origin, as [`Duration`] prints it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Time({:?})", self.0)
    }
}

impl fmt::Display for Time {
    /// Seconds since the origin, as `12.5s`; a precision applies to the seconds.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match f.precision() {
            Some(p) => write!(f, "{:.*}s", p, self.as_secs_f64()),
            None => write!(f, "{}s", self.as_secs_f64()),
        }
    }
}

impl Add<Duration> for Time {
    type Output = Self;

    /// `d` after `self`, saturating at [`Time::MAX`].
    fn add(self, d: Duration) -> Self {
        self.checked_add(d).unwrap_or(Self::MAX)
    }
}

impl AddAssign<Duration> for Time {
    /// Move `d` later, saturating at [`Time::MAX`].
    fn add_assign(&mut self, d: Duration) {
        *self = *self + d;
    }
}

impl Sub<Duration> for Time {
    type Output = Self;

    /// `d` before `self`, saturating at [`Time::ZERO`].
    fn sub(self, d: Duration) -> Self {
        self.checked_sub(d).unwrap_or(Self::ZERO)
    }
}

impl SubAssign<Duration> for Time {
    /// Move `d` earlier, saturating at [`Time::ZERO`].
    fn sub_assign(&mut self, d: Duration) {
        *self = *self - d;
    }
}

impl Sub for Time {
    type Output = Duration;

    /// The span from `earlier` to `self`, or zero ([`Time::since`]).
    fn sub(self, earlier: Self) -> Duration {
        self.since(earlier)
    }
}

#[cfg(feature = "serde")]
impl Serialize for Time {
    /// Nanoseconds since the origin.
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(self.as_nanos())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Time {
    /// From nanoseconds since the origin.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        u64::deserialize(d).map(Self::from_nanos)
    }
}

/// `secs` seconds as a span, rounded to the nanosecond like [`Time::from_secs_f64`], for turning
/// f64 arithmetic back into a [`Duration`]: negative and NaN give zero, and anything too long for
/// a `Duration` gives [`Duration::MAX`].
pub(crate) fn secs(secs: f64) -> Duration {
    let nanos = (secs * 1e9).round();
    if nanos.is_nan() || nanos <= 0.0 {
        Duration::ZERO
    } else if nanos < u64::MAX as f64 {
        Duration::from_nanos(nanos as u64)
    } else {
        Duration::try_from_secs_f64(secs).unwrap_or(Duration::MAX)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Time, secs};

    /// Arithmetic saturates at both ends.
    #[test]
    fn saturates() {
        let d = Duration::from_secs(1);
        assert_eq!(Time::MAX + d, Time::MAX);
        assert_eq!(Time::ZERO - d, Time::ZERO);
        assert_eq!(Time::ZERO + Duration::MAX, Time::MAX);
        assert_eq!(Time::from_secs(u64::MAX / 2), Time::MAX);
        assert_eq!(Time::ZERO - Time::MAX, Duration::ZERO);
    }

    /// The f64 conversions round to the nanosecond and handle the edges.
    #[test]
    fn f64_edges() {
        assert_eq!(Time::from_secs_f64(1e-10), Time::ZERO);
        assert_eq!(Time::from_secs_f64(1.5e-9), Time::from_nanos(2));
        assert_eq!(Time::from_secs_f64(1e30), Time::MAX);
        assert_eq!(secs(-1.0), Duration::ZERO);
        assert_eq!(secs(f64::NAN), Duration::ZERO);
        assert_eq!(secs(f64::INFINITY), Duration::MAX);
        assert_eq!(secs(0.5), Duration::from_millis(500));
    }

    /// Negative seconds are a caller's bug.
    #[test]
    #[should_panic(expected = "is negative or NaN")]
    fn negative_panics() {
        Time::from_secs_f64(-1.0);
    }

    /// Display prints seconds.
    #[test]
    fn display() {
        assert_eq!(Time::from_millis(12_500).to_string(), "12.5s");
        assert_eq!(format!("{:.1}", Time::from_millis(1234)), "1.2s");
    }

    /// A time serialises as integer nanoseconds and reads back exactly.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_nanos() {
        let t = Time::from_nanos(1_234_567_890_123);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "1234567890123");
        assert_eq!(serde_json::from_str::<Time>(&json).unwrap(), t);
        assert_eq!(
            serde_json::to_string(&Time::MAX).unwrap(),
            u64::MAX.to_string()
        );
    }
}
