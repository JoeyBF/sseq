//! Points in time on the caller's clock, [`Time`].

use std::{
    ops::{Add, AddAssign, Sub, SubAssign},
    time::Duration,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::Policy;

/// A point in time: the [`Duration`] since the caller's clock origin.
///
/// A [`Policy`] never reads a clock; every call hands it the current `now`. Any origin works --
/// the start of the run, the Unix epoch -- as long as the caller keeps to it and `now` never
/// decreases. Unlike [`std::time::Instant`], a `Time` can be built from a number, so tests and
/// replays name the moments they mean, and a run fed the same times behaves the same. Spans are
/// [`Duration`]s, and `.0` reaches the span since the origin, with all of `Duration`'s methods.
///
/// Arithmetic saturates rather than panics: adding past the latest representable time stays
/// there, subtracting past the origin gives [`Time::ORIGIN`], and the difference of two times is
/// [`Time::duration_since`]. The `checked_*` methods say when that happened.
///
/// With feature `serde` a `Time` serialises as its `Duration`.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use whelm::Time;
///
/// let start = Time::ORIGIN + Duration::from_secs(30);
/// let end = start + Duration::from_millis(1500);
/// assert_eq!(end, Time(Duration::from_millis(31_500)));
/// assert_eq!(end - start, Duration::from_millis(1500));
/// assert_eq!(start - end, Duration::ZERO);
/// assert_eq!(end.0.as_secs_f64(), 31.5);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(transparent))]
pub struct Time(pub Duration);

impl Time {
    /// The caller's clock origin.
    pub const ORIGIN: Self = Self(Duration::ZERO);

    /// The span from `earlier` to `self`, or zero if `earlier` is later (also `self - earlier`).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let (a, b) = (Time(Duration::from_secs(5)), Time(Duration::from_secs(8)));
    /// assert_eq!(b.duration_since(a), Duration::from_secs(3));
    /// assert_eq!(a.duration_since(b), Duration::ZERO);
    /// ```
    pub fn duration_since(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    /// `d` after `self`, or `None` if that is not representable.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use whelm::Time;
    ///
    /// let t = Time(Duration::from_secs(1));
    /// assert_eq!(
    ///     t.checked_add(Duration::from_secs(1)),
    ///     Some(Time(Duration::from_secs(2)))
    /// );
    /// assert_eq!(
    ///     Time(Duration::MAX).checked_add(Duration::from_nanos(1)),
    ///     None
    /// );
    /// ```
    pub fn checked_add(self, d: Duration) -> Option<Self> {
        self.0.checked_add(d).map(Self)
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
    /// let t = Time(Duration::from_secs(1));
    /// assert_eq!(t.checked_sub(Duration::from_secs(1)), Some(Time::ORIGIN));
    /// assert_eq!(t.checked_sub(Duration::from_secs(2)), None);
    /// ```
    pub fn checked_sub(self, d: Duration) -> Option<Self> {
        self.0.checked_sub(d).map(Self)
    }
}

impl Add<Duration> for Time {
    type Output = Self;

    /// `d` after `self`, saturating at the latest representable time.
    fn add(self, d: Duration) -> Self {
        Self(self.0.saturating_add(d))
    }
}

impl AddAssign<Duration> for Time {
    /// Move `d` later, saturating at the latest representable time.
    fn add_assign(&mut self, d: Duration) {
        *self = *self + d;
    }
}

impl Sub<Duration> for Time {
    type Output = Self;

    /// `d` before `self`, saturating at [`Time::ORIGIN`].
    fn sub(self, d: Duration) -> Self {
        Self(self.0.saturating_sub(d))
    }
}

impl SubAssign<Duration> for Time {
    /// Move `d` earlier, saturating at [`Time::ORIGIN`].
    fn sub_assign(&mut self, d: Duration) {
        *self = *self - d;
    }
}

impl Sub for Time {
    type Output = Duration;

    /// The span from `earlier` to `self`, or zero ([`Time::duration_since`]).
    fn sub(self, earlier: Self) -> Duration {
        self.duration_since(earlier)
    }
}

/// `secs` seconds as a span, rounded to the nanosecond, for turning
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
        let end = Time(Duration::MAX);
        assert_eq!(end + d, end);
        assert_eq!(Time::ORIGIN - d, Time::ORIGIN);
        assert_eq!(Time::ORIGIN + Duration::MAX, end);
        assert_eq!(Time::ORIGIN - end, Duration::ZERO);
        assert_eq!(end - Time::ORIGIN, Duration::MAX);
        let mut t = end;
        t += d;
        assert_eq!(t, end);
        t = Time::ORIGIN;
        t -= d;
        assert_eq!(t, Time::ORIGIN);
    }

    /// `secs` rounds to the nanosecond and handles the edges.
    #[test]
    fn secs_edges() {
        assert_eq!(secs(1e-10), Duration::ZERO);
        assert_eq!(secs(1.5e-9), Duration::from_nanos(2));
        assert_eq!(secs(-1.0), Duration::ZERO);
        assert_eq!(secs(f64::NAN), Duration::ZERO);
        assert_eq!(secs(f64::INFINITY), Duration::MAX);
        assert_eq!(secs(0.5), Duration::from_millis(500));
    }

    /// A time serialises as its `Duration` and reads back exactly.
    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let t = Time(Duration::new(1234, 567_890_123));
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, serde_json::to_string(&t.0).unwrap());
        assert_eq!(serde_json::from_str::<Time>(&json).unwrap(), t);
        let end = Time(Duration::MAX);
        let json = serde_json::to_string(&end).unwrap();
        assert_eq!(serde_json::from_str::<Time>(&json).unwrap(), end);
    }
}
