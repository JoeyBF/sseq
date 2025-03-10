use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, Sub},
};

use serde::{Deserialize, Serialize};

/// Type synonym for (s, t) bidegrees.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bidegree {
    /// Homological degree
    s: i32,
    /// Internal degree
    t: i32,
}

impl Bidegree {
    pub const fn s_t(s: i32, t: i32) -> Self {
        Self { s, t }
    }

    pub const fn t_s(t: i32, s: i32) -> Self {
        Self::s_t(s, t)
    }

    pub const fn n_s(n: i32, s: i32) -> Self {
        Self::s_t(s, n + s)
    }

    pub const fn zero() -> Self {
        Self { s: 0, t: 0 }
    }

    pub fn s(&self) -> i32 {
        self.s
    }

    pub fn t(&self) -> i32 {
        self.t
    }

    pub fn n(&self) -> i32 {
        self.t - self.s
    }

    /// Returns difference as a bidegree if the difference in homological degrees is nonnegative,
    /// otherwise returns None.
    pub fn try_subtract(&self, smaller: Self) -> Option<Self> {
        if self.s >= smaller.s {
            Some(*self - smaller)
        } else {
            None
        }
    }

    /// Computes the bidegree containing the Massey product of elements in the given bidegrees.
    ///
    /// # Panics
    /// Panics if every element is in homological degree 0. This is the only case that would result
    /// in a bidegree in negative homological degree.
    pub fn massey_bidegree(a: Self, b: Self, c: Self) -> Self {
        (a + b + c)
            .try_subtract(Self::s_t(1, 0))
            .expect("Trying to compute Massey product of elements in s = 0")
    }
}

impl Display for Bidegree {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.n(), self.s())
    }
}

impl Add for Bidegree {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            s: self.s + other.s,
            t: self.t + other.t,
        }
    }
}

impl Sub for Bidegree {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            s: self.s - other.s,
            t: self.t - other.t,
        }
    }
}
