use std::fmt::{self, Display, Formatter};

use fp::vector::{FpCow, FpSlice, FpVector};

use crate::coordinates::{Bidegree, BidegreeGenerator};

/// An element of a bigraded vector space. Most commonly used to index elements of spectral
/// sequences.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BidegreeElement<'a> {
    /// Bidegree of the element
    degree: Bidegree,
    /// Representing vector
    vec: FpCow<'a>,
}

impl BidegreeElement<'static> {
    pub fn new(degree: Bidegree, vec: FpVector) -> Self {
        Self::from_cow(degree, vec.into_cow())
    }

    pub fn into_vec(self) -> FpVector {
        self.vec.into_vec()
    }
}

impl<'a> BidegreeElement<'a> {
    pub fn from_cow(degree: Bidegree, vec: FpCow<'a>) -> Self {
        Self { degree, vec }
    }

    pub fn from_slice(degree: Bidegree, vec: FpSlice<'a>) -> Self {
        Self::from_cow(degree, vec.into_cow())
    }

    pub fn s(&self) -> i32 {
        self.degree.s()
    }

    pub fn t(&self) -> i32 {
        self.degree.t()
    }

    pub fn degree(&self) -> Bidegree {
        self.degree
    }

    pub fn n(&self) -> i32 {
        self.degree.n()
    }

    pub fn x(&self) -> i32 {
        self.degree.x()
    }

    pub fn y(&self) -> i32 {
        self.degree.y()
    }

    pub fn vec(&self) -> FpSlice<'_> {
        self.vec.as_slice()
    }

    /// Get the string representation of the element as a linear combination of generators. For
    /// example, an element in bidegree `(n,s)` with vector `[0,2,1]` will be printed as `2 x_(n, s,
    /// 1) + x_(n, s, 2)`.
    pub fn to_basis_string(&self) -> String {
        self.vec
            .iter_nonzero()
            .map(|(i, v)| {
                let g = BidegreeGenerator::new(self.degree(), i);
                let coeff_str = if v != 1 {
                    format!("{v} ")
                } else {
                    String::new()
                };
                format!("{coeff_str}x_{g}")
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

impl<'a> Display for BidegreeElement<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if f.alternate() {
            write!(f, "({},{}){}", self.n(), self.s(), self.vec())
        } else {
            write!(f, "({}, {}, {})", self.n(), self.s(), self.vec())
        }
    }
}
