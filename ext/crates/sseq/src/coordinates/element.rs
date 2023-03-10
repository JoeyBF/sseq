use crate::coordinates::{Bidegree, BidegreeGenerator};

use std::fmt::{self, Display, Formatter};

use algebra::{
    module::{Module, MuFreeModule},
    MuAlgebra,
};
use fp::vector::{prelude::*, FpVector, Slice};

/// An element of a bigraded vector space. Most commonly used to index elements of spectral
/// sequences.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BidegreeElement<T> {
    /// Bidegree of the element
    degree: Bidegree,
    /// Representing vector
    vec: T,
}

impl<T> BidegreeElement<T> {
    pub fn new(degree: Bidegree, vec: T) -> Self {
        BidegreeElement { degree, vec }
    }

    pub fn s(&self) -> u32 {
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

    pub fn vec(&self) -> &T {
        &self.vec
    }
}

impl<T: BaseVector> BidegreeElement<T> {
    pub fn into_owned(self) -> BidegreeElement<FpVector> {
        BidegreeElement {
            degree: self.degree(),
            vec: self.vec.into_owned(),
        }
    }

    pub fn to_slice(&self) -> BidegreeElement<Slice> {
        BidegreeElement {
            degree: self.degree,
            vec: self.vec().as_slice(),
        }
    }

    pub fn decompose(&self) -> impl Iterator<Item = (BidegreeGenerator, u32)> + '_ {
        self.vec()
            .iter_nonzero()
            .map(|(idx, coeff)| (BidegreeGenerator::new(self.degree(), idx), coeff))
    }

    /// Prints the element to stdout. For example, an element in bidegree `(n,s)` with vector
    /// `[0,2,1]` will be printed as `2 x_(n, s, 1) + x_(n, s, 2)`.
    pub fn print(&self) {
        let output = self
            .vec
            .iter_nonzero()
            .map(|(i, v)| {
                let gen = BidegreeGenerator::new(self.degree(), i);
                let coeff_str = if v != 1 {
                    format!("{v} ")
                } else {
                    String::new()
                };
                format!("{coeff_str}x_{gen}")
            })
            .collect::<Vec<_>>()
            .join(" + ");
        print!("{output}");
    }

    /// An algebra-aware string representation. This assumes that the element belongs to `module`,
    /// and uses the string representation of its underlying algebra's operations.
    pub fn to_string_pretty<const U: bool, A: MuAlgebra<U>>(
        &self,
        module: &MuFreeModule<U, A>,
        compact: bool,
    ) -> String {
        self.vec
            .iter_nonzero()
            .map(|(i, c)| {
                let coeff_str = if c != 1 {
                    format!("{c} ")
                } else {
                    String::new()
                };
                let opgen = module.index_to_op_gen(self.t(), i);
                let mut op_str = module
                    .algebra()
                    .basis_element_to_string(opgen.operation_degree, opgen.operation_index);
                op_str = if op_str != "1" {
                    format!("{op_str} ")
                } else {
                    String::new()
                };
                let gen = BidegreeGenerator::s_t(
                    self.s() - 1,
                    opgen.generator_degree,
                    opgen.generator_index,
                );
                let gen_str = if compact {
                    format!("x_{gen:#}")
                } else {
                    format!("x_{gen}")
                };
                format!("{coeff_str}{op_str}{gen_str}")
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

impl<T: Display> Display for BidegreeElement<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.n(), self.s(), self.vec())
    }
}
