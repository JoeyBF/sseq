use std::fmt::{self, Display, Formatter};

use algebra::{
    module::{Module, MuFreeModule},
    MuAlgebra,
};
use fp::vector::{FpVector, Slice};

use crate::coordinates::{Bidegree, BidegreeGenerator};

/// An element of a bigraded vector space. Most commonly used to index elements of spectral
/// sequences.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BidegreeElement {
    /// Bidegree of the element
    degree: Bidegree,
    /// Representing vector
    vec: FpVector,
}

impl BidegreeElement {
    pub fn new(degree: Bidegree, vec: FpVector) -> BidegreeElement {
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

    pub fn vec(&self) -> Slice {
        self.vec.as_slice()
    }

    pub fn into_vec(self) -> FpVector {
        self.vec
    }

    pub fn decompose(&self) -> impl Iterator<Item = (BidegreeGenerator, u32)> + '_ {
        self.vec
            .iter_nonzero()
            .map(|(i, v)| (BidegreeGenerator::new(self.degree(), i), v))
    }

    /// Get the string representation of the element as a linear combination of generators.
    /// ```
    /// # use fp::{prime::P3, vector::FpVector};
    /// # use sseq::coordinates::{Bidegree, BidegreeElement};
    /// let v = FpVector::from_slice(P3, &[0, 2, 1]);
    /// let element = BidegreeElement::new(Bidegree::n_s(23, 5), v);
    ///
    /// assert_eq!(element.to_basis_string(), "2 x_(23, 5, 1) + x_(23, 5, 2)");
    /// ```
    pub fn to_basis_string(&self) -> String {
        self.decompose()
            .map(|(gen, v)| {
                let coeff_str = if v != 1 {
                    format!("{v} ")
                } else {
                    String::new()
                };
                format!("{coeff_str}x_{gen}")
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// An algebra-aware string representation. This assumes that the element belongs to `module`,
    /// and uses the string representation of its underlying algebra's operations.
    pub fn to_string_module<const U: bool, A: MuAlgebra<U>>(
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

impl Display for BidegreeElement {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.n(), self.s(), self.vec())
    }
}
