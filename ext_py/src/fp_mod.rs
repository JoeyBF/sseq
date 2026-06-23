use pyo3::prelude::*;

#[pymodule]
pub mod fp_py {
    use fp::field::{element::FieldElement, Field, Fp as RustFp, SmallFq as RustSmallFq};
    use fp::prime::{self, Binomial, Prime};
    use pyo3::basic::CompareOp;
    use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;

    const MAX_VALID_PRIME: u32 = 1 << 31;

    type DynFp = RustFp<prime::ValidPrime>;
    type DynSmallFq = RustSmallFq<prime::ValidPrime>;
    type DynFpElement = FieldElement<DynFp>;
    type DynSmallFqElement = FieldElement<DynSmallFq>;

    #[pyclass(name = "Fp", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    pub struct PyFp(DynFp);

    #[pyclass(name = "SmallFq", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    pub struct PySmallFq(DynSmallFq);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum FieldElementKind {
        Fp(DynFpElement),
        SmallFq(DynSmallFqElement),
    }

    #[pyclass(name = "FieldElement", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    pub struct PyFieldElement(FieldElementKind);

    fn valid_prime(p: u32) -> PyResult<prime::ValidPrime> {
        if p < 2 || p >= MAX_VALID_PRIME {
            return Err(PyValueError::new_err(format!("{p} is not prime")));
        }
        prime::ValidPrime::try_from(p)
            .map_err(|_| PyValueError::new_err(format!("{p} is not prime")))
    }

    fn table_prime(p: u32) -> PyResult<prime::ValidPrime> {
        if fp::PRIMES.contains(&p) {
            valid_prime(p)
        } else {
            Err(PyValueError::new_err(format!(
                "{p} is not a supported table prime"
            )))
        }
    }

    fn small_fq(p: u32, degree: u32) -> PyResult<DynSmallFq> {
        let p = valid_prime(p)?;
        if degree <= 1 {
            return Err(PyValueError::new_err("degree must be greater than 1"));
        }
        if degree > 16 || p.as_u32().checked_pow(degree).is_none_or(|q| q >= 1 << 16) {
            return Err(PyValueError::new_err("field is too large"));
        }
        Ok(DynSmallFq::new(p, degree))
    }

    fn py_hash<T: Hash>(value: &T) -> isize {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        match hasher.finish() as isize {
            -1 => -2,
            hash => hash,
        }
    }

    impl FieldElementKind {
        fn field_repr(self) -> String {
            match self {
                Self::Fp(x) => format!("Fp({})", x.field().characteristic().as_u32()),
                Self::SmallFq(x) => {
                    let f = x.field();
                    format!("SmallFq({}, {})", f.characteristic().as_u32(), f.degree())
                }
            }
        }

        fn mismatched_field_error(lhs: Self, rhs: Self) -> PyErr {
            PyValueError::new_err(format!(
                "cannot combine elements from {} and {}",
                lhs.field_repr(),
                rhs.field_repr()
            ))
        }
    }

    #[pymethods]
    impl PyFp {
        #[new]
        pub fn new(p: u32) -> PyResult<Self> {
            Ok(Self(DynFp::new(valid_prime(p)?)))
        }

        pub fn characteristic(&self) -> u32 {
            self.0.characteristic().as_u32()
        }

        pub fn degree(&self) -> u32 {
            self.0.degree()
        }

        pub fn zero(&self) -> PyFieldElement {
            PyFieldElement(FieldElementKind::Fp(self.0.zero()))
        }

        pub fn one(&self) -> PyFieldElement {
            PyFieldElement(FieldElementKind::Fp(self.0.one()))
        }

        pub fn element(&self, value: u32) -> PyFieldElement {
            PyFieldElement(FieldElementKind::Fp(self.0.element(value)))
        }

        pub fn __repr__(&self) -> String {
            format!("Fp({})", self.characteristic())
        }

        pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
            let eq = other
                .extract::<PyRef<Self>>()
                .is_ok_and(|other| self.0 == other.0);
            match op {
                CompareOp::Eq => eq,
                CompareOp::Ne => !eq,
                _ => false,
            }
        }

        pub fn __hash__(&self) -> isize {
            py_hash(&self.0)
        }
    }

    #[pymethods]
    impl PySmallFq {
        #[new]
        pub fn new(p: u32, degree: u32) -> PyResult<Self> {
            Ok(Self(small_fq(p, degree)?))
        }

        pub fn p(&self) -> u32 {
            self.0.characteristic().as_u32()
        }

        pub fn degree(&self) -> u32 {
            self.0.degree()
        }

        pub fn a(&self) -> PyFieldElement {
            PyFieldElement(FieldElementKind::SmallFq(self.0.a()))
        }

        pub fn q(&self) -> u32 {
            self.0.q()
        }

        pub fn zero(&self) -> PyFieldElement {
            PyFieldElement(FieldElementKind::SmallFq(self.0.zero()))
        }

        pub fn one(&self) -> PyFieldElement {
            PyFieldElement(FieldElementKind::SmallFq(self.0.one()))
        }

        pub fn __repr__(&self) -> String {
            format!("SmallFq({}, {})", self.p(), self.degree())
        }

        pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
            let eq = other
                .extract::<PyRef<Self>>()
                .is_ok_and(|other| self.0 == other.0);
            match op {
                CompareOp::Eq => eq,
                CompareOp::Ne => !eq,
                _ => false,
            }
        }

        pub fn __hash__(&self) -> isize {
            py_hash(&self.0)
        }
    }

    #[pymethods]
    impl PyFieldElement {
        pub fn inv(&self) -> Option<Self> {
            match self.0 {
                FieldElementKind::Fp(x) => x.inv().map(|x| Self(FieldElementKind::Fp(x))),
                FieldElementKind::SmallFq(x) => x.inv().map(|x| Self(FieldElementKind::SmallFq(x))),
            }
        }

        pub fn frobenius(&self) -> Self {
            match self.0 {
                FieldElementKind::Fp(x) => Self(FieldElementKind::Fp(x.frobenius())),
                FieldElementKind::SmallFq(x) => Self(FieldElementKind::SmallFq(x.frobenius())),
            }
        }

        pub fn field<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            match self.0 {
                FieldElementKind::Fp(x) => {
                    Py::new(py, PyFp(x.field())).map(|x| x.into_bound(py).into_any())
                }
                FieldElementKind::SmallFq(x) => {
                    Py::new(py, PySmallFq(x.field())).map(|x| x.into_bound(py).into_any())
                }
            }
        }

        pub fn __add__(&self, rhs: Self) -> PyResult<Self> {
            match (self.0, rhs.0) {
                (FieldElementKind::Fp(a), FieldElementKind::Fp(b)) if a.field() == b.field() => {
                    Ok(Self(FieldElementKind::Fp(a + b)))
                }
                (FieldElementKind::SmallFq(a), FieldElementKind::SmallFq(b))
                    if a.field() == b.field() =>
                {
                    Ok(Self(FieldElementKind::SmallFq(a + b)))
                }
                (a, b) => Err(FieldElementKind::mismatched_field_error(a, b)),
            }
        }

        pub fn __sub__(&self, rhs: Self) -> PyResult<Self> {
            match (self.0, rhs.0) {
                (FieldElementKind::Fp(a), FieldElementKind::Fp(b)) if a.field() == b.field() => {
                    Ok(Self(FieldElementKind::Fp(a - b)))
                }
                (FieldElementKind::SmallFq(a), FieldElementKind::SmallFq(b))
                    if a.field() == b.field() =>
                {
                    Ok(Self(FieldElementKind::SmallFq(a - b)))
                }
                (a, b) => Err(FieldElementKind::mismatched_field_error(a, b)),
            }
        }

        pub fn __mul__(&self, rhs: Self) -> PyResult<Self> {
            match (self.0, rhs.0) {
                (FieldElementKind::Fp(a), FieldElementKind::Fp(b)) if a.field() == b.field() => {
                    Ok(Self(FieldElementKind::Fp(a * b)))
                }
                (FieldElementKind::SmallFq(a), FieldElementKind::SmallFq(b))
                    if a.field() == b.field() =>
                {
                    Ok(Self(FieldElementKind::SmallFq(a * b)))
                }
                (a, b) => Err(FieldElementKind::mismatched_field_error(a, b)),
            }
        }

        pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
            match (self.0, rhs.0) {
                (FieldElementKind::Fp(a), FieldElementKind::Fp(b)) if a.field() == b.field() => (a
                    / b)
                    .map(|x| Self(FieldElementKind::Fp(x)))
                    .ok_or_else(|| PyZeroDivisionError::new_err("division by zero")),
                (FieldElementKind::SmallFq(a), FieldElementKind::SmallFq(b))
                    if a.field() == b.field() =>
                {
                    (a / b)
                        .map(|x| Self(FieldElementKind::SmallFq(x)))
                        .ok_or_else(|| PyZeroDivisionError::new_err("division by zero"))
                }
                (a, b) => Err(FieldElementKind::mismatched_field_error(a, b)),
            }
        }

        pub fn __neg__(&self) -> Self {
            match self.0 {
                FieldElementKind::Fp(x) => Self(FieldElementKind::Fp(-x)),
                FieldElementKind::SmallFq(x) => Self(FieldElementKind::SmallFq(-x)),
            }
        }

        pub fn __int__(&self) -> PyResult<u32> {
            match self.0 {
                FieldElementKind::Fp(x) => Ok(*x),
                FieldElementKind::SmallFq(_) => Err(PyValueError::new_err(
                    "SmallFq elements do not have a canonical integer value",
                )),
            }
        }

        pub fn __repr__(&self) -> String {
            match self.0 {
                FieldElementKind::Fp(x) => {
                    format!("FieldElement(Fp({}), {x})", x.field().characteristic())
                }
                FieldElementKind::SmallFq(x) => {
                    let f = x.field();
                    format!(
                        "FieldElement(SmallFq({}, {}), {x})",
                        f.characteristic(),
                        f.degree()
                    )
                }
            }
        }

        pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
            let eq = other
                .extract::<PyRef<Self>>()
                .is_ok_and(|other| self.0 == other.0);
            match op {
                CompareOp::Eq => eq,
                CompareOp::Ne => !eq,
                _ => false,
            }
        }

        pub fn __hash__(&self) -> isize {
            py_hash(&self.0)
        }
    }

    #[pyfunction]
    pub fn power_mod(p: u32, b: u32, e: u32) -> PyResult<u32> {
        Ok(valid_prime(p)?.pow_mod(b, e))
    }

    #[pyfunction]
    pub fn log2(n: usize) -> usize {
        prime::log2(n)
    }

    #[pyfunction]
    pub fn logp(p: u32, n: u32) -> PyResult<u32> {
        Ok(prime::logp(valid_prime(p)?, n))
    }

    #[pyfunction]
    pub fn factor_pk(p: u32, n: u32) -> PyResult<(u32, u32)> {
        Ok(prime::factor_pk(valid_prime(p)?, n))
    }

    #[pyfunction]
    pub fn inverse(p: u32, k: u32) -> PyResult<u32> {
        Ok(prime::inverse(valid_prime(p)?, k))
    }

    #[pyfunction]
    pub fn minus_one_to_the_n(p: u32, i: i32) -> PyResult<u32> {
        Ok(prime::minus_one_to_the_n(valid_prime(p)?, i))
    }

    #[pyfunction]
    pub fn is_prime(p: u32) -> bool {
        valid_prime(p).is_ok()
    }

    #[pyfunction]
    pub fn binomial(p: u32, n: u32, k: u32) -> PyResult<u32> {
        Ok(u32::binomial(table_prime(p)?, n, k))
    }

    #[pyfunction]
    pub fn multinomial(p: u32, mut l: Vec<u32>) -> PyResult<u32> {
        Ok(u32::multinomial(table_prime(p)?, &mut l))
    }

    #[pyfunction]
    pub fn binomial_odd_is_zero(p: u32, n: u32, k: u32) -> PyResult<bool> {
        Ok(u32::binomial_odd_is_zero(table_prime(p)?, n, k))
    }

    #[pyfunction]
    pub fn binomial2(n: u32, k: u32) -> u32 {
        u32::binomial2(n, k)
    }

    #[pyfunction]
    pub fn multinomial2(l: Vec<u32>) -> u32 {
        u32::multinomial2(&l)
    }

    #[pyfunction]
    pub fn binomial4(n: u32, k: u32) -> u32 {
        u32::binomial4(n, k)
    }

    #[pyfunction]
    pub fn binomial4_rec(n: u32, k: u32) -> u32 {
        u32::binomial4_rec(n, k)
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("F2", PyFp(DynFp::new(prime::TWO)))?;
        m.add("F3", PyFp(DynFp::new(prime::P3.to_dyn())))?;
        m.add("F5", PyFp(DynFp::new(prime::P5.to_dyn())))?;
        m.add("F7", PyFp(DynFp::new(prime::P7.to_dyn())))?;
        m.add("TWO", prime::TWO.as_u32())?;
        m.add("PRIMES", fp::PRIMES.to_vec())?;
        m.add("NUM_PRIMES", fp::NUM_PRIMES)?;
        m.add("PRIME_TO_INDEX_MAP", fp::PRIME_TO_INDEX_MAP.to_vec())?;
        m.add("MAX_MULTINOMIAL_LEN", fp::MAX_MULTINOMIAL_LEN)?;
        m.add("ODD_PRIMES", fp::ODD_PRIMES)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn unwrap_py_err<T>(result: PyResult<T>) -> PyErr {
            match result {
                Ok(_) => panic!("expected Python error"),
                Err(err) => err,
            }
        }

        fn assert_zero_division(err: PyErr) {
            Python::initialize();
            Python::attach(|py| assert!(err.is_instance_of::<PyZeroDivisionError>(py)));
        }

        #[test]
        fn valid_prime_conversion_stays_private() {
            let p = valid_prime(5).unwrap();
            assert_eq!(p.as_i32(), 5);
            assert_eq!(p.as_u32(), 5);
            assert_eq!(p.as_usize(), 5);
            assert_eq!(p.sum(3, 4), 2);
            assert_eq!(p.product(3, 4), 2);
            assert_eq!(p.inverse(2), 3);
            assert_eq!(p.pow(3), 125);
            assert_eq!(p.pow_mod(3, 4), 1);
            assert!(valid_prime(0).is_err());
            assert!(valid_prime(1).is_err());
            assert!(valid_prime(9).is_err());
            assert!(valid_prime(1 << 31).is_err());
        }

        #[test]
        fn module_helpers() {
            assert_eq!(power_mod(5, 3, 4).unwrap(), 1);
            assert!(power_mod(0, 3, 4).is_err());
            assert!(power_mod(1, 3, 4).is_err());
            assert!(power_mod(4, 3, 4).is_err());
            assert_eq!(log2(0b1011), 3);
            assert_eq!(logp(3, 27).unwrap(), 4);
            assert!(logp(1, 27).is_err());
            assert_eq!(factor_pk(3, 45).unwrap(), (2, 5));
            assert!(factor_pk(0, 45).is_err());
            assert_eq!(inverse(3, 2).unwrap(), 2);
            assert!(inverse(1, 2).is_err());
            assert_eq!(minus_one_to_the_n(3, 3).unwrap(), 2);
            assert!(is_prime(7));
            assert!(!is_prime(0));
            assert!(!is_prime(1));
            assert!(!is_prime(9));
        }

        #[test]
        fn binomial_helpers() {
            assert_eq!(binomial(3, 1090, 730).unwrap(), 1);
            assert_eq!(multinomial(5, vec![1, 2, 3]).unwrap(), 0);
            assert!(binomial_odd_is_zero(3, 3, 1).unwrap());
            assert!(binomial(4, 5, 2).is_err());
            assert!(binomial(257, 5, 2).is_err());
            assert!(multinomial(257, vec![1, 2]).is_err());
            assert!(binomial_odd_is_zero(257, 5, 2).is_err());
            assert_eq!(binomial2(3, 1), 1);
            assert_eq!(multinomial2(vec![1, 2]), 1);
            assert_eq!(binomial4(5, 2), 2);
            assert_eq!(binomial4_rec(5, 2), 2);
        }

        #[test]
        fn fp_field_methods_and_elements() {
            let f = PyFp::new(5).unwrap();
            assert_eq!(f.characteristic(), 5);
            assert_eq!(f.degree(), 1);
            assert_eq!(f.__repr__(), "Fp(5)");
            assert_eq!(f.zero().__int__().unwrap(), 0);
            assert_eq!(f.one().__int__().unwrap(), 1);
            assert_eq!(f.element(7).__int__().unwrap(), 2);
            assert!(PyFp::new(1).is_err());

            let two = f.element(2);
            let four = f.element(4);
            assert_eq!(two.__add__(four).unwrap().__int__().unwrap(), 1);
            assert_eq!(two.__sub__(four).unwrap().__int__().unwrap(), 3);
            assert_eq!(two.__mul__(four).unwrap().__int__().unwrap(), 3);
            assert_eq!(two.__truediv__(four).unwrap().__int__().unwrap(), 3);
            assert_eq!(two.__neg__().__int__().unwrap(), 3);
            assert_eq!(two.inv().unwrap().__int__().unwrap(), 3);
            assert_eq!(two.frobenius().__int__().unwrap(), 2);
            assert!(f.zero().inv().is_none());
            let err = unwrap_py_err(two.__truediv__(f.zero()));
            assert_zero_division(err);
            let err = unwrap_py_err(two.__add__(PyFp::new(7).unwrap().one()));
            assert!(err.to_string().contains("Fp(5) and Fp(7)"));
        }

        #[test]
        fn small_fq_field_methods_and_elements() {
            let f = PySmallFq::new(2, 3).unwrap();
            assert_eq!(f.p(), 2);
            assert_eq!(f.degree(), 3);
            assert_eq!(f.q(), 8);
            assert_eq!(f.__repr__(), "SmallFq(2, 3)");
            assert!(PySmallFq::new(2, 1).is_err());
            assert!(PySmallFq::new(2, 16).is_err());
            assert!(PySmallFq::new(4, 2).is_err());

            let zero = f.zero();
            let one = f.one();
            let a = f.a();
            assert!(zero.inv().is_none());
            assert!(zero.__int__().is_err());
            assert_eq!(one.__repr__(), "FieldElement(SmallFq(2, 3), 1)");
            assert_eq!(
                a.__mul__(a).unwrap().__repr__(),
                "FieldElement(SmallFq(2, 3), a^2)"
            );
            assert_eq!(a.__truediv__(a).unwrap().__repr__(), one.__repr__());
            assert_eq!(a.frobenius().__repr__(), "FieldElement(SmallFq(2, 3), a^2)");
            let err = unwrap_py_err(a.__truediv__(zero));
            assert_zero_division(err);
            let err = unwrap_py_err(a.__add__(PySmallFq::new(2, 2).unwrap().one()));
            assert!(err.to_string().contains("SmallFq(2, 3) and SmallFq(2, 2)"));
            let err = unwrap_py_err(a.__add__(PyFp::new(2).unwrap().one()));
            assert!(err.to_string().contains("SmallFq(2, 3) and Fp(2)"));
        }

        #[test]
        fn field_value_equality_and_hashing() {
            Python::initialize();
            Python::attach(|py| {
                let fp5 = PyFp::new(5).unwrap();
                let fp5_again = Py::new(py, PyFp::new(5).unwrap()).unwrap();
                let fp7 = Py::new(py, PyFp::new(7).unwrap()).unwrap();
                let small = Py::new(py, PySmallFq::new(2, 3).unwrap()).unwrap();

                assert!(fp5.__richcmp__(fp5_again.bind(py).as_any(), CompareOp::Eq));
                assert!(!fp5.__richcmp__(fp7.bind(py).as_any(), CompareOp::Eq));
                assert!(fp5.__richcmp__(small.bind(py).as_any(), CompareOp::Ne));
                assert_eq!(fp5.__hash__(), PyFp::new(5).unwrap().__hash__());

                let small23 = PySmallFq::new(2, 3).unwrap();
                let small23_again = Py::new(py, PySmallFq::new(2, 3).unwrap()).unwrap();
                let small22 = Py::new(py, PySmallFq::new(2, 2).unwrap()).unwrap();

                assert!(small23.__richcmp__(small23_again.bind(py).as_any(), CompareOp::Eq));
                assert!(!small23.__richcmp__(small22.bind(py).as_any(), CompareOp::Eq));
                assert!(small23.__richcmp__(fp5_again.bind(py).as_any(), CompareOp::Ne));
                assert_eq!(small23.__hash__(), PySmallFq::new(2, 3).unwrap().__hash__());

                let f = PyFp::new(5).unwrap();
                let two = f.element(2);
                let seven = Py::new(py, f.element(7)).unwrap();
                let two_in_fp7 = Py::new(py, PyFp::new(7).unwrap().element(2)).unwrap();
                let small_one = Py::new(py, PySmallFq::new(2, 3).unwrap().one()).unwrap();

                assert!(two.__richcmp__(seven.bind(py).as_any(), CompareOp::Eq));
                assert!(!two.__richcmp__(two_in_fp7.bind(py).as_any(), CompareOp::Eq));
                assert!(two.__richcmp__(small_one.bind(py).as_any(), CompareOp::Ne));
                assert_eq!(two.__hash__(), f.element(7).__hash__());
            });
        }
    }
}
