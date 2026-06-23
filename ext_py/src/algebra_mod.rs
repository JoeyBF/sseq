use pyo3::prelude::*;

#[pymodule]
pub mod algebra_py {
    use ::algebra::{Algebra, Bialgebra, GeneratedAlgebra};
    use ::fp::prime::{self, Prime};
    use pyo3::basic::CompareOp;
    use pyo3::exceptions::{PyIndexError, PyValueError};

    use super::*;

    /// Upper bound on accepted primes, mirroring `fp_py::valid_prime`.
    const MAX_VALID_PRIME: u32 = 1 << 31;

    /// Convert a plain `int` prime from Python into a `ValidPrime`, raising
    /// `ValueError` (never panicking) for a non-prime. `ValidPrime` itself is
    /// never exposed to Python. Mirrors the `valid_prime` helper in `fp_mod`.
    fn valid_prime(p: u32) -> PyResult<prime::ValidPrime> {
        if p < 2 || p >= MAX_VALID_PRIME {
            return Err(PyValueError::new_err(format!("{p} is not prime")));
        }
        prime::ValidPrime::try_from(p)
            .map_err(|_| PyValueError::new_err(format!("{p} is not prime")))
    }

    fn checked_same_prime(lhs: u32, rhs: u32) -> PyResult<()> {
        if lhs == rhs {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "prime mismatch: {lhs} != {rhs}"
            )))
        }
    }

    fn checked_equal_len(lhs: usize, rhs: usize) -> PyResult<()> {
        if lhs == rhs {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "length mismatch: {lhs} != {rhs}"
            )))
        }
    }

    /// Ensure a result slice is long enough to receive a product landing in a
    /// space of dimension `dim`, raising `ValueError` rather than letting an
    /// upstream `add_basis_element` index panic.
    fn checked_result_len(len: usize, dim: usize) -> PyResult<()> {
        if len >= dim {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "result has length {len} but the target degree has dimension {dim}"
            )))
        }
    }

    fn non_negative_degree(degree: i32) -> PyResult<()> {
        if degree >= 0 {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!(
                "degree {degree} is negative"
            )))
        }
    }

    /// Convert a Python value (`dict`/`list`/`int`/`float`/`str`/`bool`/`None`)
    /// into a `serde_json::Value`. This is the minimal hand-rolled half of the
    /// `serde_json::Value` <-> Python bridge described in API_PROPOSAL §2.6
    /// (we have no `pythonize` dependency); only the directions exercised by
    /// `SteenrodAlgebra.from_json` are implemented. Booleans are checked before
    /// integers because Python `bool` is a subclass of `int`. Raises
    /// `ValueError` for unsupported types or non-finite floats rather than
    /// panicking.
    fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
        use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
        if value.is_none() {
            return Ok(serde_json::Value::Null);
        }
        if let Ok(b) = value.cast::<PyBool>() {
            return Ok(serde_json::Value::Bool(b.is_true()));
        }
        if let Ok(i) = value.cast::<PyInt>() {
            let n: i64 = i.extract()?;
            return Ok(serde_json::Value::from(n));
        }
        if let Ok(f) = value.cast::<PyFloat>() {
            let f: f64 = f.extract()?;
            return serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .ok_or_else(|| PyValueError::new_err("cannot represent non-finite float as JSON"));
        }
        if let Ok(s) = value.cast::<PyString>() {
            return Ok(serde_json::Value::String(s.extract()?));
        }
        if let Ok(dict) = value.cast::<PyDict>() {
            let mut map = serde_json::Map::with_capacity(dict.len());
            for (k, v) in dict.iter() {
                let key: String = k
                    .cast::<PyString>()
                    .map_err(|_| PyValueError::new_err("JSON object keys must be strings"))?
                    .extract()?;
                map.insert(key, py_to_json(&v)?);
            }
            return Ok(serde_json::Value::Object(map));
        }
        if let Ok(list) = value.cast::<PyList>() {
            let mut arr = Vec::with_capacity(list.len());
            for item in list.iter() {
                arr.push(py_to_json(&item)?);
            }
            return Ok(serde_json::Value::Array(arr));
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            let mut arr = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                arr.push(py_to_json(&item)?);
            }
            return Ok(serde_json::Value::Array(arr));
        }
        Err(PyValueError::new_err(format!(
            "cannot convert {} to JSON",
            value.get_type().name()?
        )))
    }

    #[pyclass] // This will be part of the module
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum AlgebraType {
        Adem,
        Milnor,
    }

    impl From<AlgebraType> for ::algebra::AlgebraType {
        fn from(value: AlgebraType) -> Self {
            match value {
                AlgebraType::Adem => ::algebra::AlgebraType::Adem,
                AlgebraType::Milnor => ::algebra::AlgebraType::Milnor,
            }
        }
    }

    /// A basis element of the Milnor algebra: a product of exterior generators
    /// `Q_k` (encoded as the bitmask `q_part`) and a polynomial part `P(p_part)`.
    #[pyclass(name = "MilnorBasisElement", skip_from_py_object)]
    #[derive(Clone)]
    pub struct MilnorBasisElement(::algebra::milnor_algebra::MilnorBasisElement);

    #[pymethods]
    impl MilnorBasisElement {
        #[new]
        #[pyo3(signature = (p_part, q_part = 0, degree = 0))]
        pub fn new(p_part: Vec<u32>, q_part: u32, degree: i32) -> Self {
            MilnorBasisElement(::algebra::milnor_algebra::MilnorBasisElement {
                q_part,
                p_part,
                degree,
            })
        }

        #[getter]
        pub fn q_part(&self) -> u32 {
            self.0.q_part
        }

        #[setter]
        pub fn set_q_part(&mut self, value: u32) {
            self.0.q_part = value;
        }

        #[getter]
        pub fn p_part(&self) -> Vec<u32> {
            self.0.p_part.clone()
        }

        #[setter]
        pub fn set_p_part(&mut self, value: Vec<u32>) {
            self.0.p_part = value;
        }

        #[getter]
        pub fn degree(&self) -> i32 {
            self.0.degree
        }

        #[setter]
        pub fn set_degree(&mut self, value: i32) {
            self.0.degree = value;
        }

        /// Recompute the `degree` field from the `p_part`/`q_part` at prime `p`.
        pub fn compute_degree(&mut self, p: u32) -> PyResult<()> {
            self.0.compute_degree(valid_prime(p)?);
            Ok(())
        }

        pub fn __repr__(&self) -> String {
            format!(
                "MilnorBasisElement(p_part={:?}, q_part={}, degree={})",
                self.0.p_part, self.0.q_part, self.0.degree
            )
        }

        pub fn __str__(&self) -> String {
            format!("{}", self.0)
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
    }

    /// A Milnor profile function, describing a sub-Hopf-algebra of the Steenrod
    /// algebra.
    #[pyclass(name = "MilnorProfile")]
    pub struct MilnorProfile(::algebra::milnor_algebra::MilnorProfile);

    impl MilnorProfile {
        /// `MilnorProfile` upstream is intentionally not `Clone`; reconstruct a
        /// fresh copy from its public fields when we need to hand one to the
        /// algebra constructor or return one to Python.
        fn to_rust(&self) -> ::algebra::milnor_algebra::MilnorProfile {
            ::algebra::milnor_algebra::MilnorProfile {
                truncated: self.0.truncated,
                q_part: self.0.q_part,
                p_part: self.0.p_part.clone(),
            }
        }
    }

    #[pymethods]
    impl MilnorProfile {
        #[new]
        #[pyo3(signature = (truncated = false, q_part = u32::MAX, p_part = Vec::new()))]
        pub fn new(truncated: bool, q_part: u32, p_part: Vec<u32>) -> Self {
            MilnorProfile(::algebra::milnor_algebra::MilnorProfile {
                truncated,
                q_part,
                p_part,
            })
        }

        #[getter]
        pub fn truncated(&self) -> bool {
            self.0.truncated
        }

        #[setter]
        pub fn set_truncated(&mut self, value: bool) {
            self.0.truncated = value;
        }

        #[getter]
        pub fn q_part(&self) -> u32 {
            self.0.q_part
        }

        #[setter]
        pub fn set_q_part(&mut self, value: u32) {
            self.0.q_part = value;
        }

        #[getter(p_part)]
        pub fn profile_p_part(&self) -> Vec<u32> {
            self.0.p_part.clone()
        }

        #[setter(p_part)]
        pub fn set_p_part(&mut self, value: Vec<u32>) {
            self.0.p_part = value;
        }

        pub fn is_trivial(&self) -> bool {
            self.0.is_trivial()
        }

        pub fn get_p_part(&self, i: usize) -> u32 {
            self.0.get_p_part(i)
        }

        pub fn is_valid(&self) -> bool {
            self.0.is_valid()
        }

        pub fn is_an(&self, generic: bool) -> bool {
            self.0.is_an(generic)
        }

        pub fn __repr__(&self) -> String {
            format!(
                "MilnorProfile(truncated={}, q_part={}, p_part={:?})",
                self.0.truncated, self.0.q_part, self.0.p_part
            )
        }
    }

    #[pyclass]
    pub struct MilnorAlgebra(::algebra::MilnorAlgebra);

    impl MilnorAlgebra {
        /// Lazily compute book-keeping up to `degree`. The Milnor algebra is
        /// infinite-dimensional and its internal `OnceVec` tables panic when
        /// indexed past the computed range, so every degree-indexed Python
        /// method funnels through here first. `compute_basis` is idempotent and
        /// cheap to re-call, so this is a safe (if slightly eager) way to avoid
        /// cross-boundary panics; it is a no-op for negative degrees.
        fn ensure_basis(&self, degree: i32) {
            if degree >= 0 {
                self.0.compute_basis(degree);
            }
        }

        /// Validate two factor degrees and compute the (basis-populated) target
        /// degree of their product.
        fn product_target(&self, r_degree: i32, s_degree: i32) -> PyResult<i32> {
            non_negative_degree(r_degree)?;
            non_negative_degree(s_degree)?;
            let target = r_degree
                .checked_add(s_degree)
                .ok_or_else(|| PyValueError::new_err("product degree overflows i32"))?;
            self.ensure_basis(target);
            Ok(target)
        }

        fn checked_basis_index(&self, degree: i32, idx: usize) -> PyResult<()> {
            let dim = self.0.dimension(degree);
            if idx < dim {
                Ok(())
            } else {
                Err(PyIndexError::new_err(format!(
                    "index {idx} out of range for degree {degree} (dimension {dim})"
                )))
            }
        }
    }

    #[pymethods]
    impl MilnorAlgebra {
        #[new]
        #[pyo3(signature = (p, unstable_enabled = false))]
        pub fn new(p: u32, unstable_enabled: bool) -> PyResult<Self> {
            Ok(MilnorAlgebra(::algebra::MilnorAlgebra::new(
                valid_prime(p)?,
                unstable_enabled,
            )))
        }

        /// Construct a Milnor algebra restricted to the given profile. Raises
        /// `ValueError` for an invalid profile rather than panicking (upstream
        /// `new_with_profile` asserts validity).
        #[staticmethod]
        #[pyo3(signature = (p, profile, unstable_enabled = false))]
        pub fn new_with_profile(
            p: u32,
            profile: PyRef<'_, MilnorProfile>,
            unstable_enabled: bool,
        ) -> PyResult<Self> {
            let p = valid_prime(p)?;
            let profile = profile.to_rust();
            if !profile.is_valid() {
                return Err(PyValueError::new_err("invalid Milnor profile"));
            }
            Ok(MilnorAlgebra(::algebra::MilnorAlgebra::new_with_profile(
                p,
                profile,
                unstable_enabled,
            )))
        }

        // --- Algebra trait surface --------------------------------------------

        /// The prime as a plain `int` (`ValidPrime` is never exposed).
        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn compute_basis(&self, degree: i32) {
            self.ensure_basis(degree);
        }

        pub fn dimension(&self, degree: i32) -> usize {
            if degree < 0 {
                return 0;
            }
            self.ensure_basis(degree);
            self.0.dimension(degree)
        }

        pub fn basis_element_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.basis_element_to_string(degree, idx))
        }

        /// Parse a basis element, returning `(degree, index)`. Raises
        /// `ValueError` if the string does not parse, or if it names an element
        /// that is not present in this (possibly profiled) algebra.
        ///
        /// Upstream's `basis_element_from_string` is *not* total: once a name
        /// parses syntactically, the nom closures call the panicking
        /// `beps_pn(..).unwrap()` (milnor_algebra.rs ~984) and
        /// `basis_element_to_index` (which `panic!`s on a missing element,
        /// ~338). So inputs like `"Sq0"`, `"P0"`, `"Q_5"`, or out-of-profile
        /// names abort across the FFI boundary. The `steenrod_parser`
        /// primitives those closures use are `pub(crate)`, so we cannot re-run
        /// the parse with the `Option`-returning (`try_*`) primitives from the
        /// binding, nor intercept the panicking calls buried inside the parser.
        /// We therefore contain the panic with `catch_unwind` and translate it
        /// into a `ValueError`. This is sound: the panic originates after
        /// `compute_basis` has already returned (it is a failed lookup/unwrap,
        /// not a half-finished mutation), so no inconsistent shared state
        /// survives the unwind, and the panic is caught before it can reach
        /// Python.
        pub fn basis_element_from_string(&self, elt: &str) -> PyResult<(i32, usize)> {
            use std::panic::{catch_unwind, AssertUnwindSafe};
            match catch_unwind(AssertUnwindSafe(|| self.0.basis_element_from_string(elt))) {
                Ok(Some(res)) => Ok(res),
                Ok(None) => Err(PyValueError::new_err(format!(
                    "could not parse basis element: {elt}"
                ))),
                Err(_) => Err(PyValueError::new_err(format!(
                    "{elt} does not name a basis element of this algebra"
                ))),
            }
        }

        pub fn element_to_string(
            &self,
            py: Python<'_>,
            degree: i32,
            element: &Bound<'_, PyAny>,
        ) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            let element = crate::fp_py::extract_input_owned(py, element)?;
            checked_same_prime(element.prime().as_u32(), self.0.prime().as_u32())?;
            checked_equal_len(element.len(), self.0.dimension(degree))?;
            Ok(self.0.element_to_string(degree, element.as_slice()))
        }

        pub fn multiply_basis_elements(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // Reduce the coefficient mod p before handing it to upstream, which
            // computes `coeff * v` (milnor_algebra.rs ~555) before reducing and
            // would overflow (panicking in debug, wrapping in release) for large
            // `coeff`. The algebra is over F_p, so this is mathematically
            // equivalent.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            self.checked_basis_index(s_degree, s_idx)?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0
                    .multiply_basis_elements(res.copy(), coeff, r_degree, r_idx, s_degree, s_idx);
                Ok(())
            })
        }

        pub fn multiply_basis_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // See `multiply_basis_elements`: reduce mod p to avoid the upstream
            // `coeff * v` overflow.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_basis_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r_idx,
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_basis_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // See `multiply_basis_elements`: reduce mod p to avoid the upstream
            // `coeff * v` overflow.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(s_degree, s_idx)?;
            let r = crate::fp_py::extract_input_owned(py, r)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_basis_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s_idx,
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // See `multiply_basis_elements`: reduce mod p to avoid the upstream
            // `coeff * v` overflow.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            let r = crate::fp_py::extract_input_owned(py, r)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
            self.0.default_filtration_one_products()
        }

        // --- GeneratedAlgebra trait surface -----------------------------------

        pub fn generators(&self, degree: i32) -> PyResult<Vec<usize>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generators(degree))
        }

        pub fn generator_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.generator_to_string(degree, idx))
        }

        pub fn decompose_basis_element(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(u32, (i32, usize), (i32, usize))>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            // The degree-0 unit has an empty `p_part`; upstream's
            // `decompose_basis_element_ppart` computes `p_part[0..len - 1]`
            // with `len == 0`, underflowing and panicking
            // (milnor_algebra.rs ~1607). The unit is the identity and is
            // indecomposable -- the trait docs note it is invalid to decompose
            // a generator, and there is no product of strictly-smaller basis
            // elements that equals the unit -- so we surface a `ValueError`
            // instead of aborting. (Empty `p_part` with `q_part == 0` can only
            // be the degree-0 unit, since any such element has degree 0.)
            let basis = self.0.basis_element_from_index(degree, idx);
            if basis.q_part == 0 && basis.p_part.is_empty() {
                return Err(PyValueError::new_err("the degree-0 unit is indecomposable"));
            }
            Ok(self.0.decompose_basis_element(degree, idx))
        }

        pub fn generating_relations(
            &self,
            degree: i32,
        ) -> PyResult<Vec<Vec<(u32, (i32, usize), (i32, usize))>>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generating_relations(degree))
        }

        // --- Bialgebra trait surface ------------------------------------------

        /// Compute a coproduct. Only supported at `p = 2` upstream; raises
        /// `ValueError` at odd primes rather than panicking on the assertion.
        pub fn coproduct(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(i32, usize, i32, usize)>> {
            if self.0.prime().as_u32() != 2 {
                return Err(PyValueError::new_err(
                    "coproduct is only supported at p = 2",
                ));
            }
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.coproduct(degree, idx))
        }

        pub fn decompose(&self, degree: i32, idx: usize) -> PyResult<Vec<(i32, usize)>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.decompose(degree, idx))
        }

        // --- Milnor-specific methods ------------------------------------------

        pub fn generic(&self) -> bool {
            self.0.generic()
        }

        pub fn q(&self) -> i32 {
            self.0.q()
        }

        pub fn profile(&self) -> MilnorProfile {
            let profile = self.0.profile();
            MilnorProfile(::algebra::milnor_algebra::MilnorProfile {
                truncated: profile.truncated,
                q_part: profile.q_part,
                p_part: profile.p_part.clone(),
            })
        }

        pub fn basis_element_from_index(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<MilnorBasisElement> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(MilnorBasisElement(
                self.0.basis_element_from_index(degree, idx).clone(),
            ))
        }

        pub fn try_basis_element_to_index(
            &self,
            elt: PyRef<'_, MilnorBasisElement>,
        ) -> Option<usize> {
            if elt.0.degree < 0 {
                return None;
            }
            self.ensure_basis(elt.0.degree);
            self.0.try_basis_element_to_index(&elt.0)
        }

        /// Like `try_basis_element_to_index`, but raises `ValueError` if the
        /// element is not in the algebra (upstream panics).
        pub fn basis_element_to_index(
            &self,
            elt: PyRef<'_, MilnorBasisElement>,
        ) -> PyResult<usize> {
            non_negative_degree(elt.0.degree)?;
            self.ensure_basis(elt.0.degree);
            self.0
                .try_basis_element_to_index(&elt.0)
                .ok_or_else(|| PyValueError::new_err(format!("element not in algebra: {}", elt.0)))
        }

        /// The list of `P(R)` partitions in degree `t`.
        pub fn ppart_table(&self, t: i32) -> PyResult<Vec<Vec<u32>>> {
            non_negative_degree(t)?;
            // The internal table is indexed by `degree / q`, so compute enough
            // book-keeping that index `t` is in range at every prime.
            let needed = t
                .checked_mul(self.0.q())
                .ok_or_else(|| PyValueError::new_err("degree overflows i32"))?;
            self.ensure_basis(needed);
            Ok(self.0.ppart_table(t).to_vec())
        }

        /// The degree and index of `Q_1^e P(x)`. Raises `ValueError` if that
        /// element is not in the (profiled) algebra (upstream panics).
        pub fn beps_pn(&self, e: u32, x: u32) -> PyResult<(i32, usize)> {
            let q = self.0.q() as u32;
            let degree = q
                .checked_mul(x)
                .and_then(|v| v.checked_add(e))
                .ok_or_else(|| PyValueError::new_err("degree overflows"))?
                as i32;
            self.ensure_basis(degree);
            let elt = ::algebra::milnor_algebra::MilnorBasisElement {
                degree,
                q_part: e,
                p_part: vec![x],
            };
            self.0
                .try_basis_element_to_index(&elt)
                .map(|idx| (degree, idx))
                .ok_or_else(|| {
                    PyValueError::new_err(format!("Q_1^{e} P({x}) is not in the algebra"))
                })
        }

        /// Multiply two `MilnorBasisElement`s, accumulating into `result`.
        pub fn multiply(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            m1: PyRef<'_, MilnorBasisElement>,
            m2: PyRef<'_, MilnorBasisElement>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // See `multiply_basis_elements`: reduce mod p to avoid the upstream
            // `coeff * v` overflow.
            let coeff = coeff % p;
            let target = self.product_target(m1.0.degree, m2.0.degree)?;
            let dim = self.0.dimension(target);
            // Reject elements that are not genuine basis elements of this
            // algebra up front, since the inner multiply panics if an
            // intermediate term cannot be indexed.
            self.ensure_basis(m1.0.degree);
            self.ensure_basis(m2.0.degree);
            if self.0.try_basis_element_to_index(&m1.0).is_none() {
                return Err(PyValueError::new_err(format!(
                    "left factor is not a basis element of this algebra: {}",
                    m1.0
                )));
            }
            if self.0.try_basis_element_to_index(&m2.0).is_none() {
                return Err(PyValueError::new_err(format!(
                    "right factor is not a basis element of this algebra: {}",
                    m2.0
                )));
            }
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply(res.copy(), coeff, &m1.0, &m2.0);
                Ok(())
            })
        }

        pub fn __repr__(&self) -> String {
            format!("{}", self.0)
        }
    }

    /// A Steenrod power `P^i`, or a Bockstein `b^e`. Mirrors upstream's
    /// `PorBockstein` enum (the pieces of an Adem basis element's
    /// decomposition).
    #[pyclass(name = "PorBockstein")]
    #[derive(Clone, Debug)]
    pub enum PorBockstein {
        P(u32),
        Bockstein(bool),
    }

    /// An Adem basis element of the Steenrod algebra: a sequence of Steenrod
    /// powers `ps` interleaved with Bocksteins encoded in the bitmask
    /// `bocksteins`.
    #[pyclass(name = "AdemBasisElement", skip_from_py_object)]
    #[derive(Clone)]
    pub struct AdemBasisElement(::algebra::adem_algebra::AdemBasisElement);

    #[pymethods]
    impl AdemBasisElement {
        #[new]
        #[pyo3(signature = (ps, bocksteins = 0, degree = 0, p_or_sq = false))]
        pub fn new(ps: Vec<u32>, bocksteins: u32, degree: i32, p_or_sq: bool) -> Self {
            AdemBasisElement(::algebra::adem_algebra::AdemBasisElement {
                degree,
                bocksteins,
                ps,
                p_or_sq,
            })
        }

        #[getter]
        pub fn degree(&self) -> i32 {
            self.0.degree
        }

        #[setter]
        pub fn set_degree(&mut self, value: i32) {
            self.0.degree = value;
        }

        #[getter]
        pub fn bocksteins(&self) -> u32 {
            self.0.bocksteins
        }

        #[setter]
        pub fn set_bocksteins(&mut self, value: u32) {
            self.0.bocksteins = value;
        }

        #[getter]
        pub fn ps(&self) -> Vec<u32> {
            self.0.ps.clone()
        }

        #[setter]
        pub fn set_ps(&mut self, value: Vec<u32>) {
            self.0.ps = value;
        }

        #[getter]
        pub fn p_or_sq(&self) -> bool {
            self.0.p_or_sq
        }

        #[setter]
        pub fn set_p_or_sq(&mut self, value: bool) {
            self.0.p_or_sq = value;
        }

        /// The decomposition into alternating Bocksteins and Steenrod powers,
        /// dropping trivial (`b^0`) Bocksteins. Mirrors the upstream private
        /// `iter_filtered`.
        pub fn iter_filtered(&self) -> Vec<PorBockstein> {
            let bocksteins: Vec<bool> =
                ::fp::prime::iter::BitflagIterator::new(self.0.bocksteins as u64).collect();
            let n = bocksteins.len().max(self.0.ps.len());
            let mut out = Vec::new();
            for i in 0..n {
                if let Some(&b) = bocksteins.get(i) {
                    if b {
                        out.push(PorBockstein::Bockstein(true));
                    }
                }
                if let Some(&p) = self.0.ps.get(i) {
                    out.push(PorBockstein::P(p));
                }
            }
            out
        }

        pub fn __repr__(&self) -> String {
            format!(
                "AdemBasisElement(ps={:?}, bocksteins={}, degree={}, p_or_sq={})",
                self.0.ps, self.0.bocksteins, self.0.degree, self.0.p_or_sq
            )
        }

        pub fn __str__(&self) -> String {
            format!("{}", self.0)
        }

        pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> bool {
            // Upstream equality compares only `ps` and `bocksteins`.
            let eq = other
                .extract::<PyRef<Self>>()
                .is_ok_and(|other| self.0 == other.0);
            match op {
                CompareOp::Eq => eq,
                CompareOp::Ne => !eq,
                _ => false,
            }
        }
    }

    #[pyclass]
    pub struct AdemAlgebra(::algebra::AdemAlgebra);

    impl AdemAlgebra {
        /// Lazily compute book-keeping up to `degree`. Like `MilnorAlgebra`,
        /// the Adem algebra is infinite-dimensional and its internal `OnceVec`
        /// tables panic when indexed past the computed range, so every
        /// degree-indexed Python method funnels through here first. A no-op for
        /// negative degrees.
        fn ensure_basis(&self, degree: i32) {
            if degree >= 0 {
                self.0.compute_basis(degree);
            }
        }

        fn product_target(&self, r_degree: i32, s_degree: i32) -> PyResult<i32> {
            non_negative_degree(r_degree)?;
            non_negative_degree(s_degree)?;
            let target = r_degree
                .checked_add(s_degree)
                .ok_or_else(|| PyValueError::new_err("product degree overflows i32"))?;
            self.ensure_basis(target);
            Ok(target)
        }

        fn checked_basis_index(&self, degree: i32, idx: usize) -> PyResult<()> {
            let dim = self.0.dimension(degree);
            if idx < dim {
                Ok(())
            } else {
                Err(PyIndexError::new_err(format!(
                    "index {idx} out of range for degree {degree} (dimension {dim})"
                )))
            }
        }
    }

    #[pymethods]
    impl AdemAlgebra {
        #[new]
        #[pyo3(signature = (p, unstable_enabled = false))]
        pub fn new(p: u32, unstable_enabled: bool) -> PyResult<Self> {
            // `generic` is not a constructor flag upstream: it is derived as
            // `p != 2`.
            Ok(AdemAlgebra(::algebra::AdemAlgebra::new(
                valid_prime(p)?,
                unstable_enabled,
            )))
        }

        // --- Algebra trait surface --------------------------------------------

        /// The prime as a plain `int` (`ValidPrime` is never exposed).
        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn compute_basis(&self, degree: i32) {
            self.ensure_basis(degree);
        }

        pub fn dimension(&self, degree: i32) -> usize {
            if degree < 0 {
                return 0;
            }
            self.ensure_basis(degree);
            self.0.dimension(degree)
        }

        pub fn basis_element_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.basis_element_to_string(degree, idx))
        }

        /// Parse a basis element, returning `(degree, index)`. Raises
        /// `ValueError` if the string does not parse, or if it names an element
        /// that is not present in this algebra.
        ///
        /// Upstream returns `Option`, but the `None` path is unreachable once
        /// the leading prefix parses: after parsing it calls the panicking
        /// `basis_element_to_index` (adem_algebra.rs ~655/306), so a parseable
        /// but absent/inadmissible name (e.g. `"Sq0"`, `"Sq1 Sq1"`) aborts
        /// across the FFI boundary. The `steenrod_parser` primitives are
        /// `pub(crate)`, so we cannot re-run the parse with `Option`-returning
        /// variants from here. We therefore contain the panic with
        /// `catch_unwind` and translate it into a `ValueError`, exactly as the
        /// `MilnorAlgebra` binding does. This is sound: the panic is a failed
        /// lookup after `compute_basis` has returned, not a half-finished
        /// mutation, so no inconsistent shared state survives the unwind.
        pub fn basis_element_from_string(&self, elt: &str) -> PyResult<(i32, usize)> {
            use std::panic::{catch_unwind, AssertUnwindSafe};
            match catch_unwind(AssertUnwindSafe(|| self.0.basis_element_from_string(elt))) {
                Ok(Some(res)) => Ok(res),
                Ok(None) => Err(PyValueError::new_err(format!(
                    "could not parse basis element: {elt}"
                ))),
                Err(_) => Err(PyValueError::new_err(format!(
                    "{elt} does not name a basis element of this algebra"
                ))),
            }
        }

        pub fn element_to_string(
            &self,
            py: Python<'_>,
            degree: i32,
            element: &Bound<'_, PyAny>,
        ) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            let element = crate::fp_py::extract_input_owned(py, element)?;
            checked_same_prime(element.prime().as_u32(), self.0.prime().as_u32())?;
            checked_equal_len(element.len(), self.0.dimension(degree))?;
            Ok(self.0.element_to_string(degree, element.as_slice()))
        }

        pub fn multiply_basis_elements(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // Reduce the coefficient mod p before handing it to upstream, which
            // computes `coeff * value` (e.g. adem_algebra.rs ~1161) before
            // reducing and would overflow for large `coeff`. The algebra is
            // over F_p, so this is mathematically equivalent.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            self.checked_basis_index(s_degree, s_idx)?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0
                    .multiply_basis_elements(res.copy(), coeff, r_degree, r_idx, s_degree, s_idx);
                Ok(())
            })
        }

        pub fn multiply_basis_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_basis_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r_idx,
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_basis_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(s_degree, s_idx)?;
            let r = crate::fp_py::extract_input_owned(py, r)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_basis_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s_idx,
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            let r = crate::fp_py::extract_input_owned(py, r)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
            self.0.default_filtration_one_products()
        }

        // --- GeneratedAlgebra trait surface -----------------------------------

        pub fn generators(&self, degree: i32) -> PyResult<Vec<usize>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generators(degree))
        }

        pub fn generator_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.generator_to_string(degree, idx))
        }

        pub fn decompose_basis_element(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(u32, (i32, usize), (i32, usize))>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            // Decomposition is only defined for non-generators. The degree-0
            // unit has an empty `ps`, so upstream's `b.ps[0]` indexes out of
            // bounds (adem_algebra.rs ~1195/1270); a generator like `Sq^2`
            // decomposes into a factor of degree 0 whose `AdemBasisElement` is
            // not in the basis, so `basis_element_to_index` panics. Both are
            // indecomposable by definition, so we surface a `ValueError` rather
            // than aborting. (Upstream's own test skips generators before
            // calling `decompose_basis_element`.)
            if degree == 0 || self.0.generators(degree).contains(&idx) {
                return Err(PyValueError::new_err(
                    "the unit and algebra generators are indecomposable",
                ));
            }
            Ok(self.0.decompose_basis_element(degree, idx))
        }

        pub fn generating_relations(
            &self,
            degree: i32,
        ) -> PyResult<Vec<Vec<(u32, (i32, usize), (i32, usize))>>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generating_relations(degree))
        }

        // --- Bialgebra trait surface ------------------------------------------

        /// Compute a coproduct. Raises `ValueError` for inputs that would trip
        /// an upstream assertion: a non-`q`-divisible degree in the generic
        /// case, or a nonzero index in the `p = 2` case (adem_algebra.rs
        /// ~1398/1409).
        pub fn coproduct(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(i32, usize, i32, usize)>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            if self.0.generic() {
                if degree != 1 {
                    let q = 2 * self.0.prime().as_u32() - 2;
                    if (degree as u32) % q != 0 {
                        return Err(PyValueError::new_err(format!(
                            "coproduct expects a degree divisible by {q}, got {degree}"
                        )));
                    }
                }
            } else if idx != 0 {
                return Err(PyValueError::new_err(
                    "at p = 2 the coproduct expects index 0",
                ));
            }
            Ok(self.0.coproduct(degree, idx))
        }

        pub fn decompose(&self, degree: i32, idx: usize) -> PyResult<Vec<(i32, usize)>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.decompose(degree, idx))
        }

        // --- Adem-specific methods --------------------------------------------

        pub fn generic(&self) -> bool {
            self.0.generic()
        }

        pub fn q(&self) -> i32 {
            self.0.q()
        }

        pub fn basis_element_from_index(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<AdemBasisElement> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(AdemBasisElement(
                self.0.basis_element_from_index(degree, idx).clone(),
            ))
        }

        pub fn try_basis_element_to_index(
            &self,
            elt: PyRef<'_, AdemBasisElement>,
        ) -> Option<usize> {
            if elt.0.degree < 0 {
                return None;
            }
            self.ensure_basis(elt.0.degree);
            self.0.try_basis_element_to_index(&elt.0)
        }

        /// Like `try_basis_element_to_index`, but raises `ValueError` if the
        /// element is not in the algebra (upstream panics).
        pub fn basis_element_to_index(&self, elt: PyRef<'_, AdemBasisElement>) -> PyResult<usize> {
            non_negative_degree(elt.0.degree)?;
            self.ensure_basis(elt.0.degree);
            self.0
                .try_basis_element_to_index(&elt.0)
                .ok_or_else(|| PyValueError::new_err(format!("element not in algebra: {}", elt.0)))
        }

        /// The degree and index of `b^e P^x`. Raises `ValueError` if that
        /// element is not in the algebra (upstream `basis_element_to_index`
        /// panics for an inadmissible `(e, x)`).
        pub fn beps_pn(&self, e: u32, x: u32) -> PyResult<(i32, usize)> {
            // Mirror the upstream `x == 0` short-circuits, which never index.
            if x == 0 && e == 1 {
                return Ok((1, 0));
            } else if x == 0 {
                return Ok((0, 0));
            }
            let p = self.0.prime().as_u32();
            let q = if self.0.generic() { 2 * p - 2 } else { 1 };
            let degree = x
                .checked_mul(q)
                .and_then(|v| v.checked_add(e))
                .ok_or_else(|| PyValueError::new_err("degree overflows"))?
                as i32;
            self.ensure_basis(degree);
            let elt = ::algebra::adem_algebra::AdemBasisElement {
                degree,
                bocksteins: e,
                ps: vec![x],
                p_or_sq: p != 2,
            };
            self.0
                .try_basis_element_to_index(&elt)
                .map(|idx| (degree, idx))
                .ok_or_else(|| PyValueError::new_err(format!("b^{e} P^{x} is not in the algebra")))
        }

        pub fn __repr__(&self) -> String {
            format!("{}", self.0)
        }
    }

    /// The `enum_dispatch` union of the Adem and Milnor Steenrod algebras
    /// (`::algebra::SteenrodAlgebra`). A single value is *either* Adem or Milnor
    /// at runtime; every `Algebra`/`GeneratedAlgebra`/`Bialgebra` method
    /// dispatches to the active variant. This is one pyclass that wraps the
    /// union and dispatches; it does not inherit from `MilnorAlgebra`/
    /// `AdemAlgebra`.
    #[pyclass]
    pub struct SteenrodAlgebra(::algebra::SteenrodAlgebra);

    impl SteenrodAlgebra {
        /// Lazily compute book-keeping up to `degree`. Both underlying algebras
        /// are infinite-dimensional with `OnceVec` tables that panic when
        /// indexed past the computed range, so every degree-indexed Python
        /// method funnels through here first (idempotent; no-op for negative
        /// degrees). The dispatch is identical for either variant.
        fn ensure_basis(&self, degree: i32) {
            if degree >= 0 {
                self.0.compute_basis(degree);
            }
        }

        fn product_target(&self, r_degree: i32, s_degree: i32) -> PyResult<i32> {
            non_negative_degree(r_degree)?;
            non_negative_degree(s_degree)?;
            let target = r_degree
                .checked_add(s_degree)
                .ok_or_else(|| PyValueError::new_err("product degree overflows i32"))?;
            self.ensure_basis(target);
            Ok(target)
        }

        fn checked_basis_index(&self, degree: i32, idx: usize) -> PyResult<()> {
            let dim = self.0.dimension(degree);
            if idx < dim {
                Ok(())
            } else {
                Err(PyIndexError::new_err(format!(
                    "index {idx} out of range for degree {degree} (dimension {dim})"
                )))
            }
        }
    }

    #[pymethods]
    impl SteenrodAlgebra {
        // --- §5.2 constructors ------------------------------------------------

        /// Construct a `SteenrodAlgebra` from a module-spec `dict` (the JSON the
        /// crate reads from a module file), the desired `AlgebraType`, and the
        /// `unstable` flag. Mirrors `::algebra::SteenrodAlgebra::from_json`,
        /// which reads `{"p": <int>, "algebra": [..]?, "profile": {..}?}`. If
        /// the spec's `algebra` list does not contain the requested type, the
        /// upstream falls back to the first listed type. The prime is validated
        /// by serde's `ValidPrime` deserializer; a bad spec/prime maps to
        /// `ValueError` (parse) or `PyRuntimeError` (other `anyhow` errors),
        /// consistent with API_PROPOSAL §2.4.
        #[staticmethod]
        #[pyo3(signature = (value, ty, unstable = false))]
        pub fn from_json(
            value: &Bound<'_, PyAny>,
            ty: AlgebraType,
            unstable: bool,
        ) -> PyResult<Self> {
            let json = py_to_json(value)?;
            ::algebra::SteenrodAlgebra::from_json(&json, ty.into(), unstable)
                .map(SteenrodAlgebra)
                .map_err(|e| {
                    use pyo3::exceptions::PyRuntimeError;
                    PyRuntimeError::new_err(e.to_string())
                })
        }

        /// Construct the Adem variant at prime `p`. Validates the prime ->
        /// `ValueError`.
        #[staticmethod]
        #[pyo3(signature = (p, unstable = false))]
        pub fn adem(p: u32, unstable: bool) -> PyResult<Self> {
            let p = valid_prime(p)?;
            Ok(SteenrodAlgebra(::algebra::SteenrodAlgebra::AdemAlgebra(
                ::algebra::AdemAlgebra::new(p, unstable),
            )))
        }

        /// Construct the Milnor variant at prime `p`. Validates the prime ->
        /// `ValueError`.
        #[staticmethod]
        #[pyo3(signature = (p, unstable = false))]
        pub fn milnor(p: u32, unstable: bool) -> PyResult<Self> {
            let p = valid_prime(p)?;
            Ok(SteenrodAlgebra(::algebra::SteenrodAlgebra::MilnorAlgebra(
                ::algebra::MilnorAlgebra::new(p, unstable),
            )))
        }

        /// Which variant this value is (`AlgebraType.ADEM`/`MILNOR`).
        pub fn algebra_type(&self) -> AlgebraType {
            match &self.0 {
                ::algebra::SteenrodAlgebra::AdemAlgebra(_) => AlgebraType::Adem,
                ::algebra::SteenrodAlgebra::MilnorAlgebra(_) => AlgebraType::Milnor,
            }
        }

        // --- Algebra trait surface --------------------------------------------

        /// The prime as a plain `int` (`ValidPrime` is never exposed).
        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn compute_basis(&self, degree: i32) {
            self.ensure_basis(degree);
        }

        pub fn dimension(&self, degree: i32) -> usize {
            if degree < 0 {
                return 0;
            }
            self.ensure_basis(degree);
            self.0.dimension(degree)
        }

        pub fn basis_element_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.basis_element_to_string(degree, idx))
        }

        /// Parse a basis element, returning `(degree, index)`. Raises
        /// `ValueError` if the string does not parse or names an element not in
        /// this algebra.
        ///
        /// The union dispatches straight to the active variant's
        /// `basis_element_from_string`, so it inherits exactly the panic the
        /// `MilnorAlgebra`/`AdemAlgebra` bindings guard against: a parseable but
        /// absent/inadmissible name (e.g. `"Sq0"`) calls the panicking
        /// `basis_element_to_index`. We contain it with `catch_unwind` and
        /// translate to `ValueError`, exactly as those bindings do (sound for
        /// the same reason: the panic is a failed lookup after `compute_basis`
        /// has returned, not a half-finished mutation).
        pub fn basis_element_from_string(&self, elt: &str) -> PyResult<(i32, usize)> {
            use std::panic::{catch_unwind, AssertUnwindSafe};
            match catch_unwind(AssertUnwindSafe(|| self.0.basis_element_from_string(elt))) {
                Ok(Some(res)) => Ok(res),
                Ok(None) => Err(PyValueError::new_err(format!(
                    "could not parse basis element: {elt}"
                ))),
                Err(_) => Err(PyValueError::new_err(format!(
                    "{elt} does not name a basis element of this algebra"
                ))),
            }
        }

        pub fn element_to_string(
            &self,
            py: Python<'_>,
            degree: i32,
            element: &Bound<'_, PyAny>,
        ) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            let element = crate::fp_py::extract_input_owned(py, element)?;
            checked_same_prime(element.prime().as_u32(), self.0.prime().as_u32())?;
            checked_equal_len(element.len(), self.0.dimension(degree))?;
            Ok(self.0.element_to_string(degree, element.as_slice()))
        }

        pub fn multiply_basis_elements(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            // Reduce mod p before handing to upstream, which computes
            // `coeff * value` before reducing and would overflow for large
            // `coeff`. The algebra is over F_p, so this is equivalent.
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            self.checked_basis_index(s_degree, s_idx)?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0
                    .multiply_basis_elements(res.copy(), coeff, r_degree, r_idx, s_degree, s_idx);
                Ok(())
            })
        }

        pub fn multiply_basis_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r_idx: usize,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(r_degree, r_idx)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_basis_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r_idx,
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_basis_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s_idx: usize,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            self.checked_basis_index(s_degree, s_idx)?;
            let r = crate::fp_py::extract_input_owned(py, r)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_basis_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s_idx,
                );
                Ok(())
            })
        }

        pub fn multiply_element_by_element(
            &self,
            py: Python<'_>,
            result: &Bound<'_, PyAny>,
            coeff: u32,
            r_degree: i32,
            r: &Bound<'_, PyAny>,
            s_degree: i32,
            s: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let p = self.0.prime().as_u32();
            let coeff = coeff % p;
            let target = self.product_target(r_degree, s_degree)?;
            let dim = self.0.dimension(target);
            let r = crate::fp_py::extract_input_owned(py, r)?;
            let s = crate::fp_py::extract_input_owned(py, s)?;
            checked_same_prime(r.prime().as_u32(), p)?;
            checked_same_prime(s.prime().as_u32(), p)?;
            checked_equal_len(r.len(), self.0.dimension(r_degree))?;
            checked_equal_len(s.len(), self.0.dimension(s_degree))?;
            crate::fp_py::with_target_slice_mut(py, result, |mut res| {
                checked_same_prime(res.prime().as_u32(), p)?;
                checked_result_len(res.as_slice().len(), dim)?;
                self.0.multiply_element_by_element(
                    res.copy(),
                    coeff,
                    r_degree,
                    r.as_slice(),
                    s_degree,
                    s.as_slice(),
                );
                Ok(())
            })
        }

        pub fn default_filtration_one_products(&self) -> Vec<(String, i32, usize)> {
            self.0.default_filtration_one_products()
        }

        // --- GeneratedAlgebra trait surface -----------------------------------

        pub fn generators(&self, degree: i32) -> PyResult<Vec<usize>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generators(degree))
        }

        pub fn generator_to_string(&self, degree: i32, idx: usize) -> PyResult<String> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.generator_to_string(degree, idx))
        }

        pub fn decompose_basis_element(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(u32, (i32, usize), (i32, usize))>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            // Decomposition is invalid for indecomposables. The union dispatches
            // to the active variant's (panicking) implementation, so we apply
            // the same per-variant guard the `MilnorAlgebra`/`AdemAlgebra`
            // bindings use: for Milnor, the degree-0 unit has an empty `p_part`
            // and underflows in `decompose_basis_element_ppart`; for Adem the
            // degree-0 unit and the algebra generators index out of bounds /
            // hit a panicking `basis_element_to_index`.
            match &self.0 {
                ::algebra::SteenrodAlgebra::MilnorAlgebra(a) => {
                    let basis = a.basis_element_from_index(degree, idx);
                    if basis.q_part == 0 && basis.p_part.is_empty() {
                        return Err(PyValueError::new_err("the degree-0 unit is indecomposable"));
                    }
                }
                ::algebra::SteenrodAlgebra::AdemAlgebra(a) => {
                    if degree == 0 || a.generators(degree).contains(&idx) {
                        return Err(PyValueError::new_err(
                            "the unit and algebra generators are indecomposable",
                        ));
                    }
                }
            }
            Ok(self.0.decompose_basis_element(degree, idx))
        }

        pub fn generating_relations(
            &self,
            degree: i32,
        ) -> PyResult<Vec<Vec<(u32, (i32, usize), (i32, usize))>>> {
            if degree < 0 {
                return Ok(Vec::new());
            }
            self.ensure_basis(degree);
            Ok(self.0.generating_relations(degree))
        }

        // --- Bialgebra trait surface ------------------------------------------

        /// Compute a coproduct. The underlying assertions differ by variant, so
        /// we apply the same guards the concrete bindings use: Milnor only
        /// supports `p = 2`; generic Adem expects a degree divisible by
        /// `q = 2p - 2` (except degree 1), and `p = 2` Adem expects index 0.
        pub fn coproduct(
            &self,
            degree: i32,
            idx: usize,
        ) -> PyResult<Vec<(i32, usize, i32, usize)>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            match &self.0 {
                ::algebra::SteenrodAlgebra::MilnorAlgebra(_) => {
                    if self.0.prime().as_u32() != 2 {
                        return Err(PyValueError::new_err(
                            "coproduct is only supported at p = 2",
                        ));
                    }
                }
                ::algebra::SteenrodAlgebra::AdemAlgebra(a) => {
                    if a.generic() {
                        if degree != 1 {
                            let q = 2 * self.0.prime().as_u32() - 2;
                            if (degree as u32) % q != 0 {
                                return Err(PyValueError::new_err(format!(
                                    "coproduct expects a degree divisible by {q}, got {degree}"
                                )));
                            }
                        }
                    } else if idx != 0 {
                        return Err(PyValueError::new_err(
                            "at p = 2 the coproduct expects index 0",
                        ));
                    }
                }
            }
            Ok(self.0.coproduct(degree, idx))
        }

        pub fn decompose(&self, degree: i32, idx: usize) -> PyResult<Vec<(i32, usize)>> {
            non_negative_degree(degree)?;
            self.ensure_basis(degree);
            self.checked_basis_index(degree, idx)?;
            Ok(self.0.decompose(degree, idx))
        }

        pub fn __repr__(&self) -> String {
            format!("{}", self.0)
        }
    }

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Arbitrary code to run at the module initialization
        // m.add("double2", m.getattr("double")?)
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn new_rejects_non_prime() {
            assert!(MilnorAlgebra::new(4, false).is_err());
            assert!(MilnorAlgebra::new(0, false).is_err());
            assert!(MilnorAlgebra::new(1, false).is_err());
            assert!(MilnorAlgebra::new(2, false).is_ok());
            assert!(MilnorAlgebra::new(3, false).is_ok());
        }

        #[test]
        fn basic_invariants_p2() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            assert_eq!(a.prime(), 2);
            assert!(!a.generic());
            assert_eq!(a.q(), 1);
            assert!(a.profile().is_trivial());
        }

        #[test]
        fn dimension_p2() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            assert_eq!(a.dimension(0), 1);
            assert_eq!(a.dimension(1), 1);
            assert_eq!(a.dimension(2), 1);
            assert_eq!(a.dimension(3), 2);
            assert_eq!(a.dimension(4), 2);
            // Negative degrees are empty, never panicking.
            assert_eq!(a.dimension(-3), 0);
        }

        #[test]
        fn basis_element_string_roundtrip_p2() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(6);
            for d in 0..=6 {
                for i in 0..a.dimension(d) {
                    let s = a.basis_element_to_string(d, i).unwrap();
                    assert_eq!(a.basis_element_from_string(&s).unwrap(), (d, i));
                }
            }
        }

        #[test]
        fn basis_element_index_roundtrip_p2() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(6);
            Python::initialize();
            Python::attach(|py| {
                for d in 0..=6 {
                    for i in 0..a.dimension(d) {
                        let elt = a.basis_element_from_index(d, i).unwrap();
                        assert_eq!(elt.0.degree, d);
                        let elt = Py::new(py, elt).unwrap();
                        assert_eq!(a.basis_element_to_index(elt.borrow(py)).unwrap(), i);
                    }
                }
            });
        }

        #[test]
        fn beps_pn_and_out_of_range() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            // Q_1^0 P(1) = Sq^1 lives in degree 1.
            assert_eq!(a.beps_pn(0, 1).unwrap().0, 1);
            // Out-of-range basis index raises rather than panicking.
            assert!(a.basis_element_to_string(2, 99).is_err());
            assert!(a.basis_element_to_string(-1, 0).is_err());
        }

        #[test]
        fn coproduct_odd_prime_raises() {
            let a = MilnorAlgebra::new(3, false).unwrap();
            a.compute_basis(4);
            assert!(a.coproduct(0, 0).is_err());
        }

        #[test]
        #[should_panic]
        fn confirm_upstream_basis_element_from_string_panics() {
            // Documents the upstream panic the binding guards against: calling
            // the raw upstream method (no `catch_unwind`) on a parseable but
            // absent name aborts.
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            let _ = a.0.basis_element_from_string("Sq0");
        }

        #[test]
        #[should_panic]
        fn confirm_upstream_decompose_unit_panics() {
            // Documents the underflow the binding guards against.
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(4);
            let _ = a.0.decompose_basis_element(0, 0);
        }

        #[test]
        fn basis_element_from_string_rejects_absent_names() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            // Parseable but absent / out-of-range names must raise, not panic.
            assert!(a.basis_element_from_string("Sq0").is_err());
            assert!(a.basis_element_from_string("P0").is_err());
            assert!(a.basis_element_from_string("Q_5").is_err());
            // Pure nonsense still raises.
            assert!(a.basis_element_from_string("not an element").is_err());
            // Valid names still round-trip.
            for d in 0..=6 {
                for i in 0..a.dimension(d) {
                    let s = a.basis_element_to_string(d, i).unwrap();
                    assert_eq!(a.basis_element_from_string(&s).unwrap(), (d, i));
                }
            }
        }

        #[test]
        fn decompose_degree_zero_unit_raises() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            a.compute_basis(4);
            // The degree-0 unit is indecomposable; must raise, not underflow.
            assert!(a.decompose_basis_element(0, 0).is_err());
            // A non-trivial decomposable element still works (Sq^3 = Sq^1 Sq^2-ish).
            // Degree 3 has a decomposable basis element.
            let _ = a.decompose_basis_element(3, 0);
        }

        #[test]
        fn ppart_table_p2() {
            let a = MilnorAlgebra::new(2, false).unwrap();
            // degree 0 has the empty partition.
            assert_eq!(a.ppart_table(0).unwrap(), vec![Vec::<u32>::new()]);
            assert!(a.ppart_table(-1).is_err());
        }

        // --- AdemAlgebra ------------------------------------------------------

        #[test]
        fn adem_new_rejects_non_prime() {
            assert!(AdemAlgebra::new(4, false).is_err());
            assert!(AdemAlgebra::new(0, false).is_err());
            assert!(AdemAlgebra::new(1, false).is_err());
            assert!(AdemAlgebra::new(2, false).is_ok());
            assert!(AdemAlgebra::new(3, false).is_ok());
        }

        #[test]
        fn adem_basic_invariants() {
            let a = AdemAlgebra::new(2, false).unwrap();
            assert_eq!(a.prime(), 2);
            assert!(!a.generic());
            assert_eq!(a.q(), 1);

            let a3 = AdemAlgebra::new(3, false).unwrap();
            assert!(a3.generic());
            assert_eq!(a3.q(), 4);
        }

        #[test]
        fn adem_dimension_p2() {
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            assert_eq!(a.dimension(0), 1);
            assert_eq!(a.dimension(1), 1);
            assert_eq!(a.dimension(2), 1);
            assert_eq!(a.dimension(3), 2); // Sq3, Sq2 Sq1
            assert_eq!(a.dimension(-3), 0);
        }

        #[test]
        fn adem_known_products_p2() {
            use ::fp::vector::FpVector;
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);

            // Sq1 * Sq1 = 0.
            let mut v = FpVector::new(::fp::prime::TWO, a.dimension(2));
            a.0.multiply_basis_elements(v.as_slice_mut(), 1, 1, 0, 1, 0);
            assert!(v.is_zero(), "Sq1 * Sq1 should be 0");

            // Sq1 * Sq2 = Sq3 (admissible, single basis term in degree 3).
            let mut v = FpVector::new(::fp::prime::TWO, a.dimension(3));
            a.0.multiply_basis_elements(v.as_slice_mut(), 1, 1, 0, 2, 0);
            let (_d, expected_idx) = a.basis_element_from_string("Sq3").unwrap();
            assert_eq!(v.entry(expected_idx), 1);
            assert_eq!(v.iter_nonzero().count(), 1);
        }

        #[test]
        fn adem_string_roundtrip_p2() {
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            for d in 0..=8 {
                for i in 0..a.dimension(d) {
                    let s = a.basis_element_to_string(d, i).unwrap();
                    assert_eq!(a.basis_element_from_string(&s).unwrap(), (d, i));
                }
            }
        }

        #[test]
        fn adem_index_roundtrip_p2() {
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            Python::initialize();
            Python::attach(|py| {
                for d in 0..=8 {
                    for i in 0..a.dimension(d) {
                        let elt = a.basis_element_from_index(d, i).unwrap();
                        let elt = Py::new(py, elt).unwrap();
                        assert_eq!(a.basis_element_to_index(elt.borrow(py)).unwrap(), i);
                    }
                }
            });
        }

        #[test]
        fn adem_beps_pn_and_out_of_range() {
            let a = AdemAlgebra::new(2, false).unwrap();
            // P^1 = Sq1 lives in degree 1.
            assert_eq!(a.beps_pn(0, 1).unwrap().0, 1);
            assert!(a.basis_element_to_string(2, 99).is_err());
            assert!(a.basis_element_to_string(-1, 0).is_err());
        }

        #[test]
        #[should_panic]
        fn confirm_upstream_adem_from_string_panics() {
            // Documents the upstream panic the binding guards against: a
            // parseable but absent name aborts via basis_element_to_index.
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            let _ = a.0.basis_element_from_string("Sq0");
        }

        #[test]
        fn adem_from_string_rejects_absent_names() {
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            assert!(a.basis_element_from_string("Sq0").is_err());
            assert!(a.basis_element_from_string("Sq1 Sq1").is_err());
            assert!(a.basis_element_from_string("not an element").is_err());
            for d in 0..=6 {
                for i in 0..a.dimension(d) {
                    let s = a.basis_element_to_string(d, i).unwrap();
                    assert_eq!(a.basis_element_from_string(&s).unwrap(), (d, i));
                }
            }
        }

        #[test]
        fn adem_decompose_unit_and_generator_raise() {
            let a = AdemAlgebra::new(2, false).unwrap();
            a.compute_basis(8);
            // Degree-0 unit is indecomposable.
            assert!(a.decompose_basis_element(0, 0).is_err());
            // Sq2 is a generator (degree 2, idx 0) -> indecomposable.
            assert!(a.decompose_basis_element(2, 0).is_err());
            // A non-generator decomposes successfully.
            assert!(a.decompose_basis_element(3, 0).is_ok());
        }

        // --- SteenrodAlgebra --------------------------------------------------

        #[test]
        fn steenrod_constructors_and_variant() {
            assert!(SteenrodAlgebra::adem(4, false).is_err());
            assert!(SteenrodAlgebra::milnor(0, false).is_err());

            let adem = SteenrodAlgebra::adem(2, false).unwrap();
            assert_eq!(adem.prime(), 2);
            assert_eq!(adem.algebra_type(), AlgebraType::Adem);

            let milnor = SteenrodAlgebra::milnor(3, false).unwrap();
            assert_eq!(milnor.prime(), 3);
            assert_eq!(milnor.algebra_type(), AlgebraType::Milnor);
        }

        #[test]
        fn steenrod_dimension_p2() {
            for a in [
                SteenrodAlgebra::adem(2, false).unwrap(),
                SteenrodAlgebra::milnor(2, false).unwrap(),
            ] {
                a.compute_basis(8);
                assert_eq!(a.dimension(0), 1);
                assert_eq!(a.dimension(1), 1);
                assert_eq!(a.dimension(2), 1);
                assert_eq!(a.dimension(3), 2);
                assert_eq!(a.dimension(-3), 0);
            }
        }

        #[test]
        fn steenrod_known_products_p2() {
            use ::fp::vector::FpVector;
            for a in [
                SteenrodAlgebra::adem(2, false).unwrap(),
                SteenrodAlgebra::milnor(2, false).unwrap(),
            ] {
                a.compute_basis(8);
                // Sq1 * Sq1 = 0.
                let mut v = FpVector::new(::fp::prime::TWO, a.dimension(2));
                a.0.multiply_basis_elements(v.as_slice_mut(), 1, 1, 0, 1, 0);
                assert!(v.is_zero(), "Sq1 * Sq1 should be 0");

                // Sq1 * Sq2 = Sq3 (single basis term in degree 3).
                let mut v = FpVector::new(::fp::prime::TWO, a.dimension(3));
                a.0.multiply_basis_elements(v.as_slice_mut(), 1, 1, 0, 2, 0);
                let (_d, idx) = a.basis_element_from_string("Sq3").unwrap();
                assert_eq!(v.entry(idx), 1);
                assert_eq!(v.iter_nonzero().count(), 1);
            }
        }

        #[test]
        fn steenrod_string_roundtrip_and_absent_names() {
            for a in [
                SteenrodAlgebra::adem(2, false).unwrap(),
                SteenrodAlgebra::milnor(2, false).unwrap(),
            ] {
                a.compute_basis(8);
                for d in 0..=8 {
                    for i in 0..a.dimension(d) {
                        let s = a.basis_element_to_string(d, i).unwrap();
                        assert_eq!(a.basis_element_from_string(&s).unwrap(), (d, i));
                    }
                }
                // Parseable but absent names raise, never panic.
                assert!(a.basis_element_from_string("Sq0").is_err());
                assert!(a.basis_element_from_string("not an element").is_err());
                // Degree-0 unit is indecomposable.
                assert!(a.decompose_basis_element(0, 0).is_err());
                // A non-generator decomposes successfully.
                assert!(a.decompose_basis_element(3, 0).is_ok());
                // Out-of-range index raises.
                assert!(a.basis_element_to_string(2, 99).is_err());
            }
        }

        #[test]
        fn steenrod_from_json_roundtrip() {
            Python::initialize();
            Python::attach(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("p", 2).unwrap();
                let adem =
                    SteenrodAlgebra::from_json(dict.as_any(), AlgebraType::Adem, false).unwrap();
                assert_eq!(adem.prime(), 2);
                assert_eq!(adem.algebra_type(), AlgebraType::Adem);

                let milnor =
                    SteenrodAlgebra::from_json(dict.as_any(), AlgebraType::Milnor, false).unwrap();
                assert_eq!(milnor.prime(), 2);
                assert_eq!(milnor.algebra_type(), AlgebraType::Milnor);

                // A bad prime in the spec surfaces as an error, not a panic.
                let bad = pyo3::types::PyDict::new(py);
                bad.set_item("p", 4).unwrap();
                assert!(
                    SteenrodAlgebra::from_json(bad.as_any(), AlgebraType::Adem, false).is_err()
                );
            });
        }
    }
}
