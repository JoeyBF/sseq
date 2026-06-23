use pyo3::prelude::*;

#[pymodule]
pub mod fp_py {
    use fp::field::{
        element::FieldElement as RustFieldElement, Field, Fp as RustFp, SmallFq as RustSmallFq,
    };
    use fp::matrix::{
        AffineSubspace as RustAffineSubspace, AugmentedMatrix as RustAugmentedMatrix,
        Matrix as RustMatrix, MatrixSliceMut as RustMatrixSliceMut,
        QuasiInverse as RustQuasiInverse, Subquotient as RustSubquotient, Subspace as RustSubspace,
    };
    use fp::prime::{self, Binomial, Prime};
    use fp::vector::{
        FpSlice as RustFpSlice, FpSliceMut as RustFpSliceMut, FpVector as RustFpVector,
    };
    use pyo3::basic::CompareOp;
    use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError, PyZeroDivisionError};
    use pyo3::types::PyBytes;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::io::Cursor;

    use super::*;

    const MAX_VALID_PRIME: u32 = 1 << 31;

    type DynFp = RustFp<prime::ValidPrime>;
    type DynSmallFq = RustSmallFq<prime::ValidPrime>;
    type DynFpElement = RustFieldElement<DynFp>;
    type DynSmallFqElement = RustFieldElement<DynSmallFq>;

    #[pyclass(name = "Fp", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    struct PyFp(DynFp);

    #[pyclass(name = "SmallFq", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    struct PySmallFq(DynSmallFq);

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    enum FieldElementKind {
        Fp(DynFpElement),
        SmallFq(DynSmallFqElement),
    }

    #[pyclass(name = "FieldElement", frozen, from_py_object)]
    #[derive(Clone, Copy)]
    struct PyFieldElement(FieldElementKind);

    #[pyclass(name = "FpVector")]
    struct PyFpVector(RustFpVector);

    /// A matrix-like parent that can back a borrowed row or rectangle view.
    ///
    /// A plain `Matrix` is held directly; an `AugmentedMatrix<N>` is held as its
    /// concrete pyclass and accessed through its `Deref<Target = Matrix>` so
    /// that segment rectangles and segment rows can revalidate against the inner
    /// matrix's current dimensions. We keep the parent Python object alive and
    /// reconstruct the underlying Rust matrix view on each call.
    enum MatrixParent {
        Matrix(Py<PyMatrix>),
        Augmented2(Py<PyAugmentedMatrix2>),
        Augmented3(Py<PyAugmentedMatrix3>),
    }

    impl MatrixParent {
        fn clone_ref(&self, py: Python<'_>) -> Self {
            match self {
                Self::Matrix(m) => Self::Matrix(m.clone_ref(py)),
                Self::Augmented2(m) => Self::Augmented2(m.clone_ref(py)),
                Self::Augmented3(m) => Self::Augmented3(m.clone_ref(py)),
            }
        }

        /// Run `f` on the current inner `Matrix`, holding the borrow for the
        /// duration of the call. Deref coercion turns an `&AugmentedMatrix<N>`
        /// into the `&Matrix` expected by `f`.
        fn with_matrix<R>(&self, py: Python<'_>, f: impl FnOnce(&RustMatrix) -> R) -> PyResult<R> {
            match self {
                Self::Matrix(m) => Ok(f(&m.try_borrow(py).map_err(borrow_error)?.0)),
                Self::Augmented2(m) => Ok(f(&m.try_borrow(py).map_err(borrow_error)?.0)),
                Self::Augmented3(m) => Ok(f(&m.try_borrow(py).map_err(borrow_error)?.0)),
            }
        }

        /// Run `f` on the current inner `Matrix` mutably, holding the borrow for
        /// the duration of the call.
        fn with_matrix_mut<R>(
            &self,
            py: Python<'_>,
            f: impl FnOnce(&mut RustMatrix) -> R,
        ) -> PyResult<R> {
            match self {
                Self::Matrix(m) => Ok(f(&mut m.try_borrow_mut(py).map_err(borrow_error)?.0)),
                Self::Augmented2(m) => Ok(f(&mut m.try_borrow_mut(py).map_err(borrow_error)?.0)),
                Self::Augmented3(m) => Ok(f(&mut m.try_borrow_mut(py).map_err(borrow_error)?.0)),
            }
        }
    }

    /// The source backing a slice handle: either an owned vector, or a row of a
    /// matrix-like parent. In both cases we keep the parent Python object alive
    /// and store enough metadata to reconstruct the underlying Rust slice on
    /// each call, revalidating against the parent's current dimensions first.
    enum SliceParent {
        Vector(Py<PyFpVector>),
        MatrixRow { matrix: MatrixParent, row: usize },
    }

    impl SliceParent {
        fn clone_ref(&self, py: Python<'_>) -> Self {
            match self {
                Self::Vector(v) => Self::Vector(v.clone_ref(py)),
                Self::MatrixRow { matrix, row } => Self::MatrixRow {
                    matrix: matrix.clone_ref(py),
                    row: *row,
                },
            }
        }
    }

    /// Run `f` on the reconstructed immutable slice for `parent[start..end]`,
    /// after revalidating the parent's current dimensions.
    ///
    /// Revalidation only guards the parent's current *dimensions* (vector length
    /// or matrix row count and row length). It deliberately does not track
    /// logical-coordinate remapping: an operation like `Matrix::trim` with
    /// `col_start > 0` shifts the data backwards in each row without shrinking it
    /// below the slice's `end`, so a surviving slice silently reads the remapped
    /// columns rather than raising. Preventing that would require tracking the
    /// origin of every coordinate, which is out of scope for the
    /// handle+range design.
    fn with_parent_slice<R>(
        parent: &SliceParent,
        start: usize,
        end: usize,
        py: Python<'_>,
        f: impl FnOnce(RustFpSlice<'_>) -> R,
    ) -> PyResult<R> {
        match parent {
            SliceParent::Vector(v) => {
                let parent = v.try_borrow(py).map_err(borrow_error)?;
                checked_range(start, end, parent.0.len())?;
                Ok(f(parent.0.slice(start, end)))
            }
            SliceParent::MatrixRow { matrix, row } => matrix.with_matrix(py, |m| {
                checked_row(*row, m.rows())?;
                let full = m.row(*row);
                checked_range(start, end, full.len())?;
                Ok(f(full.restrict(start, end)))
            })?,
        }
    }

    /// Run `f` on the reconstructed mutable slice for `parent[start..end]`,
    /// after revalidating the parent's current dimensions.
    fn with_parent_slice_mut<R>(
        parent: &SliceParent,
        start: usize,
        end: usize,
        py: Python<'_>,
        f: impl FnOnce(RustFpSliceMut<'_>) -> R,
    ) -> PyResult<R> {
        match parent {
            SliceParent::Vector(v) => {
                let mut parent = v.try_borrow_mut(py).map_err(borrow_error)?;
                checked_range(start, end, parent.0.len())?;
                Ok(f(parent.0.slice_mut(start, end)))
            }
            SliceParent::MatrixRow { matrix, row } => matrix.with_matrix_mut(py, |m| {
                checked_row(*row, m.rows())?;
                // Validate against the actual current row length, matching the
                // read path (`with_parent_slice`). For a `Matrix` this equals
                // `columns()`, but deriving it from the row keeps both paths
                // consistent regardless of that invariant.
                let row_len = m.row(*row).len();
                checked_range(start, end, row_len)?;
                Ok(f(m.row_mut(*row).slice_mut(start, end)))
            })?,
        }
    }

    #[pyclass(name = "FpSlice")]
    struct PyFpSlice {
        parent: SliceParent,
        start: usize,
        end: usize,
    }

    #[pyclass(name = "FpSliceMut")]
    struct PyFpSliceMut {
        parent: SliceParent,
        start: usize,
        end: usize,
    }

    #[pyclass(name = "FpVectorIterator")]
    struct PyFpVectorIterator {
        entries: Vec<u32>,
        index: usize,
    }

    /// A borrowed mutable rectangular view into a matrix-like parent. We hold
    /// the parent plus the rectangle (row range + column range) and reconstruct
    /// the Rust `MatrixSliceMut` on each call, revalidating the rectangle
    /// against the parent's current dimensions first.
    #[pyclass(name = "MatrixSliceMut")]
    struct PyMatrixSliceMut {
        parent: MatrixParent,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    }

    #[pyclass(name = "Matrix")]
    struct PyMatrix(RustMatrix);

    #[pyclass(name = "Subspace")]
    struct PySubspace(RustSubspace);

    #[pyclass(name = "QuasiInverse")]
    struct PyQuasiInverse(RustQuasiInverse);

    #[pyclass(name = "Subquotient")]
    struct PySubquotient(RustSubquotient);

    #[pyclass(name = "AffineSubspace")]
    struct PyAffineSubspace(RustAffineSubspace);

    /// Lazy iterator over every vector in a subspace.
    ///
    /// The upstream `Subspace::iter_all_vectors` iterator borrows the subspace,
    /// so it cannot be stored alongside an owned subspace in a `#[pyclass]`
    /// without a self-referential struct. Instead we own a clone of the
    /// subspace and an index counter, regenerating the i-th vector on each
    /// `__next__` from the base-`p` decomposition of the index. This keeps
    /// iteration lazy (O(1) memory) while yielding the same owned `FpVector`s
    /// in the same order as the eager version.
    #[pyclass(name = "SubspaceVectorIterator")]
    struct PySubspaceVectorIterator {
        subspace: RustSubspace,
        index: u128,
        total: u128,
    }

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

    fn checked_index(index: usize, len: usize) -> PyResult<usize> {
        if index < len {
            Ok(index)
        } else {
            Err(PyIndexError::new_err(format!(
                "index {index} out of range for vector of length {len}"
            )))
        }
    }

    fn py_index(index: isize, len: usize) -> PyResult<usize> {
        let index = if index < 0 {
            len as isize + index
        } else {
            index
        };
        if index >= 0 && (index as usize) < len {
            Ok(index as usize)
        } else {
            Err(PyIndexError::new_err(format!(
                "index {index} out of range for vector of length {len}"
            )))
        }
    }

    fn checked_range(start: usize, end: usize, len: usize) -> PyResult<()> {
        if start <= end && end <= len {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!(
                "range {start}..{end} out of range for vector of length {len}"
            )))
        }
    }

    fn borrow_error(err: impl ToString) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
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

    fn checked_same_prime(lhs: u32, rhs: u32) -> PyResult<()> {
        if lhs == rhs {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "prime mismatch: {lhs} != {rhs}"
            )))
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

    impl PyFpSlice {
        fn with_slice<R>(
            &self,
            py: Python<'_>,
            f: impl FnOnce(RustFpSlice<'_>) -> R,
        ) -> PyResult<R> {
            with_parent_slice(&self.parent, self.start, self.end, py, f)
        }

        /// Cached span of the handle, used only for computing index bounds.
        /// This does NOT revalidate the parent; callers that touch the parent
        /// go through `with_slice`/`with_slice_mut`, which revalidate.
        fn span(&self) -> usize {
            self.end - self.start
        }

        fn to_owned_checked(&self, py: Python<'_>) -> PyResult<RustFpVector> {
            self.with_slice(py, |s| s.to_owned())
        }
    }

    impl PyFpSliceMut {
        fn with_slice<R>(
            &self,
            py: Python<'_>,
            f: impl FnOnce(RustFpSlice<'_>) -> R,
        ) -> PyResult<R> {
            with_parent_slice(&self.parent, self.start, self.end, py, f)
        }

        /// Cached span of the handle, used only for computing index bounds.
        /// This does NOT revalidate the parent; callers that touch the parent
        /// go through `with_slice`/`with_slice_mut`, which revalidate.
        fn span(&self) -> usize {
            self.end - self.start
        }

        fn with_slice_mut<R>(
            &self,
            py: Python<'_>,
            f: impl FnOnce(RustFpSliceMut<'_>) -> R,
        ) -> PyResult<R> {
            with_parent_slice_mut(&self.parent, self.start, self.end, py, f)
        }
    }

    fn checked_row(row: usize, rows: usize) -> PyResult<usize> {
        if row < rows {
            Ok(row)
        } else {
            Err(PyIndexError::new_err(format!(
                "row {row} out of range for matrix with {rows} rows"
            )))
        }
    }

    /// Validate a `row_start..row_end` x `col_start..col_end` rectangle against
    /// a matrix's current `rows` x `columns`, raising `IndexError` otherwise.
    fn checked_rect(
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
        rows: usize,
        columns: usize,
    ) -> PyResult<()> {
        if row_start <= row_end && row_end <= rows && col_start <= col_end && col_end <= columns {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!(
                "rectangle [{row_start}..{row_end}] x [{col_start}..{col_end}] out of range \
                 for matrix with {rows} rows and {columns} columns"
            )))
        }
    }

    /// Guard that a matrix has had its pivots initialized (via `row_reduce`)
    /// before a `compute_*` method reads them.
    ///
    /// Upstream's `compute_kernel`/`compute_image`/`compute_quasi_inverse(s)`
    /// funnel through `Matrix::find_first_row_in_block`, which slices
    /// `pivots[first_source_col..]`. `pivots` is an empty `Vec` until row
    /// reduction (`row_reduce` calls `initialize_pivots`, which resizes it to
    /// `columns`), so with a positive `first_source_col` the slice range would
    /// be out of bounds and panic across the PyO3 boundary.
    ///
    /// The only two reachable pivot states in the upstream API are "empty"
    /// (never initialized) or "length == columns" (initialized by
    /// `initialize_pivots`/`row_reduce`/`extend_column_dimension`); there is no
    /// partial-pivots state. We therefore use `pivots().len() == columns()` as
    /// the exact "initialized" invariant. We deliberately raise an explicit
    /// error rather than silently row-reducing, since auto-reduction would
    /// mutate the matrix and change observable state. Note this guards only the
    /// panic: an `initialize_pivots`-only matrix passes the check but is not a
    /// true rref, so callers are still responsible for having row reduced.
    fn ensure_pivots_initialized(pivots_len: usize, columns: usize) -> PyResult<()> {
        if pivots_len == columns {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "matrix must be row-reduced before compute_*",
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

    #[pymethods]
    impl PyFpVector {
        #[new]
        pub fn new(p: u32, len: usize) -> PyResult<Self> {
            Ok(Self(RustFpVector::new(valid_prime(p)?, len)))
        }

        #[staticmethod]
        pub fn new_with_capacity(p: u32, len: usize, capacity: usize) -> PyResult<Self> {
            Ok(Self(RustFpVector::new_with_capacity(
                valid_prime(p)?,
                len,
                capacity,
            )))
        }

        #[staticmethod]
        pub fn from_slice(p: u32, entries: Vec<u32>) -> PyResult<Self> {
            Ok(Self(RustFpVector::from_slice(valid_prime(p)?, &entries)))
        }

        #[staticmethod]
        pub fn from_bytes(p: u32, len: usize, data: &[u8]) -> PyResult<Self> {
            RustFpVector::from_bytes(valid_prime(p)?, len, &mut Cursor::new(data))
                .map(Self)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        pub fn entry(&self, index: usize) -> PyResult<u32> {
            Ok(self.0.entry(checked_index(index, self.0.len())?))
        }

        pub fn density(&self) -> f32 {
            self.0.density()
        }

        pub fn is_zero(&self) -> bool {
            self.0.is_zero()
        }

        pub fn first_nonzero(&self) -> Option<(usize, u32)> {
            self.0.first_nonzero()
        }

        pub fn slice(slf: PyRef<'_, Self>, start: usize, end: usize) -> PyResult<PyFpSlice> {
            checked_range(start, end, slf.0.len())?;
            let py = slf.py();
            Ok(PyFpSlice {
                parent: SliceParent::Vector(slf.into_pyobject(py)?.unbind()),
                start,
                end,
            })
        }

        pub fn slice_mut(slf: PyRef<'_, Self>, start: usize, end: usize) -> PyResult<PyFpSliceMut> {
            checked_range(start, end, slf.0.len())?;
            let py = slf.py();
            Ok(PyFpSliceMut {
                parent: SliceParent::Vector(slf.into_pyobject(py)?.unbind()),
                start,
                end,
            })
        }

        pub fn set_entry(&mut self, index: usize, value: u32) -> PyResult<()> {
            self.0.set_entry(checked_index(index, self.0.len())?, value);
            Ok(())
        }

        pub fn scale(&mut self, c: u32) {
            self.0.scale(c)
        }

        pub fn set_to_zero(&mut self) {
            self.0.set_to_zero()
        }

        pub fn add_basis_element(&mut self, index: usize, value: u32) -> PyResult<()> {
            self.0
                .add_basis_element(checked_index(index, self.0.len())?, value);
            Ok(())
        }

        pub fn extend_len(&mut self, len: usize) {
            self.0.extend_len(len)
        }

        pub fn set_scratch_vector_size(&mut self, len: usize) {
            self.0.set_scratch_vector_size(len)
        }

        pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let mut buffer = Vec::new();
            self.0
                .to_bytes(&mut buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyBytes::new(py, &buffer))
        }

        pub fn update_from_bytes(&mut self, data: &[u8]) -> PyResult<()> {
            self.0
                .update_from_bytes(&mut Cursor::new(data))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        pub fn __len__(&self) -> usize {
            self.0.len()
        }

        pub fn __getitem__(&self, index: isize) -> PyResult<u32> {
            Ok(self.0.entry(py_index(index, self.0.len())?))
        }

        pub fn __setitem__(&mut self, index: isize, value: u32) -> PyResult<()> {
            self.0.set_entry(py_index(index, self.0.len())?, value);
            Ok(())
        }

        pub fn __iter__(slf: PyRef<'_, Self>) -> PyFpVectorIterator {
            PyFpVectorIterator {
                entries: slf.0.iter().collect(),
                index: 0,
            }
        }

        pub fn __repr__(&self) -> String {
            format!("FpVector({}, {})", self.prime(), self.0)
        }
    }

    #[pymethods]
    impl PyFpSlice {
        pub fn prime(&self, py: Python<'_>) -> PyResult<u32> {
            self.with_slice(py, |s| s.prime().as_u32())
        }

        pub fn len(&self, py: Python<'_>) -> PyResult<usize> {
            self.with_slice(py, |s| s.len())
        }

        pub fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
            self.with_slice(py, |s| s.is_empty())
        }

        pub fn entry(&self, py: Python<'_>, index: usize) -> PyResult<u32> {
            let index = checked_index(index, self.span())?;
            self.with_slice(py, |s| s.entry(index))
        }

        pub fn iter(&self, py: Python<'_>) -> PyResult<PyFpVectorIterator> {
            let entries = self.with_slice(py, |s| s.iter().collect())?;
            Ok(PyFpVectorIterator { entries, index: 0 })
        }

        pub fn iter_nonzero(&self, py: Python<'_>) -> PyResult<Vec<(usize, u32)>> {
            self.with_slice(py, |s| s.iter_nonzero().collect())
        }

        pub fn is_zero(&self, py: Python<'_>) -> PyResult<bool> {
            self.with_slice(py, |s| s.is_zero())
        }

        pub fn first_nonzero(&self, py: Python<'_>) -> PyResult<Option<(usize, u32)>> {
            self.with_slice(py, |s| s.first_nonzero())
        }

        pub fn restrict(&self, py: Python<'_>, start: usize, end: usize) -> PyResult<Self> {
            checked_range(start, end, self.span())?;
            Ok(Self {
                parent: self.parent.clone_ref(py),
                start: self.start + start,
                end: self.start + end,
            })
        }

        pub fn to_owned(&self, py: Python<'_>) -> PyResult<PyFpVector> {
            Ok(PyFpVector(self.to_owned_checked(py)?))
        }

        pub fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
            self.len(py)
        }

        pub fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<u32> {
            let index = py_index(index, self.span())?;
            self.with_slice(py, |s| s.entry(index))
        }

        pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
            self.with_slice(py, |s| format!("FpSlice({}, {})", s.prime().as_u32(), s))
        }
    }

    #[pymethods]
    impl PyFpSliceMut {
        pub fn prime(&self, py: Python<'_>) -> PyResult<u32> {
            self.with_slice(py, |s| s.prime().as_u32())
        }

        pub fn len(&self, py: Python<'_>) -> PyResult<usize> {
            self.with_slice(py, |s| s.len())
        }

        pub fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
            self.with_slice(py, |s| s.is_empty())
        }

        pub fn set_entry(&self, py: Python<'_>, index: usize, value: u32) -> PyResult<()> {
            let index = checked_index(index, self.span())?;
            self.with_slice_mut(py, |mut s| s.set_entry(index, value))
        }

        pub fn set_to_zero(&self, py: Python<'_>) -> PyResult<()> {
            self.with_slice_mut(py, |mut s| s.set_to_zero())
        }

        pub fn scale(&self, py: Python<'_>, c: u32) -> PyResult<()> {
            self.with_slice_mut(py, |mut s| s.scale(c))
        }

        pub fn add(&self, py: Python<'_>, other: &PyFpSlice, c: u32) -> PyResult<()> {
            checked_equal_len(self.span(), other.span())?;
            let other_owned = other.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), other_owned.prime().as_u32())?;
                target.add(other_owned.as_slice(), c);
                Ok(())
            })?
        }

        pub fn add_offset(
            &self,
            py: Python<'_>,
            other: &PyFpSlice,
            c: u32,
            offset: usize,
        ) -> PyResult<()> {
            checked_equal_len(self.span(), other.span())?;
            checked_range(offset, self.span(), self.span())?;
            let other_owned = other.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), other_owned.prime().as_u32())?;
                target.add_offset(other_owned.as_slice(), c, offset);
                Ok(())
            })?
        }

        pub fn add_masked(
            &self,
            py: Python<'_>,
            other: &PyFpSlice,
            c: u32,
            mask: Vec<usize>,
        ) -> PyResult<()> {
            checked_equal_len(self.span(), mask.len())?;
            if let Some(&index) = mask.iter().find(|&&index| index >= other.span()) {
                return Err(PyIndexError::new_err(format!(
                    "mask index {index} out of range for vector of length {}",
                    other.span()
                )));
            }
            let other_owned = other.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), other_owned.prime().as_u32())?;
                target.add_masked(other_owned.as_slice(), c, &mask);
                Ok(())
            })?
        }

        pub fn add_unmasked(
            &self,
            py: Python<'_>,
            other: &PyFpSlice,
            c: u32,
            mask: Vec<usize>,
        ) -> PyResult<()> {
            if other.span() > mask.len() {
                return Err(PyValueError::new_err(format!(
                    "mask length {} shorter than source length {}",
                    mask.len(),
                    other.span()
                )));
            }
            if let Some(&index) = mask
                .iter()
                .take(other.span())
                .find(|&&index| index >= self.span())
            {
                return Err(PyIndexError::new_err(format!(
                    "mask index {index} out of range for vector of length {}",
                    self.span()
                )));
            }
            let other_owned = other.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), other_owned.prime().as_u32())?;
                target.add_unmasked(other_owned.as_slice(), c, &mask);
                Ok(())
            })?
        }

        pub fn assign(&self, py: Python<'_>, other: &PyFpSlice) -> PyResult<()> {
            checked_equal_len(self.span(), other.span())?;
            let other_owned = other.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), other_owned.prime().as_u32())?;
                target.assign(other_owned.as_slice());
                Ok(())
            })?
        }

        pub fn add_tensor(
            &self,
            py: Python<'_>,
            offset: usize,
            coeff: u32,
            left: &PyFpSlice,
            right: &PyFpSlice,
        ) -> PyResult<()> {
            let width = left
                .span()
                .checked_mul(right.span())
                .and_then(|width| offset.checked_add(width))
                .ok_or_else(|| PyIndexError::new_err("tensor range overflows usize"))?;
            checked_range(offset, width, self.span())?;
            let left_owned = left.to_owned_checked(py)?;
            let right_owned = right.to_owned_checked(py)?;
            self.with_slice_mut(py, |mut target| {
                checked_same_prime(target.prime().as_u32(), left_owned.prime().as_u32())?;
                checked_same_prime(target.prime().as_u32(), right_owned.prime().as_u32())?;
                target.add_tensor(offset, coeff, left_owned.as_slice(), right_owned.as_slice());
                Ok(())
            })?
        }

        pub fn add_basis_element(&self, py: Python<'_>, index: usize, value: u32) -> PyResult<()> {
            let index = checked_index(index, self.span())?;
            self.with_slice_mut(py, |mut s| s.add_basis_element(index, value))
        }

        pub fn as_slice(&self, py: Python<'_>) -> PyFpSlice {
            PyFpSlice {
                parent: self.parent.clone_ref(py),
                start: self.start,
                end: self.end,
            }
        }

        pub fn slice_mut(&self, py: Python<'_>, start: usize, end: usize) -> PyResult<Self> {
            checked_range(start, end, self.span())?;
            Ok(Self {
                parent: self.parent.clone_ref(py),
                start: self.start + start,
                end: self.start + end,
            })
        }

        pub fn to_owned(&self, py: Python<'_>) -> PyResult<PyFpVector> {
            Ok(PyFpVector(self.with_slice(py, |s| s.to_owned())?))
        }

        pub fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
            self.len(py)
        }

        pub fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<u32> {
            let index = py_index(index, self.span())?;
            self.with_slice(py, |s| s.entry(index))
        }

        pub fn __setitem__(&self, py: Python<'_>, index: isize, value: u32) -> PyResult<()> {
            let index = py_index(index, self.span())?;
            self.with_slice_mut(py, |mut s| s.set_entry(index, value))
        }

        pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
            self.with_slice(py, |s| format!("FpSliceMut({}, {})", s.prime().as_u32(), s))
        }
    }

    impl PyMatrixSliceMut {
        /// Number of rows spanned by the rectangle (cached; `with_slice_mut`
        /// revalidates against the parent before any data access).
        fn rows_span(&self) -> usize {
            self.row_end - self.row_start
        }

        /// Number of columns spanned by the rectangle (cached; see `rows_span`).
        fn cols_span(&self) -> usize {
            self.col_end - self.col_start
        }

        /// Run `f` on the reconstructed `MatrixSliceMut`, after revalidating the
        /// rectangle against the parent's current dimensions.
        fn with_slice_mut<R>(
            &self,
            py: Python<'_>,
            f: impl FnOnce(RustMatrixSliceMut<'_>) -> R,
        ) -> PyResult<R> {
            self.parent.with_matrix_mut(py, |m| {
                checked_rect(
                    self.row_start,
                    self.row_end,
                    self.col_start,
                    self.col_end,
                    m.rows(),
                    m.columns(),
                )?;
                Ok(f(m.slice_mut(
                    self.row_start,
                    self.row_end,
                    self.col_start,
                    self.col_end,
                )))
            })?
        }
    }

    #[pymethods]
    impl PyMatrixSliceMut {
        pub fn prime(&self, py: Python<'_>) -> PyResult<u32> {
            self.with_slice_mut(py, |s| s.prime().as_u32())
        }

        pub fn rows(&self, py: Python<'_>) -> PyResult<usize> {
            self.with_slice_mut(py, |s| s.rows())
        }

        pub fn columns(&self, py: Python<'_>) -> PyResult<usize> {
            self.with_slice_mut(py, |s| s.columns())
        }

        /// Return an immutable `FpSlice` over row `i` of the rectangle (the
        /// columns `col_start..col_end` of the parent's absolute row
        /// `row_start + i`). The handle revalidates against the parent on use.
        pub fn row(&self, py: Python<'_>, i: usize) -> PyResult<PyFpSlice> {
            let row = checked_row(i, self.rows_span())? + self.row_start;
            Ok(PyFpSlice {
                parent: SliceParent::MatrixRow {
                    matrix: self.parent.clone_ref(py),
                    row,
                },
                start: self.col_start,
                end: self.col_end,
            })
        }

        /// Return a mutable `FpSliceMut` over row `i` of the rectangle; mutating
        /// it writes through to the parent matrix.
        pub fn row_mut(&self, py: Python<'_>, i: usize) -> PyResult<PyFpSliceMut> {
            let row = checked_row(i, self.rows_span())? + self.row_start;
            Ok(PyFpSliceMut {
                parent: SliceParent::MatrixRow {
                    matrix: self.parent.clone_ref(py),
                    row,
                },
                start: self.col_start,
                end: self.col_end,
            })
        }

        /// Restrict the rectangle to rows `row_start..row_end` (relative to this
        /// view), returning a new `MatrixSliceMut` over the same columns and
        /// parent.
        pub fn row_slice(
            &self,
            py: Python<'_>,
            row_start: usize,
            row_end: usize,
        ) -> PyResult<Self> {
            checked_range(row_start, row_end, self.rows_span())?;
            Ok(Self {
                parent: self.parent.clone_ref(py),
                row_start: self.row_start + row_start,
                row_end: self.row_start + row_end,
                col_start: self.col_start,
                col_end: self.col_end,
            })
        }

        /// Return immutable `FpSlice` handles for every row of the rectangle.
        ///
        /// We materialize a list of row handles (rather than a lazy iterator)
        /// because PyO3 cannot store the borrowing Rust iterator alongside the
        /// owned parent. Each handle points into the parent and revalidates on
        /// use, mirroring the `Matrix`/`Subspace` choice of returning concrete
        /// per-row objects.
        pub fn iter(&self, py: Python<'_>) -> PyResult<Vec<PyFpSlice>> {
            (0..self.rows_span()).map(|i| self.row(py, i)).collect()
        }

        /// Return mutable `FpSliceMut` handles for every row of the rectangle.
        ///
        /// As with `iter`, this is an eager list of index-based row handles
        /// rather than a lazy borrowing iterator. Mutating any handle writes
        /// through to the parent matrix, so this is the safe PyO3 analogue of
        /// the upstream `iter_mut`.
        pub fn iter_mut(&self, py: Python<'_>) -> PyResult<Vec<PyFpSliceMut>> {
            (0..self.rows_span()).map(|i| self.row_mut(py, i)).collect()
        }

        /// Add an identity matrix into the rectangle. Requires a square
        /// rectangle (`rows == columns`), matching upstream's invariant;
        /// otherwise a `ValueError` is raised rather than panicking.
        pub fn add_identity(&self, py: Python<'_>) -> PyResult<()> {
            if self.rows_span() != self.cols_span() {
                return Err(PyValueError::new_err(format!(
                    "add_identity requires a square rectangle: {} rows but {} columns",
                    self.rows_span(),
                    self.cols_span()
                )));
            }
            self.with_slice_mut(py, |mut s| s.add_identity())
        }

        /// For each row, add the `mask[i]`th entry of the corresponding row of
        /// `other` into this rectangle. `other` must have the same prime and the
        /// same number of rows as the rectangle, `mask` must have length equal
        /// to the rectangle's column count, and every mask index must be a valid
        /// column of `other`.
        pub fn add_masked(
            &self,
            py: Python<'_>,
            other: &PyMatrix,
            mask: Vec<usize>,
        ) -> PyResult<()> {
            checked_same_prime(self.prime(py)?, other.0.prime().as_u32())?;
            checked_equal_len(self.rows_span(), other.0.rows())?;
            checked_equal_len(mask.len(), self.cols_span())?;
            let other_columns = other.0.columns();
            if let Some(&index) = mask.iter().find(|&&index| index >= other_columns) {
                return Err(PyIndexError::new_err(format!(
                    "mask index {index} out of range for matrix with {other_columns} columns"
                )));
            }
            // Clone `other` so the rectangle's `borrow_mut` cannot alias it even
            // if the same matrix object is passed as both parent and source.
            let other_matrix = other.0.clone();
            self.with_slice_mut(py, |mut s| s.add_masked(&other_matrix, &mask))
        }

        pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
            let (prime, rows, columns) =
                self.with_slice_mut(py, |s| (s.prime().as_u32(), s.rows(), s.columns()))?;
            Ok(format!("MatrixSliceMut({prime}, {rows}x{columns})"))
        }
    }

    #[pymethods]
    impl PyMatrix {
        #[new]
        pub fn new(p: u32, rows: usize, columns: usize) -> PyResult<Self> {
            Ok(Self(RustMatrix::new(valid_prime(p)?, rows, columns)))
        }

        #[staticmethod]
        pub fn from_rows(
            p: u32,
            rows: Vec<PyRef<'_, PyFpVector>>,
            columns: usize,
        ) -> PyResult<Self> {
            let p = valid_prime(p)?;
            for row in &rows {
                checked_same_prime(row.0.prime().as_u32(), p.as_u32())?;
                checked_equal_len(row.0.len(), columns)?;
            }
            let input = rows.iter().map(|row| row.0.clone()).collect();
            Ok(Self(RustMatrix::from_rows(p, input, columns)))
        }

        #[staticmethod]
        pub fn from_row(p: u32, row: PyRef<'_, PyFpVector>, columns: usize) -> PyResult<Self> {
            let p = valid_prime(p)?;
            checked_same_prime(row.0.prime().as_u32(), p.as_u32())?;
            checked_equal_len(row.0.len(), columns)?;
            Ok(Self(RustMatrix::from_row(p, row.0.clone(), columns)))
        }

        #[staticmethod]
        pub fn from_vec(p: u32, input: Vec<Vec<u32>>) -> PyResult<Self> {
            let p = valid_prime(p)?;
            if let Some(first) = input.first() {
                let columns = first.len();
                for row in &input {
                    checked_equal_len(row.len(), columns)?;
                }
            }
            Ok(Self(RustMatrix::from_vec(p, &input)))
        }

        #[staticmethod]
        pub fn identity(p: u32, dim: usize) -> PyResult<Self> {
            Ok(Self(RustMatrix::identity(valid_prime(p)?, dim)))
        }

        #[staticmethod]
        pub fn augmented_from_vec(p: u32, input: Vec<Vec<u32>>) -> PyResult<(usize, Self)> {
            let p = valid_prime(p)?;
            if input.is_empty() {
                return Err(PyValueError::new_err(
                    "augmented_from_vec requires at least one row",
                ));
            }
            let columns = input[0].len();
            for row in &input {
                checked_equal_len(row.len(), columns)?;
            }
            let (first_source_column, matrix) = RustMatrix::augmented_from_vec(p, &input);
            Ok((first_source_column, Self(matrix)))
        }

        #[staticmethod]
        pub fn from_bytes(p: u32, rows: usize, columns: usize, data: &[u8]) -> PyResult<Self> {
            RustMatrix::from_bytes(valid_prime(p)?, rows, columns, &mut Cursor::new(data))
                .map(Self)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn rows(&self) -> usize {
            self.0.rows()
        }

        pub fn columns(&self) -> usize {
            self.0.columns()
        }

        pub fn pivots(&self) -> Vec<isize> {
            self.0.pivots().to_vec()
        }

        pub fn is_zero(&self) -> bool {
            self.0.is_zero()
        }

        pub fn to_vec(&self) -> Vec<Vec<u32>> {
            self.0.to_vec()
        }

        pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let mut buffer = Vec::new();
            self.0
                .to_bytes(&mut buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyBytes::new(py, &buffer))
        }

        pub fn row(slf: PyRef<'_, Self>, row: usize) -> PyResult<PyFpSlice> {
            checked_row(row, slf.0.rows())?;
            let end = slf.0.columns();
            let py = slf.py();
            Ok(PyFpSlice {
                parent: SliceParent::MatrixRow {
                    matrix: MatrixParent::Matrix(slf.into_pyobject(py)?.unbind()),
                    row,
                },
                start: 0,
                end,
            })
        }

        pub fn row_mut(slf: PyRef<'_, Self>, row: usize) -> PyResult<PyFpSliceMut> {
            checked_row(row, slf.0.rows())?;
            let end = slf.0.columns();
            let py = slf.py();
            Ok(PyFpSliceMut {
                parent: SliceParent::MatrixRow {
                    matrix: MatrixParent::Matrix(slf.into_pyobject(py)?.unbind()),
                    row,
                },
                start: 0,
                end,
            })
        }

        pub fn set_to_zero(&mut self) {
            self.0.set_to_zero()
        }

        pub fn assign(&mut self, other: &Self) -> PyResult<()> {
            checked_same_prime(self.0.prime().as_u32(), other.0.prime().as_u32())?;
            checked_equal_len(self.0.rows(), other.0.rows())?;
            checked_equal_len(self.0.columns(), other.0.columns())?;
            self.0.assign(&other.0);
            Ok(())
        }

        pub fn swap_rows(&mut self, i: usize, j: usize) -> PyResult<()> {
            checked_row(i, self.0.rows())?;
            checked_row(j, self.0.rows())?;
            self.0.swap_rows(i, j);
            Ok(())
        }

        pub fn safe_row_op(&mut self, target: usize, source: usize, c: u32) -> PyResult<()> {
            checked_row(target, self.0.rows())?;
            checked_row(source, self.0.rows())?;
            if target == source {
                return Err(PyValueError::new_err(
                    "target and source rows must be distinct",
                ));
            }
            self.0.safe_row_op(target, source, c);
            Ok(())
        }

        pub fn initialize_pivots(&mut self) {
            self.0.initialize_pivots()
        }

        pub fn extend_column_dimension(&mut self, columns: usize) {
            self.0.extend_column_dimension(columns)
        }

        pub fn extend_column_capacity(&mut self, columns: usize) {
            self.0.extend_column_capacity(columns)
        }

        pub fn add_row(slf: PyRef<'_, Self>) -> PyResult<PyFpSliceMut> {
            let py = slf.py();
            let parent = slf.into_pyobject(py)?.unbind();
            let (row, end) = {
                let mut matrix = parent.try_borrow_mut(py).map_err(borrow_error)?;
                matrix.0.add_row();
                (matrix.0.rows() - 1, matrix.0.columns())
            };
            Ok(PyFpSliceMut {
                parent: SliceParent::MatrixRow {
                    matrix: MatrixParent::Matrix(parent),
                    row,
                },
                start: 0,
                end,
            })
        }

        /// Return a mutable rectangular view over rows `row_start..row_end` and
        /// columns `col_start..col_end`. The returned `MatrixSliceMut` holds
        /// this matrix and revalidates the rectangle against the matrix's
        /// current dimensions on every call.
        pub fn slice_mut(
            slf: PyRef<'_, Self>,
            row_start: usize,
            row_end: usize,
            col_start: usize,
            col_end: usize,
        ) -> PyResult<PyMatrixSliceMut> {
            checked_rect(
                row_start,
                row_end,
                col_start,
                col_end,
                slf.0.rows(),
                slf.0.columns(),
            )?;
            let py = slf.py();
            Ok(PyMatrixSliceMut {
                parent: MatrixParent::Matrix(slf.into_pyobject(py)?.unbind()),
                row_start,
                row_end,
                col_start,
                col_end,
            })
        }

        pub fn trim(&mut self, row_start: usize, row_end: usize, col_start: usize) -> PyResult<()> {
            checked_range(row_start, row_end, self.0.rows())?;
            if col_start > self.0.columns() {
                return Err(PyIndexError::new_err(format!(
                    "column {col_start} out of range for matrix with {} columns",
                    self.0.columns()
                )));
            }
            self.0.trim(row_start, row_end, col_start, false);
            Ok(())
        }

        pub fn rotate_down(&mut self, start: usize, end: usize, shift: usize) -> PyResult<()> {
            checked_range(start, end, self.0.rows())?;
            if shift > end - start {
                return Err(PyValueError::new_err(format!(
                    "shift {shift} exceeds range length {}",
                    end - start
                )));
            }
            self.0.rotate_down(start..end, shift);
            Ok(())
        }

        pub fn row_reduce(&mut self) -> usize {
            self.0.row_reduce()
        }

        /// Compute the quasi-inverse of a row-reduced augmented matrix `[A|0|I]`.
        ///
        /// `last_target_col` is the last column of `A`, and `first_source_col`
        /// is the first column of `I` (typically the padded column count
        /// returned by `augmented_from_vec`). The matrix is expected to already
        /// be row reduced.
        pub fn compute_quasi_inverse(
            &self,
            last_target_col: usize,
            first_source_col: usize,
        ) -> PyResult<PyQuasiInverse> {
            let columns = self.0.columns();
            ensure_pivots_initialized(self.0.pivots().len(), columns)?;
            if last_target_col > columns {
                return Err(PyIndexError::new_err(format!(
                    "last_target_col {last_target_col} out of range for matrix with {columns} columns"
                )));
            }
            if first_source_col > columns {
                return Err(PyIndexError::new_err(format!(
                    "first_source_col {first_source_col} out of range for matrix with {columns} columns"
                )));
            }
            Ok(PyQuasiInverse(
                self.0
                    .compute_quasi_inverse(last_target_col, first_source_col),
            ))
        }

        pub fn __len__(&self) -> usize {
            self.0.rows()
        }

        pub fn __getitem__(slf: PyRef<'_, Self>, row: usize) -> PyResult<PyFpSlice> {
            Self::row(slf, row)
        }

        pub fn __repr__(&self) -> String {
            format!("Matrix({}, {})", self.prime(), self.0)
        }
    }

    impl PySubspace {
        /// Validate that `vector` matches this subspace's prime and ambient
        /// dimension, returning an error otherwise.
        fn check_compatible(&self, vector: &RustFpVector) -> PyResult<()> {
            checked_same_prime(self.0.prime().as_u32(), vector.prime().as_u32())?;
            checked_equal_len(vector.len(), self.0.ambient_dimension())?;
            Ok(())
        }

        /// Validate that `other` matches this subspace's prime and ambient
        /// dimension, returning an error otherwise.
        fn check_compatible_space(&self, other: &RustSubspace) -> PyResult<()> {
            checked_same_prime(self.0.prime().as_u32(), other.prime().as_u32())?;
            checked_equal_len(self.0.ambient_dimension(), other.ambient_dimension())?;
            Ok(())
        }
    }

    #[pymethods]
    impl PySubspace {
        #[new]
        pub fn new(p: u32, dim: usize) -> PyResult<Self> {
            Ok(Self(RustSubspace::new(valid_prime(p)?, dim)))
        }

        #[staticmethod]
        pub fn from_matrix(matrix: &PyMatrix) -> Self {
            Self(RustSubspace::from_matrix(matrix.0.clone()))
        }

        #[staticmethod]
        pub fn entire_space(p: u32, dim: usize) -> PyResult<Self> {
            Ok(Self(RustSubspace::entire_space(valid_prime(p)?, dim)))
        }

        #[staticmethod]
        pub fn from_bytes(p: u32, data: &[u8]) -> PyResult<Self> {
            RustSubspace::from_bytes(valid_prime(p)?, &mut Cursor::new(data))
                .map(Self)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn dimension(&self) -> usize {
            self.0.dimension()
        }

        pub fn ambient_dimension(&self) -> usize {
            self.0.ambient_dimension()
        }

        pub fn contains(&self, vector: &PyFpVector) -> PyResult<bool> {
            self.check_compatible(&vector.0)?;
            Ok(self.0.contains(vector.0.as_slice()))
        }

        pub fn contains_space(&self, other: &Self) -> PyResult<bool> {
            self.check_compatible_space(&other.0)?;
            Ok(self.0.contains_space(&other.0))
        }

        pub fn add_vector(&mut self, vector: &PyFpVector) -> PyResult<usize> {
            self.check_compatible(&vector.0)?;
            Ok(self.0.add_vector(vector.0.as_slice()))
        }

        /// Reduce `vector` in place against this subspace, projecting it onto a
        /// complement of the subspace.
        pub fn reduce(&self, vector: &mut PyFpVector) -> PyResult<()> {
            self.check_compatible(&vector.0)?;
            self.0.reduce(vector.0.as_slice_mut());
            Ok(())
        }

        pub fn sum(&self, other: &Self) -> PyResult<Self> {
            checked_same_prime(self.0.prime().as_u32(), other.0.prime().as_u32())?;
            checked_equal_len(self.0.ambient_dimension(), other.0.ambient_dimension())?;
            Ok(Self(self.0.sum(&other.0)))
        }

        /// Return the basis of the subspace as a list of owned `FpVector`s.
        pub fn iter(&self) -> Vec<PyFpVector> {
            self.0
                .iter()
                .map(|row| PyFpVector(row.to_owned()))
                .collect()
        }

        /// Return a lazy iterator over every vector in the subspace.
        pub fn iter_all_vectors(&self) -> PySubspaceVectorIterator {
            let p = u128::from(self.0.prime().as_u32());
            let dim = self.0.dimension() as u32;
            let total = p.checked_pow(dim).unwrap_or(u128::MAX);
            PySubspaceVectorIterator {
                subspace: self.0.clone(),
                index: 0,
                total,
            }
        }

        pub fn set_to_zero(&mut self) {
            self.0.set_to_zero()
        }

        pub fn set_to_entire(&mut self) {
            self.0.set_to_entire()
        }

        pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let mut buffer = Vec::new();
            self.0
                .to_bytes(&mut buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyBytes::new(py, &buffer))
        }

        pub fn __len__(&self) -> usize {
            self.0.dimension()
        }

        pub fn __contains__(&self, vector: &PyFpVector) -> PyResult<bool> {
            self.contains(vector)
        }

        pub fn __repr__(&self) -> String {
            format!(
                "Subspace({}, dim={}, ambient={})",
                self.prime(),
                self.0.dimension(),
                self.0.ambient_dimension()
            )
        }
    }

    /// Extract an owned copy of a vector-like argument (`FpVector` or
    /// `FpSlice`) for use as an immutable input.
    fn extract_input_owned(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<RustFpVector> {
        if let Ok(vector) = obj.extract::<PyRef<'_, PyFpVector>>() {
            Ok(vector.0.clone())
        } else if let Ok(slice) = obj.extract::<PyRef<'_, PyFpSlice>>() {
            slice.to_owned_checked(py)
        } else {
            Err(PyValueError::new_err("expected an FpVector or FpSlice"))
        }
    }

    /// Run `f` on the mutable slice backing a vector-like argument
    /// (`FpVector` or `FpSliceMut`), used as an output target.
    fn with_target_slice_mut<R>(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        f: impl FnOnce(RustFpSliceMut<'_>) -> PyResult<R>,
    ) -> PyResult<R> {
        if let Ok(mut vector) = obj.extract::<PyRefMut<'_, PyFpVector>>() {
            f(vector.0.as_slice_mut())
        } else if let Ok(slice) = obj.extract::<PyRef<'_, PyFpSliceMut>>() {
            slice.with_slice_mut(py, f)?
        } else {
            Err(PyValueError::new_err("expected an FpVector or FpSliceMut"))
        }
    }

    #[pymethods]
    impl PyQuasiInverse {
        /// Construct a `QuasiInverse` from an optional `image` (pivot list) and a
        /// `preimage` matrix.
        ///
        /// # Invariant enforced
        ///
        /// `apply` (and `stream_quasi_inverse`) walk `image` and, for every
        /// non-negative pivot entry, consume one row of `preimage` (the rows are
        /// addressed by a running counter that increments once per non-negative
        /// pivot). Upstream `QuasiInverse::new` performs no validation, so without
        /// the checks below a Python caller could supply an `image` whose count of
        /// non-negative pivots exceeds `preimage.rows()`, causing `apply` to index
        /// `preimage.row(row)` out of bounds and panic across the PyO3 boundary.
        ///
        /// We therefore require, when `image` is `Some`:
        ///  * the number of non-negative pivot entries is `<= preimage.rows()`
        ///    (this is the exact invariant that makes `apply` safe), and
        ///  * every non-negative pivot is a valid `preimage` row index, i.e. in
        ///    `0..preimage.rows()` (pivots are row indices into `preimage`).
        ///
        /// When `image` is `None` the image is the standard basis (identity) and
        /// no pivot validation is needed; that path is always safe.
        #[new]
        #[pyo3(signature = (image, preimage))]
        pub fn new(image: Option<Vec<isize>>, preimage: &PyMatrix) -> PyResult<Self> {
            if let Some(pivots) = image.as_ref() {
                let rows = preimage.0.rows();
                let mut nonneg = 0usize;
                for &p in pivots {
                    if p >= 0 {
                        nonneg += 1;
                        if (p as usize) >= rows {
                            return Err(PyValueError::new_err(format!(
                                "inconsistent QuasiInverse: pivot {p} is out of range for a \
                                 preimage with {rows} rows"
                            )));
                        }
                    }
                }
                if nonneg > rows {
                    return Err(PyValueError::new_err(format!(
                        "inconsistent QuasiInverse: image has {nonneg} non-negative pivots but \
                         preimage only has {rows} rows"
                    )));
                }
            }
            Ok(Self(RustQuasiInverse::new(image, preimage.0.clone())))
        }

        /// Deserialize a `QuasiInverse` from bytes produced by [`Self::to_bytes`].
        ///
        /// Note on `image = None`: serialization does not preserve a `None` image.
        /// [`Self::to_bytes`] writes a `None` image as an explicit identity pivot
        /// list `[0, 1, 2, ...]` (matching upstream), so a quasi-inverse built with
        /// `image=None` round-trips to one whose `pivots()` are `Some([0, 1, ...])`
        /// rather than `None`. This is intended upstream behavior and is not changed
        /// here.
        #[staticmethod]
        pub fn from_bytes(p: u32, data: &[u8]) -> PyResult<Self> {
            RustQuasiInverse::from_bytes(valid_prime(p)?, &mut Cursor::new(data))
                .map(Self)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn image_dimension(&self) -> usize {
            self.0.image_dimension()
        }

        pub fn source_dimension(&self) -> usize {
            self.0.source_dimension()
        }

        pub fn target_dimension(&self) -> usize {
            self.0.target_dimension()
        }

        pub fn preimage(&self) -> PyMatrix {
            PyMatrix(self.0.preimage().clone())
        }

        pub fn pivots(&self) -> Option<Vec<isize>> {
            self.0.pivots().map(<[isize]>::to_vec)
        }

        /// Apply the quasi-inverse to `input` and add `coeff` times the result
        /// to `target`.
        ///
        /// `input` is a vector in the target space (length `target_dimension`)
        /// and `target` receives the result in the source space (length
        /// `source_dimension`). Both accept either an `FpVector` or the
        /// corresponding slice handle.
        pub fn apply(
            &self,
            py: Python<'_>,
            target: &Bound<'_, PyAny>,
            coeff: u32,
            input: &Bound<'_, PyAny>,
        ) -> PyResult<()> {
            let input_owned = extract_input_owned(py, input)?;
            checked_same_prime(self.0.prime().as_u32(), input_owned.prime().as_u32())?;
            checked_equal_len(input_owned.len(), self.0.target_dimension())?;
            with_target_slice_mut(py, target, |target_slice| {
                checked_same_prime(
                    self.0.prime().as_u32(),
                    target_slice.as_slice().prime().as_u32(),
                )?;
                checked_equal_len(target_slice.as_slice().len(), self.0.source_dimension())?;
                // Reduce `coeff` mod p before calling upstream. Upstream computes
                // `(coeff * c) % p`; with `c < p` and an unreduced `coeff` the
                // product `coeff * c` can overflow u32 (debug panic / wrong result
                // in release). Reducing first is mathematically equivalent since
                // `(coeff % p) * c % p == coeff * c % p`.
                let coeff = coeff % self.0.prime().as_u32();
                self.0.apply(target_slice, coeff, input_owned.as_slice());
                Ok(())
            })
        }

        /// Serialize the quasi-inverse to bytes.
        ///
        /// Note: a `None` image (identity) is serialized as an explicit identity
        /// pivot list `[0, 1, 2, ...]` (matching upstream), so it does not survive
        /// a round-trip as `None`; see [`Self::from_bytes`].
        pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            let mut buffer = Vec::new();
            self.0
                .to_bytes(&mut buffer)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyBytes::new(py, &buffer))
        }

        pub fn __repr__(&self) -> String {
            format!(
                "QuasiInverse({}, image_dim={}, source_dim={}, target_dim={})",
                self.prime(),
                self.0.image_dimension(),
                self.0.source_dimension(),
                self.0.target_dimension()
            )
        }
    }

    impl PySubquotient {
        /// Validate that `vector` matches this subquotient's prime and ambient
        /// dimension, returning an error otherwise.
        fn check_compatible(&self, vector: &RustFpVector) -> PyResult<()> {
            checked_same_prime(self.0.prime().as_u32(), vector.prime().as_u32())?;
            checked_equal_len(vector.len(), self.0.ambient_dimension())?;
            Ok(())
        }

        /// Validate that `space` matches this subquotient's prime and ambient
        /// dimension, returning an error otherwise.
        fn check_compatible_space(&self, space: &RustSubspace) -> PyResult<()> {
            checked_same_prime(self.0.prime().as_u32(), space.prime().as_u32())?;
            checked_equal_len(self.0.ambient_dimension(), space.ambient_dimension())?;
            Ok(())
        }
    }

    #[pymethods]
    impl PySubquotient {
        /// Create a new subquotient of an ambient space of dimension `dim`,
        /// defaulting to the zero subspace.
        #[new]
        pub fn new(p: u32, dim: usize) -> PyResult<Self> {
            Ok(Self(RustSubquotient::new(valid_prime(p)?, dim)))
        }

        /// Create a new subquotient of an ambient space of dimension `dim`,
        /// where the subspace is the full space and the quotient is trivial.
        #[staticmethod]
        pub fn new_full(p: u32, dim: usize) -> PyResult<Self> {
            Ok(Self(RustSubquotient::new_full(valid_prime(p)?, dim)))
        }

        /// Construct the subquotient `(sub + quotient) / quotient` from a chain
        /// of subspaces. The two subspaces must share a prime and ambient
        /// dimension.
        #[staticmethod]
        pub fn from_parts(sub: &PySubspace, quotient: &PySubspace) -> PyResult<Self> {
            checked_same_prime(sub.0.prime().as_u32(), quotient.0.prime().as_u32())?;
            checked_equal_len(sub.0.ambient_dimension(), quotient.0.ambient_dimension())?;
            Ok(Self(RustSubquotient::from_parts(
                sub.0.clone(),
                quotient.0.clone(),
            )))
        }

        pub fn prime(&self) -> u32 {
            self.0.prime().as_u32()
        }

        pub fn dimension(&self) -> usize {
            self.0.dimension()
        }

        pub fn ambient_dimension(&self) -> usize {
            self.0.ambient_dimension()
        }

        pub fn quotient_dimension(&self) -> usize {
            self.0.quotient_dimension()
        }

        pub fn subspace_dimension(&self) -> usize {
            self.0.subspace_dimension()
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        /// The quotient (zero) subspace of the subquotient, returned as an owned
        /// `Subspace`.
        pub fn zeros(&self) -> PySubspace {
            PySubspace(self.0.zeros().clone())
        }

        /// The generators of the subquotient, returned as a list of owned
        /// `FpVector`s. Mirrors the choice made for `Subspace.iter`: the
        /// upstream iterator borrows the subquotient, so we materialize owned
        /// vectors rather than expose borrowed slice handles.
        pub fn gens(&self) -> Vec<PyFpVector> {
            self.0
                .gens()
                .map(|row| PyFpVector(row.to_owned()))
                .collect()
        }

        /// The generators of the subspace part of the subquotient, returned as
        /// a list of owned `FpVector`s (see `gens` for the ownership choice).
        pub fn subspace_gens(&self) -> Vec<PyFpVector> {
            self.0
                .subspace_gens()
                .map(|row| PyFpVector(row.to_owned()))
                .collect()
        }

        /// The pivot columns of the complement to the subspace.
        pub fn complement_pivots(&self) -> Vec<usize> {
            self.0.complement_pivots().collect()
        }

        /// The pivot table of the quotient subspace.
        pub fn quotient_pivots(&self) -> Vec<isize> {
            self.0.quotient_pivots().to_vec()
        }

        /// Reduce `vector` in place: project it onto a complement of the
        /// quotient and express it relative to the generators. Returns the list
        /// of coefficients with respect to the generators. After the call,
        /// `vector` holds the residual; a nonzero residual means the vector was
        /// not in the subspace.
        pub fn reduce(&self, vector: &mut PyFpVector) -> PyResult<Vec<u32>> {
            self.check_compatible(&vector.0)?;
            Ok(self.0.reduce(vector.0.as_slice_mut()))
        }

        /// Project `vector` in place onto the complement of the quotient part.
        pub fn reduce_by_quotient(&self, vector: &mut PyFpVector) -> PyResult<()> {
            self.check_compatible(&vector.0)?;
            self.0.reduce_by_quotient(vector.0.as_slice_mut());
            Ok(())
        }

        /// Add `vector` to the quotient part of the subquotient.
        pub fn quotient(&mut self, vector: &PyFpVector) -> PyResult<()> {
            self.check_compatible(&vector.0)?;
            self.0.quotient(vector.0.as_slice());
            Ok(())
        }

        /// Add `vector` as a generator of the subquotient.
        pub fn add_gen(&mut self, vector: &PyFpVector) -> PyResult<()> {
            self.check_compatible(&vector.0)?;
            self.0.add_gen(vector.0.as_slice());
            Ok(())
        }

        /// Remove all generators, leaving the quotient part untouched.
        pub fn clear_gens(&mut self) {
            self.0.clear_gens()
        }

        /// Set the subquotient to be the full ambient space quotiented by zero.
        pub fn set_to_full(&mut self) {
            self.0.set_to_full()
        }

        /// Apply `matrix` to each generator of `source`, then reduce the image
        /// in `target`, returning the coefficient lists. `matrix` must map the
        /// ambient space of `source` into the ambient space of `target`.
        #[staticmethod]
        pub fn reduce_matrix(
            matrix: &PyMatrix,
            source: &Self,
            target: &Self,
        ) -> PyResult<Vec<Vec<u32>>> {
            checked_same_prime(source.0.prime().as_u32(), target.0.prime().as_u32())?;
            checked_same_prime(source.0.prime().as_u32(), matrix.0.prime().as_u32())?;
            checked_equal_len(matrix.0.rows(), source.0.ambient_dimension())?;
            checked_equal_len(matrix.0.columns(), target.0.ambient_dimension())?;
            Ok(RustSubquotient::reduce_matrix(
                &matrix.0, &source.0, &target.0,
            ))
        }

        pub fn __len__(&self) -> usize {
            self.0.dimension()
        }

        pub fn __repr__(&self) -> String {
            format!(
                "Subquotient({}, dim={}, ambient={})",
                self.prime(),
                self.0.dimension(),
                self.0.ambient_dimension()
            )
        }
    }

    impl PyAffineSubspace {
        /// Validate that `other` matches this affine subspace's prime and
        /// ambient dimension, returning an error otherwise.
        fn check_compatible_space(&self, other: &Self) -> PyResult<()> {
            checked_same_prime(self.prime(), other.prime())?;
            checked_equal_len(self.ambient_dimension(), other.ambient_dimension())?;
            Ok(())
        }
    }

    #[pymethods]
    impl PyAffineSubspace {
        /// Construct an affine subspace `offset + linear_part`.
        ///
        /// Upstream `AffineSubspace::new` `assert_eq!`s that the offset length
        /// matches the linear part's ambient dimension and reduces the offset
        /// against the linear part (which requires a shared prime), so we
        /// pre-check both here to raise `ValueError` instead of panicking.
        #[new]
        pub fn new(offset: &PyFpVector, linear_part: &PySubspace) -> PyResult<Self> {
            checked_same_prime(offset.0.prime().as_u32(), linear_part.0.prime().as_u32())?;
            checked_equal_len(offset.0.len(), linear_part.0.ambient_dimension())?;
            Ok(Self(RustAffineSubspace::new(
                offset.0.clone(),
                linear_part.0.clone(),
            )))
        }

        pub fn prime(&self) -> u32 {
            self.0.linear_part().prime().as_u32()
        }

        pub fn ambient_dimension(&self) -> usize {
            self.0.linear_part().ambient_dimension()
        }

        pub fn dimension(&self) -> usize {
            self.0.linear_part().dimension()
        }

        /// Return an owned copy of the (reduced) offset vector.
        ///
        /// We return an owned `FpVector` rather than a borrowed view, matching
        /// the owned-return precedent used by `Subspace`/`Subquotient`. The
        /// offset stored upstream is the input reduced against the linear part,
        /// so it may differ from the vector passed to `new`.
        pub fn offset(&self) -> PyFpVector {
            PyFpVector(self.0.offset().clone())
        }

        /// Return an owned copy (clone) of the linear part `Subspace`,
        /// consistent with the owned-return precedent.
        pub fn linear_part(&self) -> PySubspace {
            PySubspace(self.0.linear_part().clone())
        }

        /// Test whether `vector` (an `FpVector` or `FpSlice`) lies in this
        /// affine subspace.
        pub fn contains(&self, py: Python<'_>, vector: &Bound<'_, PyAny>) -> PyResult<bool> {
            let vector = extract_input_owned(py, vector)?;
            checked_same_prime(self.prime(), vector.prime().as_u32())?;
            checked_equal_len(vector.len(), self.ambient_dimension())?;
            Ok(self.0.contains(vector.as_slice()))
        }

        pub fn contains_space(&self, other: &Self) -> PyResult<bool> {
            self.check_compatible_space(other)?;
            Ok(self.0.contains_space(&other.0))
        }

        /// Return the affine subspace spanned by the union of `self` and
        /// `other`: the sum of the linear parts translated by the sum of the
        /// offsets.
        pub fn sum(&self, other: &Self) -> PyResult<Self> {
            self.check_compatible_space(other)?;
            Ok(Self(self.0.sum(&other.0)))
        }

        pub fn __contains__(&self, py: Python<'_>, vector: &Bound<'_, PyAny>) -> PyResult<bool> {
            self.contains(py, vector)
        }

        pub fn __repr__(&self) -> String {
            format!("AffineSubspace({})", self.0)
        }
    }

    /// Validate that `seg` is a segment index in `0..n`.
    fn checked_segment(seg: usize, n: usize) -> PyResult<()> {
        if seg < n {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!(
                "segment {seg} out of range for {n} segments"
            )))
        }
    }

    /// Validate a `[start, end]` segment-index range against an augmented
    /// matrix and return the width (column count) of the spanned rectangle.
    fn segment_cols<const N: usize>(
        m: &RustAugmentedMatrix<N>,
        start: usize,
        end: usize,
    ) -> PyResult<usize> {
        checked_segment(start, N)?;
        checked_segment(end, N)?;
        let lo = m.start[start];
        let hi = m.end[end];
        if lo > hi {
            return Err(PyValueError::new_err(format!(
                "segment range [{start}, {end}] is empty or inverted"
            )));
        }
        Ok(hi - lo)
    }

    /// `AugmentedMatrix<N>` is a const-generic type, and PyO3 cannot expose a
    /// generic `#[pyclass]`. We therefore bind the two concrete arities used in
    /// the codebase (`N = 2` and `N = 3`) as separate classes `AugmentedMatrix2`
    /// and `AugmentedMatrix3`. To avoid duplicating the shared glue, this
    /// `macro_rules!` macro generates each class from a single definition; the
    /// per-arity methods are spliced in through the `$extra` token block, and
    /// `$variant` names the matching `MatrixParent` enum case so that the
    /// shared `segment`/`row_segment_mut` methods can build borrowed views that
    /// revalidate against this concrete arity. Each
    /// generated class still goes through `#[pyclass]` / `#[pymethods]`, so this
    /// is not hand-desugared PyO3 registration. However, the `#[pymodule]`
    /// proc-macro cannot see through a `macro_rules!` expansion to auto-collect
    /// the classes, so they are registered explicitly with `add_class` in
    /// `#[pymodule_init]`.
    macro_rules! augmented_matrix_pyclass {
        ($name:ident, $pyname:literal, $n:literal, $variant:ident, { $($extra:tt)* }) => {
            #[pyclass(name = $pyname)]
            struct $name(RustAugmentedMatrix<$n>);

            #[pymethods]
            impl $name {
                /// Construct an `rows x sum(columns)` augmented matrix whose
                /// column blocks have the given widths. `columns` must contain
                /// exactly `N` segment widths.
                #[new]
                fn new(p: u32, rows: usize, columns: Vec<usize>) -> PyResult<Self> {
                    let len = columns.len();
                    let cols: [usize; $n] = columns.try_into().map_err(|_| {
                        PyValueError::new_err(format!(
                            "expected {} segment widths, got {len}",
                            $n
                        ))
                    })?;
                    Ok(Self(RustAugmentedMatrix::<$n>::new(
                        valid_prime(p)?,
                        rows,
                        cols,
                    )))
                }

                fn prime(&self) -> u32 {
                    self.0.prime().as_u32()
                }

                fn rows(&self) -> usize {
                    self.0.rows()
                }

                fn columns(&self) -> usize {
                    self.0.columns()
                }

                /// Number of column segments (`N`).
                fn segments(&self) -> usize {
                    $n
                }

                /// The starting column index of each segment.
                fn segment_starts(&self) -> Vec<usize> {
                    self.0.start.to_vec()
                }

                /// The (exclusive) ending column index of each segment.
                fn segment_ends(&self) -> Vec<usize> {
                    self.0.end.to_vec()
                }

                fn pivots(&self) -> Vec<isize> {
                    self.0.pivots().to_vec()
                }

                fn is_zero(&self) -> bool {
                    self.0.is_zero()
                }

                fn to_vec(&self) -> Vec<Vec<u32>> {
                    self.0.to_vec()
                }

                fn row_reduce(&mut self) -> usize {
                    self.0.row_reduce()
                }

                /// Add an identity matrix into the rectangular segment spanning
                /// segment indices `start..=end`. The segment must be square
                /// (its row count equals its column width), matching upstream's
                /// `MatrixSliceMut::add_identity` invariant; otherwise a
                /// `ValueError` is raised rather than panicking.
                fn add_identity(&mut self, start: usize, end: usize) -> PyResult<()> {
                    let cols = segment_cols(&self.0, start, end)?;
                    if self.0.rows() != cols {
                        return Err(PyValueError::new_err(format!(
                            "add_identity requires a square segment: matrix has {} rows but \
                             segment [{start}, {end}] has {cols} columns",
                            self.0.rows()
                        )));
                    }
                    self.0.segment(start, end).add_identity();
                    Ok(())
                }

                /// Return an owned copy of row `i` restricted to the columns of
                /// the segment range `start..=end`.
                ///
                /// Upstream `row_segment` returns a borrowed `FpSlice`. We copy
                /// into an owned `FpVector` instead, matching the owned-return
                /// precedent used elsewhere (e.g. `Subspace.iter`); the mutable
                /// `row_segment_mut` and rectangle-returning `segment` provide
                /// the write-through borrowed views (see below).
                fn row_segment(
                    &self,
                    i: usize,
                    start: usize,
                    end: usize,
                ) -> PyResult<PyFpVector> {
                    checked_row(i, self.0.rows())?;
                    segment_cols(&self.0, start, end)?;
                    Ok(PyFpVector(self.0.row_segment(i, start, end).to_owned()))
                }

                /// Return a mutable rectangular view spanning all rows and the
                /// columns of segment range `start..=end`, as a
                /// `MatrixSliceMut` over the inner matrix. Mutations write
                /// through to this augmented matrix. The handle revalidates the
                /// rectangle against the inner matrix's current dimensions on
                /// every call.
                fn segment(
                    slf: PyRef<'_, Self>,
                    start: usize,
                    end: usize,
                ) -> PyResult<PyMatrixSliceMut> {
                    segment_cols(&slf.0, start, end)?;
                    let row_end = slf.0.rows();
                    let col_start = slf.0.start[start];
                    let col_end = slf.0.end[end];
                    let py = slf.py();
                    Ok(PyMatrixSliceMut {
                        parent: MatrixParent::$variant(slf.into_pyobject(py)?.unbind()),
                        row_start: 0,
                        row_end,
                        col_start,
                        col_end,
                    })
                }

                /// Return a mutable `FpSliceMut` over row `i` restricted to the
                /// columns of segment range `start..=end`. Mutations write
                /// through to this augmented matrix. Now thin glue over the
                /// unified slice-handle machinery (it reuses the matrix-row
                /// `SliceParent` variant with this augmented matrix as parent),
                /// so it is bound here rather than deferred.
                fn row_segment_mut(
                    slf: PyRef<'_, Self>,
                    i: usize,
                    start: usize,
                    end: usize,
                ) -> PyResult<PyFpSliceMut> {
                    checked_row(i, slf.0.rows())?;
                    segment_cols(&slf.0, start, end)?;
                    let col_start = slf.0.start[start];
                    let col_end = slf.0.end[end];
                    let py = slf.py();
                    Ok(PyFpSliceMut {
                        parent: SliceParent::MatrixRow {
                            matrix: MatrixParent::$variant(slf.into_pyobject(py)?.unbind()),
                            row: i,
                        },
                        start: col_start,
                        end: col_end,
                    })
                }

                /// Compute the kernel of the augmented matrix (which must be row
                /// reduced), returning an owned `Subspace`. Available for all
                /// arities. Raises `ValueError` if the matrix has not been row
                /// reduced (its pivots are uninitialized), instead of panicking.
                fn compute_kernel(&self) -> PyResult<PySubspace> {
                    ensure_pivots_initialized(self.0.pivots().len(), self.0.columns())?;
                    Ok(PySubspace(self.0.compute_kernel()))
                }

                /// Return the inner `Matrix` as an owned `Matrix`.
                ///
                /// Upstream `into_matrix` consumes `self`, but PyO3 methods
                /// borrow the pyclass and cannot move out of it, so we clone the
                /// inner matrix. The augmented matrix remains usable afterward.
                fn into_matrix(&self) -> PyMatrix {
                    PyMatrix(self.0.inner.clone())
                }

                fn __repr__(&self) -> String {
                    format!(
                        concat!($pyname, "({}, {}x{})"),
                        self.0.prime().as_u32(),
                        self.0.rows(),
                        self.0.columns()
                    )
                }

                $($extra)*
            }
        };
    }

    augmented_matrix_pyclass!(PyAugmentedMatrix2, "AugmentedMatrix2", 2, Augmented2, {
        /// Compute the image of the augmented matrix `[A | I]` (which must be
        /// row reduced), returning an owned `Subspace`. Raises `ValueError` if
        /// the matrix has not been row reduced, instead of panicking.
        fn compute_image(&self) -> PyResult<PySubspace> {
            ensure_pivots_initialized(self.0.pivots().len(), self.0.columns())?;
            Ok(PySubspace(self.0.compute_image()))
        }

        /// Compute the quasi-inverse of the augmented matrix `[A | I]` (which
        /// must be row reduced), returning an owned `QuasiInverse`. Raises
        /// `ValueError` if the matrix has not been row reduced, instead of
        /// panicking.
        fn compute_quasi_inverse(&self) -> PyResult<PyQuasiInverse> {
            ensure_pivots_initialized(self.0.pivots().len(), self.0.columns())?;
            Ok(PyQuasiInverse(self.0.compute_quasi_inverse()))
        }
    });

    augmented_matrix_pyclass!(PyAugmentedMatrix3, "AugmentedMatrix3", 3, Augmented3, {
        /// Compute the two quasi-inverses for a row-reduced augmented matrix of
        /// the form `[A | 0 | B | 0 | I]` where `A` is surjective, returning the
        /// pair `(quasi_inverse_of_A, residual_quasi_inverse)`.
        ///
        /// Upstream `compute_quasi_inverses` consumes and heavily mutates the
        /// matrix; since PyO3 cannot move out of a borrowed pyclass we operate
        /// on a clone, leaving the original augmented matrix unchanged.
        ///
        /// Raises `ValueError` if the matrix has not been row reduced (its
        /// pivots are uninitialized), instead of panicking.
        fn compute_quasi_inverses(&self) -> PyResult<(PyQuasiInverse, PyQuasiInverse)> {
            ensure_pivots_initialized(self.0.pivots().len(), self.0.columns())?;
            let (a, b) = self.0.clone().compute_quasi_inverses();
            Ok((PyQuasiInverse(a), PyQuasiInverse(b)))
        }
    });

    #[pymethods]
    impl PyFpVectorIterator {
        pub fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
            slf
        }

        pub fn __next__(&mut self) -> Option<u32> {
            let value = self.entries.get(self.index).copied();
            self.index += usize::from(value.is_some());
            value
        }
    }

    #[pymethods]
    impl PySubspaceVectorIterator {
        pub fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
            slf
        }

        pub fn __next__(&mut self) -> Option<PyFpVector> {
            if self.index >= self.total {
                return None;
            }
            let p = u128::from(self.subspace.prime().as_u32());
            let dim = self.subspace.dimension();
            // Decode `index` into base-`p` digits, most significant first, to
            // match the lexicographic order of `combinations` upstream where
            // the first digit (matching the first basis row) varies slowest.
            let mut digits = vec![0u32; dim];
            let mut rem = self.index;
            for slot in digits.iter_mut().rev() {
                *slot = (rem % p) as u32;
                rem /= p;
            }
            let mut vector =
                RustFpVector::new(self.subspace.prime(), self.subspace.ambient_dimension());
            for (&c, row) in digits.iter().zip(self.subspace.iter()) {
                vector.as_slice_mut().add(row, c);
            }
            self.index += 1;
            Some(PyFpVector(vector))
        }
    }

    #[pyfunction]
    fn power_mod(p: u32, b: u32, e: u32) -> PyResult<u32> {
        Ok(valid_prime(p)?.pow_mod(b, e))
    }

    #[pyfunction]
    fn log2(n: usize) -> usize {
        prime::log2(n)
    }

    #[pyfunction]
    fn logp(p: u32, n: u32) -> PyResult<u32> {
        Ok(prime::logp(valid_prime(p)?, n))
    }

    #[pyfunction]
    fn factor_pk(p: u32, n: u32) -> PyResult<(u32, u32)> {
        Ok(prime::factor_pk(valid_prime(p)?, n))
    }

    #[pyfunction]
    fn inverse(p: u32, k: u32) -> PyResult<u32> {
        Ok(prime::inverse(valid_prime(p)?, k))
    }

    #[pyfunction]
    fn minus_one_to_the_n(p: u32, i: i32) -> PyResult<u32> {
        Ok(prime::minus_one_to_the_n(valid_prime(p)?, i))
    }

    #[pyfunction]
    fn is_prime(p: u32) -> bool {
        valid_prime(p).is_ok()
    }

    #[pyfunction]
    fn binomial(p: u32, n: u32, k: u32) -> PyResult<u32> {
        Ok(u32::binomial(table_prime(p)?, n, k))
    }

    #[pyfunction]
    fn multinomial(p: u32, mut l: Vec<u32>) -> PyResult<u32> {
        Ok(u32::multinomial(table_prime(p)?, &mut l))
    }

    #[pyfunction]
    fn binomial_odd_is_zero(p: u32, n: u32, k: u32) -> PyResult<bool> {
        Ok(u32::binomial_odd_is_zero(table_prime(p)?, n, k))
    }

    #[pyfunction]
    fn binomial2(n: u32, k: u32) -> u32 {
        u32::binomial2(n, k)
    }

    #[pyfunction]
    fn multinomial2(l: Vec<u32>) -> u32 {
        u32::multinomial2(&l)
    }

    #[pyfunction]
    fn binomial4(n: u32, k: u32) -> u32 {
        u32::binomial4(n, k)
    }

    #[pyfunction]
    fn binomial4_rec(n: u32, k: u32) -> u32 {
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
        // The `AugmentedMatrix2`/`AugmentedMatrix3` classes are produced by a
        // `macro_rules!` macro, which the `#[pymodule]` proc-macro cannot see
        // through to auto-collect, so register them explicitly here.
        m.add_class::<PyAugmentedMatrix2>()?;
        m.add_class::<PyAugmentedMatrix3>()?;
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
        fn quasi_inverse_new_validates_image() {
            Python::initialize();
            Python::attach(|py| {
                let preimage = PyMatrix(RustMatrix::from_vec(
                    valid_prime(2).unwrap(),
                    &[vec![1, 0], vec![0, 1]],
                ));
                // Too many non-negative pivots for a 2-row preimage.
                let err = unwrap_py_err(PyQuasiInverse::new(Some(vec![0, 1, 0]), &preimage));
                assert!(err.is_instance_of::<PyValueError>(py));
                // Non-negative pivot out of row range.
                let err = unwrap_py_err(PyQuasiInverse::new(Some(vec![0, 2]), &preimage));
                assert!(err.is_instance_of::<PyValueError>(py));
                // Valid image with gaps (negative pivots) is accepted.
                assert!(PyQuasiInverse::new(Some(vec![0, -1, 1]), &preimage).is_ok());
                // None image is always accepted.
                assert!(PyQuasiInverse::new(None, &preimage).is_ok());
            });
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
        fn fp_vector_constructors_and_prime_return() {
            let v = PyFpVector::new(5, 4).unwrap();
            assert_eq!(v.prime(), 5);
            assert_eq!(v.len(), 4);
            assert_eq!(v.__len__(), 4);
            assert!(!v.is_empty());
            assert!(v.is_zero());
            assert_eq!(v.__repr__(), "FpVector(5, [0, 0, 0, 0])");

            let empty = PyFpVector::new_with_capacity(3, 0, 8).unwrap();
            assert_eq!(empty.prime(), 3);
            assert!(empty.is_empty());

            let from_slice = PyFpVector::from_slice(5, vec![0, 1, 7, 4]).unwrap();
            assert_eq!(from_slice.prime(), 5);
            assert_eq!(from_slice.len(), 4);
            assert_eq!(from_slice.entry(2).unwrap(), 2);
            assert_eq!(from_slice.first_nonzero(), Some((1, 1)));
            assert_eq!(from_slice.density(), 0.75);
        }

        #[test]
        fn fp_vector_indexing_and_mutation() {
            let mut v = PyFpVector::new(5, 3).unwrap();
            v.set_entry(0, 7).unwrap();
            v.__setitem__(-1, 4).unwrap();
            assert_eq!(v.entry(0).unwrap(), 2);
            assert_eq!(v.__getitem__(-3).unwrap(), 2);
            assert_eq!(v.__getitem__(2).unwrap(), 4);
            assert!(v.entry(3).is_err());
            assert!(v.__getitem__(-4).is_err());

            v.add_basis_element(0, 4).unwrap();
            assert_eq!(v.entry(0).unwrap(), 1);
            v.scale(3);
            assert_eq!(v.entry(0).unwrap(), 3);
            assert_eq!(v.entry(2).unwrap(), 2);
            v.extend_len(5);
            assert_eq!(v.len(), 5);
            assert_eq!(v.entry(4).unwrap(), 0);
            v.set_scratch_vector_size(2);
            assert_eq!(v.len(), 2);
            assert!(v.is_zero());
            v.set_entry(1, 1).unwrap();
            v.set_to_zero();
            assert!(v.is_zero());
        }

        #[test]
        fn fp_vector_slice_range_helper_errors() {
            assert!(checked_range(0, 3, 3).is_ok());
            assert!(checked_range(1, 1, 3).is_ok());
            assert!(checked_range(2, 1, 3).is_err());
            assert!(checked_range(0, 4, 3).is_err());
        }

        #[test]
        fn fp_vector_rejects_invalid_prime() {
            assert!(PyFpVector::new(1, 3).is_err());
            assert!(PyFpVector::new_with_capacity(9, 3, 8).is_err());
            assert!(PyFpVector::from_slice(4, vec![1, 2]).is_err());
            assert!(PyFpVector::from_bytes(0, 2, &[]).is_err());
        }

        #[test]
        fn fp_vector_bytes_roundtrip() {
            Python::initialize();
            Python::attach(|py| {
                let v = PyFpVector::from_slice(5, vec![0, 1, 2, 3, 4, 7]).unwrap();
                let bytes = v.to_bytes(py).unwrap();
                let w = PyFpVector::from_bytes(5, v.len(), bytes.as_bytes()).unwrap();
                assert_eq!(w.__getitem__(5).unwrap(), 2);
                assert_eq!(w.__repr__(), "FpVector(5, [0, 1, 2, 3, 4, 2])");

                let mut z = PyFpVector::new(5, v.len()).unwrap();
                z.update_from_bytes(bytes.as_bytes()).unwrap();
                assert_eq!(z.__repr__(), w.__repr__());
            });
        }

        #[test]
        fn fp_vector_iteration() {
            let v = PyFpVector::from_slice(3, vec![0, 1, 2, 4]).unwrap();
            let mut iter = PyFpVectorIterator {
                entries: v.0.iter().collect(),
                index: 0,
            };
            assert_eq!(iter.__next__(), Some(0));
            assert_eq!(iter.__next__(), Some(1));
            assert_eq!(iter.__next__(), Some(2));
            assert_eq!(iter.__next__(), Some(1));
            assert_eq!(iter.__next__(), None);
        }

        #[test]
        fn matrix_constructors_and_queries() {
            let m = PyMatrix::from_vec(7, vec![vec![1, 3, 6], vec![0, 3, 4]]).unwrap();
            assert_eq!(m.prime(), 7);
            assert_eq!(m.rows(), 2);
            assert_eq!(m.columns(), 3);
            assert!(!m.is_zero());
            assert_eq!(m.to_vec(), vec![vec![1, 3, 6], vec![0, 3, 4]]);
            assert_eq!(m.__len__(), 2);

            let id = PyMatrix::identity(5, 3).unwrap();
            assert_eq!(
                id.to_vec(),
                vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
            );

            assert!(PyMatrix::new(1, 2, 2).is_err());
            assert!(PyMatrix::from_vec(4, vec![vec![1]]).is_err());
            assert!(PyMatrix::from_vec(7, vec![vec![1, 2], vec![1]]).is_err());
        }

        #[test]
        fn matrix_row_reduce_rank() {
            let mut m =
                PyMatrix::from_vec(2, vec![vec![1, 1, 0], vec![0, 1, 1], vec![1, 0, 1]]).unwrap();
            assert_eq!(m.row_reduce(), 2);
        }

        #[test]
        fn matrix_mutation_and_invalid_rows() {
            let mut m = PyMatrix::from_vec(5, vec![vec![1, 2], vec![3, 4]]).unwrap();
            m.swap_rows(0, 1).unwrap();
            assert_eq!(m.to_vec(), vec![vec![3, 4], vec![1, 2]]);
            assert!(m.swap_rows(0, 2).is_err());
            assert!(m.safe_row_op(0, 0, 1).is_err());
            m.safe_row_op(0, 1, 1).unwrap();
            assert_eq!(m.to_vec(), vec![vec![4, 1], vec![1, 2]]);
            m.set_to_zero();
            assert!(m.is_zero());
        }

        #[test]
        fn matrix_bytes_roundtrip() {
            Python::initialize();
            Python::attach(|py| {
                let m = PyMatrix::from_vec(5, vec![vec![1, 2, 3], vec![4, 0, 2]]).unwrap();
                let bytes = m.to_bytes(py).unwrap();
                let n = PyMatrix::from_bytes(5, 2, 3, bytes.as_bytes()).unwrap();
                assert_eq!(n.to_vec(), m.to_vec());
            });
        }

        #[test]
        fn matrix_row_slice_unifies_with_vector_slice() {
            Python::initialize();
            Python::attach(|py| {
                let matrix = Py::new(
                    py,
                    PyMatrix::from_vec(5, vec![vec![1, 2, 3], vec![4, 0, 1]]).unwrap(),
                )
                .unwrap();

                // Read a matrix row through the unified FpSlice handle.
                let row = PyMatrix::row(matrix.borrow(py), 1).unwrap();
                assert_eq!(row.prime(py).unwrap(), 5);
                assert_eq!(row.len(py).unwrap(), 3);
                assert_eq!(row.entry(py, 0).unwrap(), 4);
                assert!(!row.is_zero(py).unwrap());
                assert_eq!(row.to_owned(py).unwrap().0.entry(2), 1);

                // Mutate a matrix row through the unified FpSliceMut handle.
                let row_mut = PyMatrix::row_mut(matrix.borrow(py), 0).unwrap();
                row_mut.set_entry(py, 0, 9).unwrap();
                assert_eq!(matrix.borrow(py).0.row(0).entry(0), 4);

                // Stale handle after trimming raises rather than panics.
                matrix.borrow_mut(py).0.trim(0, 1, 0, false);
                assert!(PyMatrix::row(matrix.borrow(py), 1).is_err());
            });
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

        #[test]
        fn subspace_basic_and_bytes_roundtrip() {
            Python::initialize();
            Python::attach(|py| {
                let mut s = PySubspace::new(3, 3).unwrap();
                assert_eq!(s.prime(), 3);
                assert_eq!(s.ambient_dimension(), 3);
                assert_eq!(s.dimension(), 0);

                let v = PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap();
                assert_eq!(s.add_vector(&v).unwrap(), 1);
                assert!(s.contains(&v).unwrap());
                assert_eq!(s.__len__(), 1);

                let w = PyFpVector::from_slice(3, vec![0, 1, 0]).unwrap();
                assert!(!s.contains(&w).unwrap());

                // Prime/dimension mismatches raise rather than panic.
                let wrong_prime = PyFpVector::from_slice(5, vec![1, 0, 0]).unwrap();
                assert!(s.contains(&wrong_prime).is_err());
                let wrong_dim = PyFpVector::from_slice(3, vec![1, 0]).unwrap();
                assert!(s.add_vector(&wrong_dim).is_err());

                // reduce in place projects onto the complement.
                let mut to_reduce = PyFpVector::from_slice(3, vec![2, 1, 0]).unwrap();
                s.reduce(&mut to_reduce).unwrap();
                assert_eq!(to_reduce.0.entry(0), 0);
                assert_eq!(to_reduce.0.entry(1), 1);

                // Bytes roundtrip.
                let bytes = s.to_bytes(py).unwrap();
                let restored = PySubspace::from_bytes(3, bytes.as_bytes()).unwrap();
                assert_eq!(restored.dimension(), 1);
                assert!(restored.contains(&v).unwrap());
                assert!(PySubspace::from_bytes(3, &[0, 1, 2]).is_err());
            });
        }

        #[test]
        fn subspace_sum_is_span_of_both() {
            // The sum of two subspaces is the span of both summands. Its
            // dimension equals the rank of the combined span, capped by the
            // ambient dimension.

            // Two complementary lines span a 2-dimensional subspace.
            let mut a = PySubspace::new(3, 3).unwrap();
            a.add_vector(&PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap())
                .unwrap();
            let mut b = PySubspace::new(3, 3).unwrap();
            b.add_vector(&PyFpVector::from_slice(3, vec![0, 1, 0]).unwrap())
                .unwrap();

            let s = a.sum(&b).unwrap();
            assert_eq!(s.dimension(), 2);
            assert!(s.contains_space(&a).unwrap());
            assert!(s.contains_space(&b).unwrap());
            assert!(s
                .contains(&PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap())
                .unwrap());
            assert!(s
                .contains(&PyFpVector::from_slice(3, vec![0, 1, 0]).unwrap())
                .unwrap());
            assert_eq!(s.iter().len(), 2);

            // Overlapping subspaces: the sum's dimension is the union's rank.
            let mut c = PySubspace::new(3, 3).unwrap();
            c.add_vector(&PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap())
                .unwrap();
            c.add_vector(&PyFpVector::from_slice(3, vec![0, 1, 0]).unwrap())
                .unwrap();
            let mut d = PySubspace::new(3, 3).unwrap();
            d.add_vector(&PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap())
                .unwrap();
            let overlap = c.sum(&d).unwrap();
            assert_eq!(overlap.dimension(), 2);
            assert!(overlap.contains_space(&c).unwrap());
            assert!(overlap.contains_space(&d).unwrap());

            // The sum is capped by the ambient dimension.
            let full = c.sum(&b).unwrap();
            assert_eq!(full.dimension(), 2);
            assert!(full.dimension() <= full.ambient_dimension());

            // Prime/ambient mismatches raise.
            let other_prime = PySubspace::new(5, 3).unwrap();
            assert!(a.sum(&other_prime).is_err());
            let other_dim = PySubspace::new(3, 4).unwrap();
            assert!(a.contains_space(&other_dim).is_err());
        }

        #[test]
        fn quasi_inverse_compute_and_dims() {
            // Example from `Matrix::compute_quasi_inverse` doc.
            let input = vec![
                vec![1, 2, 1, 1, 0],
                vec![1, 0, 2, 1, 1],
                vec![2, 2, 0, 2, 1],
            ];
            let (padded_cols, mut m) = PyMatrix::augmented_from_vec(3, input.clone()).unwrap();
            m.row_reduce();
            let qi = m
                .compute_quasi_inverse(input[0].len(), padded_cols)
                .unwrap();
            assert_eq!(qi.prime(), 3);
            assert_eq!(qi.source_dimension(), 3);
            assert_eq!(qi.preimage().to_vec(), vec![vec![0, 1, 0], vec![0, 2, 2]]);
            // Out-of-range columns raise rather than panic.
            assert!(m.compute_quasi_inverse(input[0].len(), 999).is_err());
        }

        #[test]
        fn subquotient_add_gen_reduce_and_quotient() {
            // Mirrors the upstream `test_add_gen` example at p = 3, dim = 5.
            let mut sq = PySubquotient::new(3, 5).unwrap();
            assert_eq!(sq.prime(), 3);
            assert_eq!(sq.ambient_dimension(), 5);
            assert_eq!(sq.dimension(), 0);
            assert!(sq.is_empty());

            sq.quotient(&PyFpVector::from_slice(3, vec![1, 1, 0, 0, 1]).unwrap())
                .unwrap();
            sq.quotient(&PyFpVector::from_slice(3, vec![0, 2, 0, 0, 1]).unwrap())
                .unwrap();
            sq.add_gen(&PyFpVector::from_slice(3, vec![1, 1, 0, 0, 0]).unwrap())
                .unwrap();
            sq.add_gen(&PyFpVector::from_slice(3, vec![0, 1, 0, 0, 0]).unwrap())
                .unwrap();

            assert_eq!(sq.dimension(), 1);
            assert_eq!(sq.gens().len(), 1);
            assert_eq!(sq.zeros().dimension(), 2);
            assert_eq!(sq.__len__(), 1);

            // The complement + quotient + gens cover the ambient space.
            assert_eq!(
                sq.zeros().dimension() + sq.gens().len() + sq.complement_pivots().len(),
                sq.ambient_dimension()
            );

            // reduce a known vector, matching upstream's expected [2].
            let mut elt = PyFpVector::from_slice(3, vec![2, 0, 0, 0, 0]).unwrap();
            assert_eq!(sq.reduce(&mut elt).unwrap(), vec![2]);

            // clear_gens drops the generators but keeps the quotient.
            sq.clear_gens();
            assert_eq!(sq.dimension(), 0);
            assert_eq!(sq.zeros().dimension(), 2);

            // Prime/dimension mismatches raise rather than panic.
            assert!(sq
                .quotient(&PyFpVector::from_slice(5, vec![1, 0, 0, 0, 0]).unwrap())
                .is_err());
            assert!(sq
                .add_gen(&PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap())
                .is_err());
        }

        #[test]
        fn subquotient_subspace_gens_quotient_pivots_dimension() {
            // Non-trivial subquotient mirroring upstream `test_add_gen`.
            let mut sq = PySubquotient::new(3, 5).unwrap();
            sq.quotient(&PyFpVector::from_slice(3, vec![1, 1, 0, 0, 1]).unwrap())
                .unwrap();
            sq.quotient(&PyFpVector::from_slice(3, vec![0, 2, 0, 0, 1]).unwrap())
                .unwrap();
            sq.add_gen(&PyFpVector::from_slice(3, vec![1, 1, 0, 0, 0]).unwrap())
                .unwrap();
            sq.add_gen(&PyFpVector::from_slice(3, vec![0, 1, 0, 0, 0]).unwrap())
                .unwrap();

            assert_eq!(sq.dimension(), 1);
            assert_eq!(sq.zeros().dimension(), 2);

            // subspace_dimension == dimension + quotient.dimension().
            assert_eq!(
                sq.subspace_dimension(),
                sq.dimension() + sq.zeros().dimension()
            );
            assert_eq!(sq.subspace_dimension(), 3);

            // subspace_gens chains gens() with the quotient basis.
            let subspace_gens: Vec<Vec<u32>> = sq
                .subspace_gens()
                .iter()
                .map(|v| v.0.iter().collect())
                .collect();
            assert_eq!(
                subspace_gens,
                vec![
                    vec![0, 0, 0, 0, 1],
                    vec![1, 0, 0, 0, 2],
                    vec![0, 1, 0, 0, 2],
                ]
            );

            // quotient_pivots: pivots[col] = pivot row index, else -1.
            assert_eq!(sq.quotient_pivots(), vec![0, 1, -1, -1, -1]);
        }

        #[test]
        fn subquotient_reduce_matrix_values_with_nontrivial_quotient() {
            // source = full dim-2 space: gens [1,0] and [0,1].
            let source = PySubquotient::new_full(3, 2).unwrap();

            // target has a non-trivial quotient killing column 1, gen [1,0].
            let mut target = PySubquotient::new(3, 2).unwrap();
            target
                .quotient(&PyFpVector::from_slice(3, vec![0, 1]).unwrap())
                .unwrap();
            target
                .add_gen(&PyFpVector::from_slice(3, vec![1, 0]).unwrap())
                .unwrap();

            // apply computes input * matrix: gen [1,0] -> [2,1]; [0,1] -> [0,1].
            let m = PyMatrix::from_vec(3, vec![vec![2, 1], vec![0, 1]]).unwrap();

            // Reduced in target: [2,1] -> [2]; [0,1] -> [0].
            let result = PySubquotient::reduce_matrix(&m, &source, &target).unwrap();
            assert_eq!(result, vec![vec![2], vec![0]]);

            // Dimension/prime mismatches raise.
            let bad_rows = PyMatrix::from_vec(3, vec![vec![1, 0], vec![0, 1], vec![0, 0]]).unwrap();
            assert!(PySubquotient::reduce_matrix(&bad_rows, &source, &target).is_err());
            let bad_cols = PyMatrix::from_vec(3, vec![vec![1, 0, 0], vec![0, 1, 0]]).unwrap();
            assert!(PySubquotient::reduce_matrix(&bad_cols, &source, &target).is_err());
            let bad_prime = PyMatrix::from_vec(5, vec![vec![1, 0], vec![0, 1]]).unwrap();
            assert!(PySubquotient::reduce_matrix(&bad_prime, &source, &target).is_err());
        }

        #[test]
        fn subquotient_set_to_full_stale_dimension() {
            // set_to_full clears the quotient and fills gens, but upstream does
            // NOT sync the cached `dimension`. Pin the surprising current state.
            let mut sq = PySubquotient::new(2, 3).unwrap();
            sq.set_to_full();
            assert_eq!(sq.zeros().dimension(), 0);
            // Stale: dimension stays 0 while gens() actually has 3 rows.
            assert_eq!(sq.dimension(), 0);
            assert_eq!(sq.__len__(), 0);
            assert_eq!(sq.gens().len(), 3);
        }

        #[test]
        fn subquotient_new_full_and_from_parts() {
            let full = PySubquotient::new_full(2, 4).unwrap();
            assert_eq!(full.dimension(), 4);
            assert_eq!(full.quotient_dimension(), 4);
            assert_eq!(full.gens().len(), 4);

            let mut sub = PySubspace::new(2, 3).unwrap();
            sub.add_vector(&PyFpVector::from_slice(2, vec![1, 0, 0]).unwrap())
                .unwrap();
            sub.add_vector(&PyFpVector::from_slice(2, vec![0, 1, 0]).unwrap())
                .unwrap();
            let mut quot = PySubspace::new(2, 3).unwrap();
            quot.add_vector(&PyFpVector::from_slice(2, vec![1, 0, 0]).unwrap())
                .unwrap();

            let sq = PySubquotient::from_parts(&sub, &quot).unwrap();
            assert_eq!(sq.dimension(), 1);
            assert_eq!(sq.ambient_dimension(), 3);

            // Mismatched ambient dimension raises.
            let bad = PySubspace::new(2, 4).unwrap();
            assert!(PySubquotient::from_parts(&sub, &bad).is_err());
        }

        /// Build a `PySubspace` over prime `p` from explicit basis rows.
        fn subspace_from_rows(p: u32, rows: &[Vec<u32>]) -> PySubspace {
            let dim = rows[0].len();
            let mut s = PySubspace::new(p, dim).unwrap();
            for row in rows {
                s.add_vector(&PyFpVector::from_slice(p, row.clone()).unwrap())
                    .unwrap();
            }
            s
        }

        #[test]
        fn affine_subspace_offset_and_linear_part() {
            // linear_part = span{[0,1,0],[0,0,1]} (pivots in columns 1 and 2),
            // offset = [1,1,0], which is NOT already reduced: its column-1 entry
            // sits in a pivot column of the linear part.
            let linear = subspace_from_rows(2, &[vec![0, 1, 0], vec![0, 0, 1]]);
            let offset = PyFpVector::from_slice(2, vec![1, 1, 0]).unwrap();
            let aff = PyAffineSubspace::new(&offset, &linear).unwrap();

            assert_eq!(aff.prime(), 2);
            assert_eq!(aff.ambient_dimension(), 3);
            assert_eq!(aff.dimension(), 2);

            // Subspace::reduce subtracts the pivot rows: [1,1,0] - [0,1,0] gives
            // [1,0,0] (column-2 entry is already 0), so the stored offset is the
            // reduced form [1,0,0]. Mirrors Python's test_offset_is_reduced.
            let stored = aff.offset();
            assert_eq!(stored.entry(0).unwrap(), 1);
            assert_eq!(stored.entry(1).unwrap(), 0);
            assert_eq!(stored.entry(2).unwrap(), 0);
            // linear_part round-trips dimension/ambient.
            assert_eq!(aff.linear_part().dimension(), 2);
            assert_eq!(aff.linear_part().ambient_dimension(), 3);
            assert_eq!(
                aff.__repr__(),
                "AffineSubspace([1, 0, 0] + {[0, 1, 0], [0, 0, 1]})"
            );
        }

        #[test]
        fn affine_subspace_contains() {
            Python::initialize();
            Python::attach(|py| {
                let linear = subspace_from_rows(2, &[vec![0, 1, 0], vec![0, 0, 1]]);
                let offset = PyFpVector::from_slice(2, vec![1, 0, 0]).unwrap();
                let aff = PyAffineSubspace::new(&offset, &linear).unwrap();

                let inside = PyFpVector::from_slice(2, vec![1, 1, 0])
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any();
                assert!(aff.contains(py, &inside).unwrap());

                let outside = PyFpVector::from_slice(2, vec![0, 1, 0])
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any();
                assert!(!aff.contains(py, &outside).unwrap());

                // Wrong ambient dimension raises.
                let bad = PyFpVector::from_slice(2, vec![1, 1])
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any();
                assert!(aff.contains(py, &bad).is_err());
            });
        }

        #[test]
        fn affine_subspace_sum() {
            Python::initialize();
            Python::attach(|py| {
                // a = [1,0,0] + span{[0,1,0]}, b = [0,0,1] + span{[0,0,1]}.
                let a = PyAffineSubspace::new(
                    &PyFpVector::from_slice(2, vec![1, 0, 0]).unwrap(),
                    &subspace_from_rows(2, &[vec![0, 1, 0]]),
                )
                .unwrap();
                let b = PyAffineSubspace::new(
                    &PyFpVector::from_slice(2, vec![0, 0, 1]).unwrap(),
                    &subspace_from_rows(2, &[vec![0, 0, 1]]),
                )
                .unwrap();

                // sum adds the already-stored, already-reduced offsets, not
                // the constructor arguments. a.offset stays [1,0,0] (already
                // reduced against span{[0,1,0]}); b's offset [0,0,1] reduces to
                // [0,0,0] against span{[0,0,1]} when b is constructed. So the
                // sum offset is a.offset [1,0,0] + b.offset [0,0,0] = [1,0,0],
                // re-reduced against span{[0,1,0],[0,0,1]} (dim 2) to [1,0,0].
                // No [1,0,1] intermediate ever occurs.
                let s = a.sum(&b).unwrap();
                assert_eq!(s.dimension(), 2);
                assert_eq!(s.offset().entry(0).unwrap(), 1);
                assert_eq!(s.offset().entry(1).unwrap(), 0);
                assert_eq!(s.offset().entry(2).unwrap(), 0);

                let inside = PyFpVector::from_slice(2, vec![1, 1, 1])
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any();
                assert!(s.contains(py, &inside).unwrap());
                let outside = PyFpVector::from_slice(2, vec![0, 0, 0])
                    .unwrap()
                    .into_pyobject(py)
                    .unwrap()
                    .into_any();
                assert!(!s.contains(py, &outside).unwrap());
            });
        }

        #[test]
        fn affine_subspace_contains_space() {
            // a = origin + span{[0,1,0],[0,0,1]} (a linear subspace).
            let a = PyAffineSubspace::new(
                &PyFpVector::from_slice(2, vec![0, 0, 0]).unwrap(),
                &subspace_from_rows(2, &[vec![0, 1, 0], vec![0, 0, 1]]),
            )
            .unwrap();
            // b = [0,1,0] + span{[0,1,0]}: linear part and offset both lie in a.
            let b = PyAffineSubspace::new(
                &PyFpVector::from_slice(2, vec![0, 1, 0]).unwrap(),
                &subspace_from_rows(2, &[vec![0, 1, 0]]),
            )
            .unwrap();
            // c = [1,0,0] + span{[0,1,0]}: offset is outside a.
            let c = PyAffineSubspace::new(
                &PyFpVector::from_slice(2, vec![1, 0, 0]).unwrap(),
                &subspace_from_rows(2, &[vec![0, 1, 0]]),
            )
            .unwrap();

            assert!(a.contains_space(&b).unwrap());
            assert!(!a.contains_space(&c).unwrap());

            // Mismatched prime/dimension raise.
            let other_prime = PyAffineSubspace::new(
                &PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap(),
                &subspace_from_rows(3, &[vec![0, 1, 0]]),
            )
            .unwrap();
            assert!(a.sum(&other_prime).is_err());
            assert!(a.contains_space(&other_prime).is_err());

            let other_dim = PyAffineSubspace::new(
                &PyFpVector::from_slice(2, vec![1, 0, 0, 0]).unwrap(),
                &subspace_from_rows(2, &[vec![0, 1, 0, 0]]),
            )
            .unwrap();
            assert!(a.sum(&other_dim).is_err());
        }

        #[test]
        fn affine_subspace_new_validates() {
            // Mismatched offset length vs linear ambient dimension.
            let linear = subspace_from_rows(2, &[vec![0, 1, 0]]);
            let bad_len = PyFpVector::from_slice(2, vec![1, 0]).unwrap();
            assert!(PyAffineSubspace::new(&bad_len, &linear).is_err());

            // Mismatched prime.
            let bad_prime = PyFpVector::from_slice(3, vec![1, 0, 0]).unwrap();
            assert!(PyAffineSubspace::new(&bad_prime, &linear).is_err());
        }

        #[test]
        fn augmented_matrix_construction_and_queries() {
            let m = PyAugmentedMatrix2::new(3, 2, vec![2, 2]).unwrap();
            assert_eq!(m.prime(), 3);
            assert_eq!(m.rows(), 2);
            assert_eq!(m.segments(), 2);
            // start[0] = 0; start[1] = padded_len(3, 2). columns() = end[1].
            assert_eq!(m.segment_starts()[0], 0);
            assert_eq!(m.segment_ends()[1], m.columns());
            assert!(m.is_zero());

            // Wrong number of segment widths raises.
            assert!(PyAugmentedMatrix2::new(3, 2, vec![2]).is_err());
            assert!(PyAugmentedMatrix2::new(3, 2, vec![2, 2, 2]).is_err());
            assert!(PyAugmentedMatrix3::new(3, 2, vec![2, 2]).is_err());
            // Invalid prime raises.
            assert!(PyAugmentedMatrix2::new(4, 2, vec![2, 2]).is_err());
        }

        #[test]
        fn augmented_matrix_add_identity_and_segment_checks() {
            // [A | I] with A having 2 columns and a square identity segment.
            let mut m = PyAugmentedMatrix2::new(2, 2, vec![2, 2]).unwrap();
            m.add_identity(1, 1).unwrap();
            // The identity sits in the second segment; row 0 has a 1 at start[1].
            let start1 = m.segment_starts()[1];
            assert_eq!(m.to_vec()[0][start1], 1);
            assert_eq!(m.to_vec()[1][start1 + 1], 1);

            // Out-of-range segment indices raise.
            assert!(m.add_identity(0, 2).is_err());
            assert!(m.row_segment(0, 0, 2).is_err());
            // Non-square segment raises (2 rows, segment of width != 2).
            let mut wide = PyAugmentedMatrix2::new(2, 2, vec![3, 3]).unwrap();
            assert!(wide.add_identity(0, 0).is_err());
        }

        #[test]
        fn augmented_matrix_row_segment_owned_copy() {
            let mut m = PyAugmentedMatrix2::new(2, 2, vec![2, 2]).unwrap();
            m.add_identity(1, 1).unwrap();
            // Row 0 restricted to segment 1 is the identity's first row.
            let seg = m.row_segment(0, 1, 1).unwrap();
            assert_eq!(seg.len(), 2);
            assert_eq!(seg.entry(0).unwrap(), 1);
            assert_eq!(seg.entry(1).unwrap(), 0);
        }

        #[test]
        fn augmented_matrix2_compute_image_and_quasi_inverse() {
            // Build [A | I] with A = [[1,1],[0,1]] over F2, then row reduce.
            let mut m = PyAugmentedMatrix2::new(2, 2, vec![2, 2]).unwrap();
            // Fill A via into_matrix? Instead set entries through to_vec round
            // trip is not possible; use add_identity for the I block and place
            // A by constructing from a known reduced form.
            m.add_identity(1, 1).unwrap();
            // Put A = identity in segment 0 as well so [I | I]; image is full.
            m.add_identity(0, 0).unwrap();
            m.row_reduce();
            let image = m.compute_image().unwrap();
            assert_eq!(image.prime(), 2);
            assert_eq!(image.dimension(), 2);
            let qi = m.compute_quasi_inverse().unwrap();
            assert_eq!(qi.prime(), 2);
            assert_eq!(qi.source_dimension(), 2);
            // into_matrix returns an owned Matrix leaving the original usable.
            let inner = m.into_matrix();
            assert_eq!(inner.rows(), 2);
            assert_eq!(m.rows(), 2);
        }

        #[test]
        fn augmented_matrix3_compute_kernel_and_quasi_inverses() {
            // [A | B | I] all square identity blocks over F3.
            let mut m = PyAugmentedMatrix3::new(3, 2, vec![2, 2, 2]).unwrap();
            m.add_identity(0, 0).unwrap();
            m.add_identity(1, 1).unwrap();
            m.add_identity(2, 2).unwrap();
            m.row_reduce();
            let ker = m.compute_kernel().unwrap();
            assert_eq!(ker.prime(), 3);
            // [A | B | I] with all identity blocks: A is the identity, so it is
            // injective and surjective. The kernel of an injective map is zero.
            assert_eq!(ker.dimension(), 0);
            let (a, b) = m.compute_quasi_inverses().unwrap();
            assert_eq!(a.prime(), 3);
            assert_eq!(b.prime(), 3);
            // A = I is surjective onto F3^2, so its quasi-inverse has source
            // and target dimension 2; the residual quasi-inverse is then trivial
            // (target dimension 0).
            assert_eq!(a.source_dimension(), 2);
            assert_eq!(a.target_dimension(), 2);
            // The residual quasi-inverse b inverts B (= I) on the kernel of A.
            // With every block the 2x2 identity it is itself a full-rank 2->2
            // map, so its source, target and image dimensions are all 2.
            assert_eq!(b.source_dimension(), 2);
            assert_eq!(b.target_dimension(), 2);
            assert_eq!(b.image_dimension(), 2);
        }

        #[test]
        fn augmented_matrix_compute_requires_row_reduce() {
            // Reproduction from the review: compute_* on a freshly constructed
            // (not row-reduced) augmented matrix must raise ValueError rather
            // than panic across the boundary on the empty-pivots slice.
            Python::initialize();
            Python::attach(|py| {
                let m2 = PyAugmentedMatrix2::new(2, 2, vec![2, 2]).unwrap();
                for err in [
                    unwrap_py_err(m2.compute_kernel()),
                    unwrap_py_err(m2.compute_image()),
                    unwrap_py_err(m2.compute_quasi_inverse()),
                ] {
                    assert!(err.is_instance_of::<PyValueError>(py));
                }

                let m3 = PyAugmentedMatrix3::new(3, 2, vec![2, 2, 2]).unwrap();
                assert!(unwrap_py_err(m3.compute_kernel()).is_instance_of::<PyValueError>(py));
                assert!(
                    unwrap_py_err(m3.compute_quasi_inverses()).is_instance_of::<PyValueError>(py)
                );

                // A bare Matrix.compute_quasi_inverse shares the gap.
                let raw = PyMatrix::new(2, 2, 4).unwrap();
                assert!(unwrap_py_err(raw.compute_quasi_inverse(2, 2))
                    .is_instance_of::<PyValueError>(py));
            });
        }

        #[test]
        fn matrix_slice_mut_mutates_parent() {
            Python::initialize();
            Python::attach(|py| {
                let matrix = Py::new(
                    py,
                    PyMatrix::from_vec(
                        5,
                        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![0, 1, 2, 3]],
                    )
                    .unwrap(),
                )
                .unwrap();

                // Rectangle over rows 0..2, columns 1..3.
                let rect = PyMatrix::slice_mut(matrix.borrow(py), 0, 2, 1, 3).unwrap();
                assert_eq!(rect.prime(py).unwrap(), 5);
                assert_eq!(rect.rows(py).unwrap(), 2);
                assert_eq!(rect.columns(py).unwrap(), 2);

                // row(i) is read-only and column-shifted into the rectangle.
                let row0 = rect.row(py, 0).unwrap();
                assert_eq!(row0.len(py).unwrap(), 2);
                assert_eq!(row0.entry(py, 0).unwrap(), 2);
                assert_eq!(row0.entry(py, 1).unwrap(), 3);

                // row_mut writes through to the parent matrix.
                let row1 = rect.row_mut(py, 1).unwrap();
                row1.set_entry(py, 0, 9).unwrap();
                assert_eq!(matrix.borrow(py).0.row(1).entry(1), 4);

                // iter_mut mutations reflect in the parent.
                for r in rect.iter_mut(py).unwrap() {
                    r.set_entry(py, 1, 0).unwrap();
                }
                assert_eq!(matrix.borrow(py).0.row(0).entry(2), 0);
                assert_eq!(matrix.borrow(py).0.row(1).entry(2), 0);

                // Out-of-range row index raises.
                assert!(rect.row(py, 2).is_err());
            });
        }

        #[test]
        fn matrix_slice_mut_add_identity_and_masked() {
            Python::initialize();
            Python::attach(|py| {
                let matrix = Py::new(py, PyMatrix::new(3, 2, 4).unwrap()).unwrap();
                // Square 2x2 rectangle over columns 2..4 gets an identity.
                let rect = PyMatrix::slice_mut(matrix.borrow(py), 0, 2, 2, 4).unwrap();
                rect.add_identity(py).unwrap();
                assert_eq!(matrix.borrow(py).0.row(0).entry(2), 1);
                assert_eq!(matrix.borrow(py).0.row(1).entry(3), 1);

                // Non-square rectangle raises on add_identity.
                let wide = PyMatrix::slice_mut(matrix.borrow(py), 0, 2, 0, 4).unwrap();
                assert!(wide.add_identity(py).is_err());

                // add_masked: pull columns [0, 1] of `other` into rect columns.
                let other = PyMatrix::from_vec(3, vec![vec![1, 2], vec![0, 1]]).unwrap();
                let rect2 = PyMatrix::slice_mut(matrix.borrow(py), 0, 2, 0, 2).unwrap();
                rect2.add_masked(py, &other, vec![0, 1]).unwrap();
                assert_eq!(matrix.borrow(py).0.row(0).entry(0), 1);
                assert_eq!(matrix.borrow(py).0.row(0).entry(1), 2);
                assert_eq!(matrix.borrow(py).0.row(1).entry(1), 1);

                // Mismatched mask length / row count raise.
                assert!(rect2.add_masked(py, &other, vec![0]).is_err());
                let bad_rows = PyMatrix::from_vec(3, vec![vec![1, 2]]).unwrap();
                assert!(rect2.add_masked(py, &bad_rows, vec![0, 1]).is_err());
            });
        }

        #[test]
        fn matrix_slice_mut_stale_handle_raises() {
            Python::initialize();
            Python::attach(|py| {
                let matrix = Py::new(
                    py,
                    PyMatrix::from_vec(2, vec![vec![1, 1, 0], vec![0, 1, 1]]).unwrap(),
                )
                .unwrap();
                let rect = PyMatrix::slice_mut(matrix.borrow(py), 0, 2, 0, 3).unwrap();
                // Trim the matrix to a single row; the 2-row rectangle is stale.
                matrix.borrow_mut(py).0.trim(0, 1, 0, false);
                assert!(rect.rows(py).is_err());
                // Invalid rectangle at construction raises too.
                assert!(PyMatrix::slice_mut(matrix.borrow(py), 0, 5, 0, 1).is_err());
            });
        }

        #[test]
        fn augmented_matrix_segment_mutates_inner() {
            Python::initialize();
            Python::attach(|py| {
                let m = Py::new(py, PyAugmentedMatrix2::new(2, 2, vec![2, 2]).unwrap()).unwrap();
                // Add identity into segment 1 via the MatrixSliceMut handle.
                let seg = PyAugmentedMatrix2::segment(m.borrow(py), 1, 1).unwrap();
                assert_eq!(seg.rows(py).unwrap(), 2);
                assert_eq!(seg.columns(py).unwrap(), 2);
                seg.add_identity(py).unwrap();
                let start1 = m.borrow(py).segment_starts()[1];
                assert_eq!(m.borrow(py).to_vec()[0][start1], 1);
                assert_eq!(m.borrow(py).to_vec()[1][start1 + 1], 1);

                // row_segment_mut writes through to the inner matrix.
                let row = PyAugmentedMatrix2::row_segment_mut(m.borrow(py), 0, 0, 0).unwrap();
                row.set_entry(py, 0, 1).unwrap();
                assert_eq!(m.borrow(py).to_vec()[0][0], 1);
            });
        }

        #[test]
        fn augmented_matrix2_compute_nontrivial_values() {
            // This Rust-side test builds the non-trivial example directly with
            // the upstream API; the equivalent value test driven entirely
            // through the Python segment-mut API lives in
            // `tests/test_matrix_slice_mut.py`.
            //
            // This mirrors the F3 doctest of `Matrix::compute_image` /
            // `Matrix::compute_quasi_inverse` (matrix_inner.rs), with the same
            // rank-2 input A and the committed expected image and preimage. We
            // build [A | I] as an `AugmentedMatrix<2>` with segment widths
            // [5, 3] (A is 3x5, I is 3x3) so the augmented wrappers exercise the
            // same code path as the bare-matrix doctest.
            let p = valid_prime(3).unwrap();
            let a = [
                vec![1, 2, 1, 1, 0],
                vec![1, 0, 2, 1, 1],
                vec![2, 2, 0, 2, 1],
            ];
            let mut aug = RustAugmentedMatrix::<2>::new(p, 3, [5, 3]);
            for (i, row) in a.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    aug.inner.row_mut(i).set_entry(j, v);
                }
            }
            aug.segment(1, 1).add_identity();
            aug.row_reduce();

            let m2 = PyAugmentedMatrix2(aug);

            // Image basis and dimension exactly as in the upstream doctest:
            // image = [[1,0,2,1,1],[0,1,1,0,1]], a 2-dim subspace of F3^5.
            let image = m2.compute_image().unwrap();
            assert_eq!(image.dimension(), 2);
            let image_rows: Vec<Vec<u32>> =
                image.iter().iter().map(|v| v.0.iter().collect()).collect();
            assert_eq!(image_rows, vec![vec![1, 0, 2, 1, 1], vec![0, 1, 1, 0, 1]]);

            // Quasi-inverse preimage exactly as in the upstream doctest:
            // preimage = [[0,1,0],[0,2,2]]; source dimension is 3 (the I block).
            let qi = m2.compute_quasi_inverse().unwrap();
            assert_eq!(qi.source_dimension(), 3);
            assert_eq!(qi.preimage().to_vec(), vec![vec![0, 1, 0], vec![0, 2, 2]]);
        }
    }
}
