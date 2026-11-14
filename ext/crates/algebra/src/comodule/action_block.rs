use fp::prime::{Prime, ValidPrime};

/// An operation of an algebra, by its degree and its index in the basis of that degree.
///
/// This is how a free module names the operation in a term $P(R) x$, so a resolution's
/// differentials can look up their action without translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId {
    pub degree: i32,
    pub index: u32,
}

/// A set of operations with a dense numbering, sorted by degree and then index.
///
/// The numbering lets an [`ActionBlock`] name an operation by a small integer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpSet {
    ops: Vec<OpId>,
}

impl OpSet {
    /// The number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The operation numbered `local`.
    pub fn get(&self, local: usize) -> OpId {
        self.ops[local]
    }

    /// The number of `op`, if it is in the set.
    pub fn position(&self, op: OpId) -> Option<usize> {
        self.ops.binary_search(&op).ok()
    }

    /// The operations, in the order of their numbers.
    pub fn iter(&self) -> impl Iterator<Item = OpId> + '_ {
        self.ops.iter().copied()
    }

    /// The operations of degree at most `degree`, which are the only ones that can act on a
    /// basis element of that degree.
    pub fn up_to_degree(&self, degree: i32) -> impl Iterator<Item = (usize, OpId)> + '_ {
        let end = self.ops.partition_point(|op| op.degree <= degree);
        self.ops[..end].iter().copied().enumerate()
    }
}

impl FromIterator<OpId> for OpSet {
    fn from_iter<I: IntoIterator<Item = OpId>>(iter: I) -> Self {
        let mut ops: Vec<OpId> = iter.into_iter().collect();
        ops.sort_unstable();
        ops.dedup();
        Self { ops }
    }
}

/// The action of each operation of an [`OpSet`] on each basis element of one degree of a
/// comodule, as sparse vectors.
///
/// Stored row by row: for each basis element, the operations that act on it nontrivially, in
/// increasing order of their numbers, and for each of those the basis elements it reaches.
/// At $p = 2$ every coefficient is one and none is stored.
#[derive(Clone, Debug)]
pub struct ActionBlock {
    prime: ValidPrime,
    /// `row_starts[i]..row_starts[i + 1]` indexes the operations acting on basis element `i`.
    row_starts: Vec<u32>,
    /// The number, in the [`OpSet`], of each operation acting.
    ops: Vec<u32>,
    /// `op_starts[k]..op_starts[k + 1]` indexes the targets of the `k`th entry of `ops`.
    op_starts: Vec<u32>,
    /// The basis elements reached, increasing within each operation.
    targets: Vec<u32>,
    /// The coefficient of each target; empty at $p = 2$.
    coeffs: Vec<u32>,
}

impl ActionBlock {
    /// An empty block, to be filled with [`Self::push_row`].
    pub fn new(prime: ValidPrime) -> Self {
        Self {
            prime,
            row_starts: vec![0],
            ops: Vec::new(),
            op_starts: vec![0],
            targets: Vec::new(),
            coeffs: Vec::new(),
        }
    }

    /// Appends the row of the next basis element, given as `(op, target, coeff)` triples sorted by
    /// `op` and then `target`, with nonzero coefficients.
    pub fn push_row(&mut self, entries: impl IntoIterator<Item = (u32, u32, u32)>) {
        let odd = self.prime != 2;
        let row_start = self.ops.len();
        for (op, target, coeff) in entries {
            debug_assert!(coeff != 0 && coeff < self.prime.as_u32());
            if self.ops.len() == row_start || self.ops.last() != Some(&op) {
                self.ops.push(op);
                self.op_starts.push(self.targets.len() as u32);
            }
            self.targets.push(target);
            if odd {
                self.coeffs.push(coeff);
            }
            *self.op_starts.last_mut().unwrap() = self.targets.len() as u32;
        }
        self.row_starts.push(self.ops.len() as u32);
    }

    /// The number of rows, which is the dimension of the degree.
    pub fn rows(&self) -> usize {
        self.row_starts.len() - 1
    }

    /// The operations acting nontrivially on basis element `idx`, each with the basis elements it
    /// reaches and their coefficients.
    pub fn row(&self, idx: usize) -> impl Iterator<Item = (u32, ActionEntry<'_>)> + '_ {
        let (start, end) = (
            self.row_starts[idx] as usize,
            self.row_starts[idx + 1] as usize,
        );
        (start..end).map(move |k| (self.ops[k], self.entry(k)))
    }

    /// The action of the operation numbered `op` on basis element `idx`.
    pub fn act(&self, idx: usize, op: u32) -> ActionEntry<'_> {
        let (start, end) = (
            self.row_starts[idx] as usize,
            self.row_starts[idx + 1] as usize,
        );
        match self.ops[start..end].binary_search(&op) {
            Ok(k) => self.entry(start + k),
            Err(_) => ActionEntry {
                targets: &[],
                coeffs: None,
            },
        }
    }

    /// The `k`th stored operation's targets.
    fn entry(&self, k: usize) -> ActionEntry<'_> {
        let range = self.op_starts[k] as usize..self.op_starts[k + 1] as usize;
        ActionEntry {
            targets: &self.targets[range.clone()],
            coeffs: (self.prime != 2).then(|| &self.coeffs[range]),
        }
    }

    /// The block of the rows in `rows` and the targets in `targets`, renumbered from the start of
    /// each range.
    pub fn restrict(&self, rows: std::ops::Range<usize>, targets: std::ops::Range<usize>) -> Self {
        let mut out = Self::new(self.prime);
        for idx in rows {
            let row: Vec<(u32, u32, u32)> = self
                .row(idx)
                .flat_map(|(op, entry)| {
                    entry
                        .iter()
                        .filter(|(t, _)| targets.contains(t))
                        .map(|(t, c)| (op, (t - targets.start) as u32, c))
                        .collect::<Vec<_>>()
                })
                .collect();
            out.push_row(row);
        }
        out
    }

    /// The number of stored `(operation, target)` pairs, a measure of the block's size.
    pub fn nnz(&self) -> usize {
        self.targets.len()
    }
}

/// The image of one basis element under one operation, as a sparse vector.
#[derive(Clone, Copy, Debug)]
pub struct ActionEntry<'a> {
    targets: &'a [u32],
    coeffs: Option<&'a [u32]>,
}

impl ActionEntry<'_> {
    /// The basis elements reached and their coefficients.
    pub fn iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.targets
            .iter()
            .enumerate()
            .map(|(i, &t)| (t as usize, self.coeffs.map_or(1, |c| c[i])))
    }

    /// Whether the image is zero.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}
