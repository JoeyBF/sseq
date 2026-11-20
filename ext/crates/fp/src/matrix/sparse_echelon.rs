use std::collections::BinaryHeap;

/// A column echelon form over $\mathbb{F}_2$ of sparse columns, each stored as its sorted row
/// indices.
///
/// The pivot of a column is its largest row index, and no two columns share a pivot. A column is
/// reduced only at its leading entry: entries below the pivot are left as they fall, which keeps
/// columns short. The caller chooses the elimination order through the numbering of the rows and
/// the order of insertion, and both mean something:
///
/// * a row numbered later is preferred as a pivot;
/// * a column is only ever reduced by columns inserted before it.
///
/// With rows and columns numbered by a filtration that the matrix respects, this is the matrix
/// reduction of persistent homology: the pivot of a reduced column is where the cycle it holds
/// first appears, and its position in the insertion order is where that cycle becomes a boundary.
pub struct SparseEchelon {
    /// The column whose pivot is each row, or [`NO_COLUMN`].
    pivot_column: Vec<u32>,
    columns: Vec<Box<[u32]>>,
    entries: usize,
}

/// Marks a row that is not a pivot.
const NO_COLUMN: u32 = u32::MAX;

/// The scratch space of [`SparseEchelon::reduce`]: a dense accumulator and the candidates for its
/// largest entry.
///
/// One workspace serves any number of reductions on echelon forms of at most its number of rows,
/// and leaves no state between them.
pub struct SparseWorkspace {
    bits: Vec<u64>,
    /// Every row toggled since the last reduction, possibly more than once, possibly cleared again.
    heap: BinaryHeap<u32>,
}

impl SparseWorkspace {
    /// A workspace for echelon forms of at most `rows` rows.
    pub fn new(rows: usize) -> Self {
        Self {
            bits: vec![0; rows.div_ceil(64)],
            heap: BinaryHeap::new(),
        }
    }

    /// Grows to hold `rows` rows.
    pub fn reserve(&mut self, rows: usize) {
        if self.bits.len() < rows.div_ceil(64) {
            self.bits.resize(rows.div_ceil(64), 0);
        }
    }

    /// Adds the basis vector of `row`.
    fn toggle(&mut self, row: u32) {
        let (word, bit) = (row as usize / 64, row % 64);
        self.bits[word] ^= 1 << bit;
        if self.bits[word] >> bit & 1 == 1 {
            self.heap.push(row);
        }
    }

    /// Whether `row` is set.
    fn get(&self, row: u32) -> bool {
        self.bits[row as usize / 64] >> (row % 64) & 1 == 1
    }

    /// Clears `row`.
    fn clear(&mut self, row: u32) {
        self.bits[row as usize / 64] &= !(1 << (row % 64));
    }
}

impl SparseEchelon {
    /// An empty echelon form on `rows` rows.
    pub fn new(rows: usize) -> Self {
        assert!(rows < NO_COLUMN as usize);
        Self {
            pivot_column: vec![NO_COLUMN; rows],
            columns: Vec::new(),
            entries: 0,
        }
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.pivot_column.len()
    }

    /// The number of columns, which is the rank.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether there are no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The number of stored entries over every column.
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// The column with pivot `row`, if any.
    pub fn pivot_column(&self, row: usize) -> Option<usize> {
        let c = self.pivot_column[row];
        (c != NO_COLUMN).then_some(c as usize)
    }

    /// The sorted row indices of column `id`; its pivot is the last.
    pub fn column(&self, id: usize) -> &[u32] {
        &self.columns[id]
    }

    /// Reduces the vector with the given entries (in any order; a repeated row cancels) and
    /// returns what is left, sorted. `used`, if given, receives the columns that were added.
    ///
    /// The result is empty exactly when the vector is in the span of the columns. Otherwise its
    /// last entry is not a pivot, and [`Self::insert`] accepts it.
    pub fn reduce(
        &self,
        ws: &mut SparseWorkspace,
        entries: impl IntoIterator<Item = u32>,
        mut used: Option<&mut Vec<u32>>,
    ) -> Vec<u32> {
        ws.reserve(self.rows());
        for row in entries {
            ws.toggle(row);
        }
        let mut rest = Vec::new();
        while let Some(row) = ws.heap.pop() {
            if !ws.get(row) {
                continue;
            }
            let c = self.pivot_column[row as usize];
            if c == NO_COLUMN {
                // The largest entry is no pivot: collect the remainder, which empties the
                // workspace. A stale duplicate of a collected row finds its bit cleared.
                rest.push(row);
                ws.clear(row);
                while let Some(row) = ws.heap.pop() {
                    if ws.get(row) {
                        rest.push(row);
                        ws.clear(row);
                    }
                }
                rest.reverse();
                break;
            }
            if let Some(used) = used.as_deref_mut() {
                used.push(c);
            }
            for &r in self.columns[c as usize].iter() {
                ws.toggle(r);
            }
        }
        rest
    }

    /// Appends a reduced, nonempty column and returns its id.
    ///
    /// # Panics
    ///
    /// If its largest entry is already a pivot.
    pub fn insert(&mut self, column: Vec<u32>) -> usize {
        let pivot = *column.last().expect("a nonzero column") as usize;
        assert_eq!(
            self.pivot_column[pivot], NO_COLUMN,
            "row {pivot} is a pivot"
        );
        let id = self.columns.len();
        self.pivot_column[pivot] = id as u32;
        self.entries += column.len();
        self.columns.push(column.into_boxed_slice());
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix::Matrix, prime::TWO};

    /// A deterministic stream of pseudorandom words.
    fn stream(mut x: u64) -> impl FnMut() -> u64 {
        move || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// The rank agrees with dense row reduction, and every column reduces to zero afterwards.
    #[test]
    fn rank_matches_dense() {
        let mut next = stream(7);
        for (rows, cols, weight) in [(40, 60, 3), (200, 150, 4), (300, 300, 2), (64, 64, 20)] {
            let columns: Vec<Vec<u32>> = (0..cols)
                .map(|_| (0..weight).map(|_| (next() % rows as u64) as u32).collect())
                .collect();
            let mut dense = Matrix::new(TWO, cols, rows);
            for (c, column) in columns.iter().enumerate() {
                for &r in column {
                    dense.row_mut(c).add_basis_element(r as usize, 1);
                }
            }
            dense.row_reduce();
            let rank = dense.pivots().iter().filter(|&&p| p >= 0).count();

            let mut echelon = SparseEchelon::new(rows);
            let mut ws = SparseWorkspace::new(rows);
            for column in &columns {
                let rest = echelon.reduce(&mut ws, column.iter().copied(), None);
                if !rest.is_empty() {
                    echelon.insert(rest);
                }
            }
            assert_eq!(echelon.len(), rank);
            for column in &columns {
                let mut used = Vec::new();
                let rest = echelon.reduce(&mut ws, column.iter().copied(), Some(&mut used));
                assert!(rest.is_empty());
                // the columns used add up to the vector
                let mut sum = std::collections::BTreeSet::new();
                for &c in &used {
                    for &r in echelon.column(c as usize) {
                        if !sum.remove(&r) {
                            sum.insert(r);
                        }
                    }
                }
                let mut expected = std::collections::BTreeSet::new();
                for &r in column {
                    if !expected.remove(&r) {
                        expected.insert(r);
                    }
                }
                assert_eq!(sum, expected);
            }
        }
    }
}
