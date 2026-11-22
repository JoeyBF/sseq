//! Measures how the order of the cochains within a level changes the work of the tower reduction of Cτ.

use std::time::Instant;

use algebra::AlgebraType;
use ext::{
    chain_complex::ChainComplex,
    ctau::{OpTable, QAlgebra, QMono},
    ctau_scale::{
        Complex, OpTables, ScaledExt, Space, StreamedBackbone, needed_operations, pullback_columns,
    },
    utils::query_module_only,
};
use fp::matrix::{SparseEchelon, SparseWorkspace};
use maybe_rayon::prelude::*;

/// An order of the cochains of one level: its name, and the key a cochain on the generator of
/// degree `t` with monomial `m` at position `pos` is sorted by. A cochain sorted later is
/// preferred as a pivot and taken later as a source.
type Order = (&'static str, fn(i64, &QMono, i64) -> Vec<i64>);

/// The exponents of `m` from $q_1$ up, each times `sign`.
fn lex(m: &QMono, sign: i64) -> Vec<i64> {
    m.iter().map(|&e| sign * e as i64).collect()
}

/// The orders compared. The first is the one [`Space`] numbers by.
const ORDERS: [Order; 10] = [
    ("generator degree down (as built)", |_, _, pos| vec![pos]),
    ("generator degree up", |t, _, pos| vec![t, pos]),
    (
        "q1 exponent down, then generator degree down",
        |t, m, pos| vec![-(m[0] as i64), -t, pos],
    ),
    ("q1 exponent up, then generator degree down", |t, m, pos| {
        vec![m[0] as i64, -t, pos]
    }),
    (
        "generator degree down, then q1 exponent down",
        |t, m, pos| vec![-t, -(m[0] as i64), pos],
    ),
    ("generator degree down, then q1 exponent up", |t, m, pos| {
        vec![-t, m[0] as i64, pos]
    }),
    (
        "exponents of q1, q2, ... down, then generator degree down",
        |t, m, pos| {
            let mut key = lex(m, -1);
            key.extend([-t, pos]);
            key
        },
    ),
    (
        "exponents of q1, q2, ... up, then generator degree down",
        |t, m, pos| {
            let mut key = lex(m, 1);
            key.extend([-t, pos]);
            key
        },
    ),
    (
        "generator degree down, then exponents of q1, q2, ... down",
        |t, m, pos| {
            let mut key = vec![-t];
            key.extend(lex(m, -1));
            key.push(pos);
            key
        },
    ),
    (
        "generator degree down, then exponents of q1, q2, ... up",
        |t, m, pos| {
            let mut key = vec![-t];
            key.extend(lex(m, 1));
            key.push(pos);
            key
        },
    ),
];

/// The rank of each position of `space` under `order`, level first.
fn ranks(q: &QAlgebra, space: &Space, order: &Order) -> Vec<u32> {
    let mut keyed: Vec<(u16, Vec<i64>, u32)> = (0..space.dim() as u32)
        .map(|pos| {
            let (_, t, _, m, level) = space.decode(q, pos);
            (level, (order.1)(t as i64, &m, pos as i64), pos)
        })
        .collect();
    keyed.sort_unstable();
    let mut rank = vec![0; space.dim()];
    for (r, (_, _, pos)) in keyed.into_iter().enumerate() {
        rank[pos as usize] = r as u32;
    }
    rank
}

/// What one order cost: the entries added, the reduced entries kept, the pairs across levels
/// (the towers, which no order within a level changes) and the columns reduced.
#[derive(Debug, Default, Clone, Copy)]
struct Cost {
    additions: usize,
    kept: usize,
    towers: usize,
    columns: usize,
    input: usize,
    /// Columns that needed no addition.
    apparent: usize,
    /// The entries of the functionals the transposed reduction keeps.
    psi: usize,
    /// The reduced rows that were added to a later one, and their entries.
    reused: usize,
    reused_entries: usize,
    /// The entries of the rows at cochains of the level of the row itself, and up to the
    /// level of its pivot: what finding the pair reads of a row that is reduced as it stands.
    level_entries: usize,
    paired_entries: usize,
    /// The entries of the rows by the number of levels between the row and the source, the
    /// last for [`DROPS`] and more.
    by_drop: [usize; DROPS + 1],
    /// The entries within as many levels of their row as the longest tower of their space:
    /// what building every row of a space to one depth, just deep enough, reads.
    uniform_entries: usize,
}

/// The level differences counted one by one in [`Cost::by_drop`].
const DROPS: usize = 6;

impl Cost {
    fn merge(self, o: Self) -> Self {
        Self {
            additions: self.additions + o.additions,
            kept: self.kept + o.kept,
            towers: self.towers + o.towers,
            columns: self.columns + o.columns,
            input: self.input + o.input,
            apparent: self.apparent + o.apparent,
            psi: self.psi + o.psi,
            reused: self.reused + o.reused,
            reused_entries: self.reused_entries + o.reused_entries,
            level_entries: self.level_entries + o.level_entries,
            paired_entries: self.paired_entries + o.paired_entries,
            by_drop: std::array::from_fn(|d| self.by_drop[d] + o.by_drop[d]),
            uniform_entries: self.uniform_entries + o.uniform_entries,
        }
    }
}

/// Reduces the coboundaries of `cx` at weight `u` out of filtrations `a_lo..a_hi` with the
/// cochains of each level in the given order, one column at a time.
#[allow(clippy::too_many_arguments)]
fn reduce(
    q: &QAlgebra,
    bb: &StreamedBackbone,
    tables: &OpTables,
    cx: &Complex,
    u: i32,
    a_lo: i32,
    a_hi: i32,
    order: &Order,
) -> Cost {
    let mut cost = Cost::default();
    let mut src = Space::new(bb, q, cx, a_lo, u);
    let mut src_rank = ranks(q, &src, order);
    // the positions of `src` that are pivots of the coboundary into it
    let mut cleared = vec![false; src.dim()];
    for a in a_lo..a_hi {
        let tgt = Space::new(bb, q, cx, a + 1, u);
        let tgt_rank = ranks(q, &tgt, order);
        let mut columns = pullback_columns(q, bb, tables, cx, &src, &tgt, 0, |pos| {
            cleared[pos as usize]
        });
        columns.sort_unstable_by_key(|c| src_rank[c.0 as usize]);
        let mut level_of = vec![0u16; tgt.dim()];
        let mut pos_of = vec![0u32; tgt.dim()];
        for pos in 0..tgt.dim() as u32 {
            level_of[tgt_rank[pos as usize] as usize] = tgt.level(pos);
            pos_of[tgt_rank[pos as usize] as usize] = pos;
        }
        let mut echelon = SparseEchelon::new(tgt.dim());
        let mut ws = SparseWorkspace::new(tgt.dim());
        let mut next_cleared = vec![false; tgt.dim()];
        for (pos, column) in columns {
            cost.input += column.len();
            cost.columns += 1;
            let rest = echelon.reduce(&mut ws, column.iter().map(|&r| tgt_rank[r as usize]), None);
            cost.additions += ws.take_additions();
            if let Some(&pivot) = rest.last() {
                cost.kept += rest.len();
                cost.towers += usize::from(level_of[pivot as usize] < src.level(pos));
                next_cleared[pos_of[pivot as usize] as usize] = true;
                echelon.insert(rest);
            }
        }
        src = tgt;
        src_rank = tgt_rank;
        cleared = next_cleared;
    }
    cost
}

/// What the transposed reduction of one coboundary $C^a \to C^{a+1}$ leaves.
struct DualStep {
    a: i32,
    /// The reduced rows: functionals on $C^a$ that vanish on cocycles, as positions counted
    /// from the end, so that the first position is the pivot [`SparseEchelon`] sees.
    rows: Vec<Vec<u32>>,
    /// For each row, the cochain $\tau$ of $C^{a+1}$ it started from and the functional on
    /// $C^{a+1}$ it is the coboundary of: $\tau$ and cochains after it.
    psi: Vec<(u32, Vec<u32>)>,
    /// The rows that are towers, with the levels at which each is born and dies.
    towers: Vec<(u32, u16, u16)>,
}

/// The same reduction on the transposed matrices, from the highest filtration down: the columns
/// are the cochains of the target, taken from the last to the first, and the pivot of a column
/// is its first source. The pairs are those of [`reduce`]; the work is not.
fn reduce_dual(
    q: &QAlgebra,
    bb: &StreamedBackbone,
    tables: &OpTables,
    cx: &Complex,
    u: i32,
    a_lo: i32,
    a_hi: i32,
) -> (Cost, Vec<DualStep>) {
    let mut cost = Cost::default();
    let mut steps = Vec::new();
    // the positions of the target that are pivots of the transposed coboundary out of it
    let mut cleared: Option<Vec<bool>> = None;
    for a in (a_lo..a_hi).rev() {
        let src = Space::new(bb, q, cx, a, u);
        let tgt = Space::new(bb, q, cx, a + 1, u);
        let columns = pullback_columns(q, bb, tables, cx, &src, &tgt, 0, |_| false);
        let flip = |pos: u32| src.dim() as u32 - 1 - pos;
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); tgt.dim()];
        for (pos, column) in columns {
            for r in column {
                // a repeated row cancels in the reduction
                rows[r as usize].push(flip(pos));
            }
        }
        let mut echelon = SparseEchelon::new(src.dim());
        let mut ws = SparseWorkspace::new(src.dim());
        let mut next_cleared = vec![false; src.dim()];
        // the reduced rows that have been added to another
        let mut reused: Vec<bool> = Vec::new();
        let mut step = DualStep {
            a,
            rows: Vec::new(),
            psi: Vec::new(),
            towers: Vec::new(),
        };
        // the entries of this coboundary by level difference, and its longest finite tower
        let mut drops: Vec<usize> = Vec::new();
        let mut longest = 0;
        for pos in (0..tgt.dim() as u32).rev() {
            if cleared.as_ref().is_some_and(|c| c[pos as usize]) {
                continue;
            }
            let row = std::mem::take(&mut rows[pos as usize]);
            cost.input += row.len();
            cost.columns += 1;
            // the entries of the row at levels up to `level`, as built
            let upto = |level: usize| {
                let from = (src.dim() - src.level_end(level)) as u32;
                row.iter().filter(|&&r| r >= from).count()
            };
            let own = upto(tgt.level(pos) as usize);
            let all = row.len();
            for &r in &row {
                let drop = (src.level(flip(r)) - tgt.level(pos)) as usize;
                if drops.len() <= drop {
                    drops.resize(drop + 1, 0);
                }
                drops[drop] += 1;
            }
            let mut used = Vec::new();
            let rest = echelon.reduce(&mut ws, row.iter().copied(), Some(&mut used));
            let added = ws.take_additions();
            cost.additions += added;
            cost.apparent += usize::from(added == 0);
            let mut psi = std::collections::BTreeSet::from([pos]);
            for &c in &used {
                if !std::mem::replace(&mut reused[c as usize], true) {
                    cost.reused += 1;
                    cost.reused_entries += step.rows[c as usize].len();
                }
            }
            for c in used {
                for &r in &step.psi[c as usize].1 {
                    if !psi.remove(&r) {
                        psi.insert(r);
                    }
                }
            }
            cost.level_entries += own;
            let Some(&pivot) = rest.last() else {
                // only in the lowest filtration, where a cocycle need not be a coboundary
                cost.paired_entries += all;
                continue;
            };
            cost.paired_entries += upto(src.level(flip(pivot)) as usize);
            cost.kept += rest.len();
            cost.psi += psi.len();
            let (birth, death) = (tgt.level(pos), src.level(flip(pivot)));
            longest = longest.max((death - birth) as usize);
            if birth < death {
                cost.towers += 1;
                step.towers.push((step.rows.len() as u32, birth, death));
            }
            next_cleared[flip(pivot) as usize] = true;
            echelon.insert(rest.clone());
            step.rows.push(rest);
            reused.push(false);
            step.psi.push((pos, psi.into_iter().collect()));
        }
        for (d, &n) in drops.iter().enumerate() {
            cost.by_drop[d.min(DROPS)] += n;
            if d <= longest {
                cost.uniform_entries += n;
            }
        }
        cleared = Some(next_cleared);
        steps.push(step);
    }
    (cost, steps)
}

/// The cost of $h_{j+1}$ on the towers at weight `u + 2^j`, computed on functionals: each tower
/// functional is pulled back to weight `u` and written in the functionals there, which form a
/// triangular basis (the reduced rows out of a filtration and the functionals into it).
///
/// Returns the entries added, the entries of the functionals pulled back, the towers and the
/// nonzero coefficients on towers.
#[allow(clippy::too_many_arguments)]
fn dual_products(
    q: &QAlgebra,
    bb: &StreamedBackbone,
    tables: &OpTables,
    cx: &Complex,
    u: i32,
    j: u32,
    low: &[DualStep],
    high: &[DualStep],
) -> [usize; 4] {
    let mut out = [0; 4];
    for top in high {
        // towers of C^{a+1} at the higher weight; their sources are in C^a at the lower
        let a = top.a;
        let (Some(into), Some(from)) = (
            low.iter().find(|s| s.a == a - 1),
            low.iter().find(|s| s.a == a),
        ) else {
            continue;
        };
        let src = Space::new(bb, q, cx, a, u);
        let tgt = Space::new(bb, q, cx, a + 1, u + (1 << j));
        let flip = |pos: u32| src.dim() as u32 - 1 - pos;
        // every position of C^a is the pivot of a reduced row out of it or starts a functional
        let mut basis = SparseEchelon::new(src.dim());
        for row in &from.rows {
            basis.insert(row.clone());
        }
        let first = basis.len();
        for (_, psi) in &into.psi {
            let mut v: Vec<u32> = psi.iter().map(|&p| flip(p)).collect();
            v.reverse();
            basis.insert(v);
        }
        let mut is_tower = vec![false; into.psi.len()];
        for &(row, _, _) in &into.towers {
            is_tower[row as usize] = true;
        }
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); tgt.dim()];
        for (pos, column) in pullback_columns(q, bb, tables, cx, &src, &tgt, 1 << j, |_| false) {
            for r in column {
                rows[r as usize].push(flip(pos));
            }
        }
        // Only the coefficients on towers alive where the pulled back tower is are wanted, and
        // they depend on the cochains below the level at which it dies alone: later positions
        // are dropped, from the functional and from every row added.
        let mut bits = vec![0u64; src.dim().div_ceil(64)];
        for &(row, _, death) in &top.towers {
            let psi = &top.psi[row as usize].1;
            // positions of level below `death`, counted from the end
            let cut = flip(0) + 1 - src.dim().min(src.level_end(death as usize - 1)) as u32;
            let mut count = 0;
            for &p in psi {
                for &r in &rows[p as usize] {
                    if r >= cut {
                        bits[r as usize / 64] ^= 1 << (r % 64);
                        count += 1;
                    }
                }
            }
            out[1] += count;
            out[2] += 1;
            let mut w = bits.len();
            while w > cut as usize / 64 {
                let word = bits[w - 1];
                if word == 0 {
                    w -= 1;
                    continue;
                }
                let r = (w - 1) * 64 + 63 - word.leading_zeros() as usize;
                if (r as u32) < cut {
                    break;
                }
                let c = basis.pivot_column(r).expect("a basis");
                let column = basis.column(c);
                let from = column.partition_point(|&x| x < cut);
                out[0] += column.len() - from;
                for &x in &column[from..] {
                    bits[x as usize / 64] ^= 1 << (x % 64);
                }
                if c >= first && is_tower[c - first] {
                    out[3] += 1;
                }
            }
            for word in &mut bits[cut as usize / 64..] {
                *word = 0;
            }
        }
    }
    out
}

/// Reads the save and a stem, and prints the cost of each of [`ORDERS`] on Cτ (or, with a
/// complex named, on Cτ smashed with a Hopf cofiber) through that stem.
fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = query_module_only("Module", Some(AlgebraType::Milnor), false)?;
    let max_n: i32 = query::with_default("Max motivic stem", "120", str::parse);
    let name: String = query::with_default("Complex (Ctau, Ceta, Cnu, Csigma)", "Ctau", |s| {
        Ok::<_, anyhow::Error>(s.to_string())
    });
    let count: usize = query::with_default("Orders to compare, from the first", "10", str::parse);
    let cx = match name.as_str() {
        "Ctau" => Complex::sphere(),
        "Ceta" => Complex::hopf("Ceta", 0),
        "Cnu" => Complex::hopf("Cnu", 1),
        "Csigma" => Complex::hopf("Csigma", 2),
        _ => anyhow::bail!("unknown complex {name}"),
    };
    let store = res.save_dir().store().expect("a saved resolution");
    let algebra = res.algebra();
    let ops = OpTable::new(algebra.clone());
    let (max_a, max_t) = ScaledExt::backbone_range(max_n);
    let bb = StreamedBackbone::read(store, &algebra, &ops, max_a, &max_t)?;
    let q_degree = max_n / 2 + 2;
    let q = QAlgebra::new(q_degree, q_degree as u32);
    let shifts: [u128; 3] = [1, 2, 4];
    let tables = OpTables::new(
        &q,
        Some(&needed_operations(
            bb.operations(ScaledExt::reach(max_n)),
            &shifts,
        )),
        q_degree,
    );
    let weights: Vec<i32> = (0..=max_n).collect();
    for order in ORDERS.iter().take(count) {
        let start = Instant::now();
        let cost = weights
            .clone()
            .into_maybe_par_iter()
            .map(|u| {
                let a_lo = (2 * u - max_n - 1).max(cx.min_a());
                let a_hi = u.max(a_lo + 1);
                reduce(&q, &bb, &tables, &cx, u, a_lo, a_hi, order)
            })
            .reduce(Cost::default, Cost::merge);
        println!(
            "{}: {} columns of {} entries, {} entries added, {} kept, {} towers: {:.1} s",
            order.0,
            cost.columns,
            cost.input,
            cost.additions,
            cost.kept,
            cost.towers,
            start.elapsed().as_secs_f64()
        );
    }
    let start = Instant::now();
    let chains: Vec<(Cost, Vec<DualStep>)> = weights
        .clone()
        .into_maybe_par_iter()
        .map(|u| {
            let a_lo = (2 * u - max_n - 1).max(cx.min_a());
            let a_hi = u.max(a_lo + 1);
            reduce_dual(&q, &bb, &tables, &cx, u, a_lo, a_hi)
        })
        .collect();
    let cost = chains
        .iter()
        .map(|c| c.0)
        .fold(Cost::default(), Cost::merge);
    println!(
        "transposed, from the top filtration down: {} columns of {} entries, {} entries added, {} \
         kept, {} towers; {} columns needed no addition, the functionals have {} entries, {} \
         rows of {} entries were added to a later one; {} entries are at the level of their \
         row and {} up to the level of its pivot; by level difference {:?}, and {} within the \
         longest tower of their space: {:.1} s",
        cost.columns,
        cost.input,
        cost.additions,
        cost.kept,
        cost.towers,
        cost.apparent,
        cost.psi,
        cost.reused,
        cost.reused_entries,
        cost.level_entries,
        cost.paired_entries,
        cost.by_drop,
        cost.uniform_entries,
        start.elapsed().as_secs_f64()
    );
    for j in 0..3u32 {
        let start = Instant::now();
        let total = weights
            .clone()
            .into_maybe_par_iter()
            .filter(|&u| u + (1 << j) <= max_n)
            .map(|u| {
                let (low, high) = (&chains[u as usize].1, &chains[(u + (1 << j)) as usize].1);
                dual_products(&q, &bb, &tables, &cx, u, j, low, high)
            })
            .reduce(
                || [0; 4],
                |x, y| [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]],
            );
        println!(
            "h_{} on functionals: {} towers pulled back as {} entries, {} entries added, {} \
             nonzero coefficients: {:.1} s",
            j + 1,
            total[2],
            total[1],
            total[0],
            total[3],
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
