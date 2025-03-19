use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use once::{
    kdtrie::{grove::TwoEndedGrove, MultiGraded},
    OnceBiVec, SparseKVec, SparseVec,
};
use rand::{rng, seq::SliceRandom};

/// A trait that matches OnceBiVec's semantics for benchmarking
trait Benchable<const K: usize, T> {
    fn name() -> &'static str;

    /// Create a new container with the given minimum bounds
    fn new(min: [i32; K]) -> Self;

    /// Push a value at the given coordinates, filling in any gaps with cloned values
    /// The coordinates must be pushed in order within each dimension
    fn push_checked(&self, coords: [i32; K], value: T);

    /// Get a value at the given coordinates if it exists and is within bounds
    fn get(&self, coords: [i32; K]) -> Option<&T>;

    /// Get all values within a given range (inclusive)
    fn range_query(&self, min: [i32; K], max: [i32; K]) -> Vec<&T>;
}

impl<T> Benchable<1, T> for SparseVec<T> {
    fn name() -> &'static str {
        "sparse_vec"
    }

    fn new(_min: [i32; 1]) -> Self {
        SparseVec::new()
    }

    fn push_checked(&self, coords: [i32; 1], value: T) {
        self.insert(coords[0], value);
    }

    fn get(&self, coords: [i32; 1]) -> Option<&T> {
        self.get(coords[0])
    }

    fn range_query(&self, min: [i32; 1], max: [i32; 1]) -> Vec<&T> {
        self.range(min[0], max[0]).map(|(_, v)| v).collect()
    }
}

impl<T, const K: usize> Benchable<K, T> for SparseKVec<T, K> {
    fn name() -> &'static str {
        "sparse_kvec"
    }

    fn new(_min: [i32; K]) -> Self {
        SparseKVec::new()
    }

    fn push_checked(&self, coords: [i32; K], value: T) {
        self.insert(coords, value);
    }

    fn get(&self, coords: [i32; K]) -> Option<&T> {
        self.get(coords)
    }

    fn range_query(&self, min: [i32; K], max: [i32; K]) -> Vec<&T> {
        self.range(min, max).map(|(_, v)| v).collect()
    }
}

impl<T, const K: usize> Benchable<K, T> for MultiGraded<K, T> {
    fn name() -> &'static str {
        "multi_graded"
    }

    fn new(_min: [i32; K]) -> Self {
        Self::new()
    }

    fn push_checked(&self, coords: [i32; K], value: T) {
        self.insert(coords, value);
    }

    fn get(&self, coords: [i32; K]) -> Option<&T> {
        self.get(coords)
    }

    fn range_query(&self, min: [i32; K], max: [i32; K]) -> Vec<&T> {
        todo!()

        //     let mut result = Vec::new();
        //     for i in min[0]..=max[0] {
        //         for j in min[1]..=max[1] {
        //             for k in min[2]..=max[2] {
        //                 let coord = [i, j, k];
        //                 if let Some(value) = self.get(coord) {
        //                     result.push(value);
        //                 }
        //             }
        //         }
        //     }
        //     result
    }
}

impl<T> Benchable<1, T> for TwoEndedGrove<T> {
    fn name() -> &'static str {
        "grove"
    }

    fn new(_min: [i32; 1]) -> Self {
        Self::new()
    }

    fn push_checked(&self, coords: [i32; 1], value: T) {
        self.insert(coords[0], value);
    }

    fn get(&self, coords: [i32; 1]) -> Option<&T> {
        self.get(coords[0])
    }

    fn range_query(&self, min: [i32; 1], max: [i32; 1]) -> Vec<&T> {
        todo!()
        // self.range(min[0], max[0]).map(|(_, v)| v).collect()
    }
}

mod benchable_oncebivec {
    use super::*;

    const ONCEBIVEC_INIT_DEPTH: i32 = 0;

    impl<T> Benchable<1, T> for OnceBiVec<T> {
        fn name() -> &'static str {
            "oncebivec"
        }

        fn new(min: [i32; 1]) -> Self {
            OnceBiVec::new(min[0])
        }

        fn push_checked(&self, coords: [i32; 1], value: T) {
            self.push_checked(value, coords[0]);
        }

        fn get(&self, coords: [i32; 1]) -> Option<&T> {
            self.get(coords[0])
        }

        fn range_query(&self, min: [i32; 1], max: [i32; 1]) -> Vec<&T> {
            (min[0]..=max[0]).filter_map(|i| self.get(i)).collect()
        }
    }

    impl<T> Benchable<2, T> for OnceBiVec<OnceBiVec<T>> {
        fn name() -> &'static str {
            "oncebivec"
        }

        fn new(min: [i32; 2]) -> Self {
            let layer0 = OnceBiVec::new(min[0]);
            // Initialize with empty middle and inner vectors
            for i in min[0]..min[0] + ONCEBIVEC_INIT_DEPTH {
                let layer1 = OnceBiVec::new(min[1]);
                layer0.push_checked(layer1, i);
            }
            layer0
        }

        fn push_checked(&self, coords: [i32; 2], value: T) {
            // Get or create inner vector
            if let Some(layer0) = self.get(coords[0]) {
                layer0.push_checked(value, coords[1]);
            } else {
                let layer1 = OnceBiVec::new(coords[1]);
                layer1.push_checked(value, coords[1]);
                self.push_checked(layer1, coords[0]);
            }
        }

        fn get(&self, coords: [i32; 2]) -> Option<&T> {
            if coords[0] >= self.min_degree() && coords[0] < self.len() {
                let layer1 = self.get(coords[0])?;
                if coords[1] >= layer1.min_degree() && coords[1] < layer1.len() {
                    layer1.get(coords[1])
                } else {
                    None
                }
            } else {
                None
            }
        }

        fn range_query(&self, min: [i32; 2], max: [i32; 2]) -> Vec<&T> {
            let mut result = Vec::new();
            for i in min[0]..=max[0] {
                if let Some(inner) = self.get(i) {
                    for j in min[1]..=max[1] {
                        if let Some(value) = inner.get(j) {
                            result.push(value);
                        }
                    }
                }
            }
            result
        }
    }

    impl<T> Benchable<3, T> for OnceBiVec<OnceBiVec<OnceBiVec<T>>> {
        fn name() -> &'static str {
            "oncebivec"
        }

        fn new(min: [i32; 3]) -> Self {
            let layer0 = OnceBiVec::new(min[0]);
            // Initialize with empty middle and inner vectors
            for i in min[0]..min[0] + ONCEBIVEC_INIT_DEPTH {
                let layer1 = OnceBiVec::new(min[1]);
                for j in min[1]..=min[1] + ONCEBIVEC_INIT_DEPTH {
                    let layer2 = OnceBiVec::new(min[2]);
                    layer1.push_checked(layer2, j);
                }
                layer0.push_checked(layer1, i);
            }
            layer0
        }

        fn push_checked(&self, coords: [i32; 3], value: T) {
            // Get or create middle vector
            if let Some(layer1) = self.get(coords[0]) {
                if let Some(layer2) = layer1.get(coords[1]) {
                    layer2.push_checked(value, coords[2]);
                } else {
                    let layer2 = OnceBiVec::new(coords[2]);
                    layer2.push_checked(value, coords[2]);
                    layer1.push_checked(layer2, coords[1]);
                }
            } else {
                let layer1 = OnceBiVec::new(coords[1]);
                let layer2 = OnceBiVec::new(coords[2]);
                layer2.push_checked(value, coords[2]);
                layer1.push_checked(layer2, coords[1]);
                self.push_checked(layer1, coords[0]);
            }
        }

        fn get(&self, coords: [i32; 3]) -> Option<&T> {
            if coords[0] >= self.min_degree() && coords[0] < self.len() {
                let layer1 = self.get(coords[0])?;
                if coords[1] >= layer1.min_degree() && coords[1] < layer1.len() {
                    let layer2 = layer1.get(coords[1])?;
                    if coords[2] >= layer2.min_degree() && coords[2] < layer2.len() {
                        layer2.get(coords[2])
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }

        fn range_query(&self, min: [i32; 3], max: [i32; 3]) -> Vec<&T> {
            let mut result = Vec::new();
            for i in min[0]..=max[0] {
                if let Some(middle) = self.get(i) {
                    for j in min[1]..=max[1] {
                        if let Some(inner) = middle.get(j) {
                            for k in min[2]..=max[2] {
                                if let Some(value) = inner.get(k) {
                                    result.push(value);
                                }
                            }
                        }
                    }
                }
            }
            result
        }
    }

    impl<T> Benchable<4, T> for OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<T>>>> {
        fn name() -> &'static str {
            "oncebivec"
        }

        fn new(min: [i32; 4]) -> Self {
            let layer0 = OnceBiVec::new(min[0]);
            // Initialize with empty middle and inner vectors
            for i in min[0]..min[0] + ONCEBIVEC_INIT_DEPTH {
                let layer1 = OnceBiVec::new(min[1]);
                for j in min[1]..=min[1] + ONCEBIVEC_INIT_DEPTH {
                    let layer2 = OnceBiVec::new(min[2]);
                    for k in min[2]..=min[2] + ONCEBIVEC_INIT_DEPTH {
                        let layer3 = OnceBiVec::new(min[3]);
                        layer2.push_checked(layer3, k);
                    }
                    layer1.push_checked(layer2, j);
                }
                layer0.push_checked(layer1, i);
            }
            layer0
        }

        fn push_checked(&self, coords: [i32; 4], value: T) {
            if let Some(layer1) = self.get(coords[0]) {
                if let Some(layer2) = layer1.get(coords[1]) {
                    if let Some(layer3) = layer2.get(coords[2]) {
                        layer3.push_checked(value, coords[3]);
                    } else {
                        let layer3 = OnceBiVec::new(coords[3]);
                        layer3.push_checked(value, coords[3]);
                        layer2.push_checked(layer3, coords[2]);
                    }
                } else {
                    let layer3 = OnceBiVec::new(coords[3]);
                    layer3.push_checked(value, coords[3]);
                    let layer2 = OnceBiVec::new(coords[2]);
                    layer2.push_checked(layer3, coords[2]);
                    layer1.push_checked(layer2, coords[1]);
                }
            } else {
                let layer3 = OnceBiVec::new(coords[3]);
                layer3.push_checked(value, coords[3]);
                let layer2 = OnceBiVec::new(coords[2]);
                layer2.push_checked(layer3, coords[3]);
                let layer1 = OnceBiVec::new(coords[1]);
                layer1.push_checked(layer2, coords[1]);
                self.push_checked(layer1, coords[0]);
            }
        }

        fn get(&self, coords: [i32; 4]) -> Option<&T> {
            if coords[0] >= self.min_degree() && coords[0] < self.len() {
                let layer1 = self.get(coords[0])?;
                if coords[1] >= layer1.min_degree() && coords[1] < layer1.len() {
                    let layer2 = layer1.get(coords[1])?;
                    if coords[2] >= layer2.min_degree() && coords[2] < layer2.len() {
                        let layer3 = layer2.get(coords[2])?;
                        if coords[3] >= layer3.min_degree() && coords[3] < layer3.len() {
                            layer3.get(coords[3])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }

        fn range_query(&self, min: [i32; 4], max: [i32; 4]) -> Vec<&T> {
            let mut result = Vec::new();
            for i in min[0]..=max[0] {
                if let Some(layer1) = self.get(i) {
                    for j in min[1]..=max[1] {
                        if let Some(layer2) = layer1.get(j) {
                            for k in min[2]..=max[2] {
                                if let Some(layer3) = layer2.get(k) {
                                    for l in min[3]..=max[3] {
                                        if let Some(value) = layer3.get(l) {
                                            result.push(value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            result
        }
    }

    impl<T> Benchable<5, T> for OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<T>>>>> {
        fn name() -> &'static str {
            "oncebivec"
        }

        fn new(min: [i32; 5]) -> Self {
            let layer0 = OnceBiVec::new(min[0]);
            // Initialize with empty middle and inner vectors
            for i in min[0]..min[0] + ONCEBIVEC_INIT_DEPTH {
                let layer1 = OnceBiVec::new(min[1]);
                for j in min[1]..=min[1] + ONCEBIVEC_INIT_DEPTH {
                    let layer2 = OnceBiVec::new(min[2]);
                    for k in min[2]..=min[2] + ONCEBIVEC_INIT_DEPTH {
                        let layer3 = OnceBiVec::new(min[3]);
                        for l in min[2]..=min[2] + ONCEBIVEC_INIT_DEPTH {
                            let layer4 = OnceBiVec::new(min[4]);
                            layer3.push_checked(layer4, l);
                        }
                        layer2.push_checked(layer3, k);
                    }
                    layer1.push_checked(layer2, j);
                }
                layer0.push_checked(layer1, i);
            }
            layer0
        }

        fn push_checked(&self, coords: [i32; 5], value: T) {
            if let Some(layer1) = self.get(coords[0]) {
                if let Some(layer2) = layer1.get(coords[1]) {
                    if let Some(layer3) = layer2.get(coords[2]) {
                        if let Some(layer4) = layer3.get(coords[3]) {
                            layer4.push_checked(value, coords[4]);
                        } else {
                            let layer4 = OnceBiVec::new(coords[4]);
                            layer4.push_checked(value, coords[4]);
                            layer3.push_checked(layer4, coords[3]);
                        }
                    } else {
                        let layer4 = OnceBiVec::new(coords[4]);
                        layer4.push_checked(value, coords[4]);
                        let layer3 = OnceBiVec::new(coords[3]);
                        layer3.push_checked(layer4, coords[3]);
                        layer2.push_checked(layer3, coords[2]);
                    }
                } else {
                    let layer4 = OnceBiVec::new(coords[4]);
                    layer4.push_checked(value, coords[4]);
                    let layer3 = OnceBiVec::new(coords[3]);
                    layer3.push_checked(layer4, coords[3]);
                    let layer2 = OnceBiVec::new(coords[2]);
                    layer2.push_checked(layer3, coords[2]);
                    layer1.push_checked(layer2, coords[1]);
                }
            } else {
                let layer4 = OnceBiVec::new(coords[4]);
                layer4.push_checked(value, coords[4]);
                let layer3 = OnceBiVec::new(coords[3]);
                layer3.push_checked(layer4, coords[3]);
                let layer2 = OnceBiVec::new(coords[2]);
                layer2.push_checked(layer3, coords[3]);
                let layer1 = OnceBiVec::new(coords[1]);
                layer1.push_checked(layer2, coords[1]);
                self.push_checked(layer1, coords[0]);
            }
        }

        fn get(&self, coords: [i32; 5]) -> Option<&T> {
            if coords[0] >= self.min_degree() && coords[0] < self.len() {
                let layer1 = self.get(coords[0])?;
                if coords[1] >= layer1.min_degree() && coords[1] < layer1.len() {
                    let layer2 = layer1.get(coords[1])?;
                    if coords[2] >= layer2.min_degree() && coords[2] < layer2.len() {
                        let layer3 = layer2.get(coords[2])?;
                        if coords[3] >= layer3.min_degree() && coords[3] < layer3.len() {
                            let layer4 = layer3.get(coords[3])?;
                            if coords[4] >= layer4.min_degree() && coords[4] < layer4.len() {
                                layer4.get(coords[4])
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }

        fn range_query(&self, min: [i32; 5], max: [i32; 5]) -> Vec<&T> {
            let mut result = Vec::new();
            for i in min[0]..=max[0] {
                if let Some(layer1) = self.get(i) {
                    for j in min[1]..=max[1] {
                        if let Some(layer2) = layer1.get(j) {
                            for k in min[2]..=max[2] {
                                if let Some(layer3) = layer2.get(k) {
                                    for l in min[3]..=max[3] {
                                        if let Some(layer4) = layer3.get(l) {
                                            for m in min[4]..=max[4] {
                                                if let Some(value) = layer4.get(m) {
                                                    result.push(value);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            result
        }
    }
}

fn get_n_coords<const K: usize>(n: usize, min: [i32; K]) -> Vec<[i32; K]> {
    (0..)
        .flat_map(|i| once::get_nth_diagonal::<K>(i))
        .map(|mut v| {
            for (xi, mi) in v.iter_mut().zip(min.iter()) {
                *xi += mi;
            }
            v
        })
        .take(n)
        .collect()
}

const NUM_ELEMENTS: usize = 1 << 12;

// Benchmark insertion for different dimensions
fn bench_insert_k<const K: usize, T, B: Benchable<K, T>>(
    c: &mut BenchmarkGroup<'_, WallTime>,
    min: [i32; K],
    make_value: impl Fn(usize) -> T,
) {
    let coords: Vec<[i32; K]> = get_n_coords(NUM_ELEMENTS, min);

    c.bench_function(
        &format!("{}_insert_k{}_{}", B::name(), K, std::any::type_name::<T>()),
        |b| {
            b.iter_batched(
                || B::new(min),
                |vec| {
                    for (i, coord) in coords.iter().enumerate() {
                        vec.push_checked(*coord, make_value(i));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        },
    );
}

// Benchmark lookups for different dimensions
fn bench_lookup_k<const K: usize, T, B: Benchable<K, T>>(
    c: &mut BenchmarkGroup<'_, WallTime>,
    min: [i32; K],
    make_value: &dyn Fn(usize) -> T,
) {
    let vec = B::new(min);
    let mut coords = get_n_coords(NUM_ELEMENTS, min);

    // Insert data
    for (i, coord) in coords.iter().enumerate() {
        vec.push_checked(*coord, make_value(i));
    }

    coords.shuffle(&mut rng());

    c.bench_function(
        &format!("{}_lookup_k{}_{}", B::name(), K, std::any::type_name::<T>()),
        |b| {
            b.iter(|| {
                for coord in coords.iter() {
                    black_box(vec.get(*coord));
                }
            })
        },
    );
}

fn run_insert_benchmark<
    const K: usize,
    T,
    B1: Benchable<K, T>,
    B2: Benchable<K, T>,
    B3: Benchable<K, T>,
>(
    c: &mut Criterion,
    min: [i32; K],
    make_value: &dyn Fn(usize) -> T,
) {
    let mut g = c.benchmark_group(format!("insert_dim{K}_{}", std::any::type_name::<T>()));
    bench_insert_k::<K, _, B1>(&mut g, min, make_value);
    // bench_insert_k::<K, _, B2>(&mut g, min, make_value);
    bench_insert_k::<K, _, B3>(&mut g, min, make_value);
    g.finish();
}

fn run_insert_benchmarks<T>(c: &mut Criterion, make_value: &dyn Fn(usize) -> T) {
    // Dim 1
    let mut g = c.benchmark_group(format!("insert_dim1_{}", std::any::type_name::<T>()));
    bench_insert_k::<1, _, OnceBiVec<_>>(&mut g, [0], make_value);
    // bench_insert_k::<1, _, SparseVec<_>>(&mut g, [0], make_value);
    // bench_insert_k::<1, _, SparseKVec<_, 1>>(&mut g, [0], make_value);
    bench_insert_k::<1, _, TwoEndedGrove<_>>(&mut g, [0], make_value);
    bench_insert_k::<1, _, MultiGraded<1, _>>(&mut g, [0], make_value);
    g.finish();

    run_insert_benchmark::<2, _, OnceBiVec<OnceBiVec<_>>, SparseKVec<_, 2>, MultiGraded<2, _>>(
        c,
        [0, 0],
        make_value,
    );
    run_insert_benchmark::<
        3,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<_>>>,
        SparseKVec<_, 3>,
        MultiGraded<3, _>,
    >(c, [0, 0, 0], make_value);
    run_insert_benchmark::<
        4,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>,
        SparseKVec<_, 4>,
        MultiGraded<4, _>,
    >(c, [0, 0, 0, 0], make_value);
    run_insert_benchmark::<
        5,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>>,
        SparseKVec<_, 5>,
        MultiGraded<5, _>,
    >(c, [0, 0, 0, 0, 0], make_value);

    let mut g = c.benchmark_group(format!("insert_dim6_{}", std::any::type_name::<T>()));
    // bench_insert_k::<6, _, SparseKVec<_, 6>>(&mut g, [0, 0, 0, 0, 0, 0], make_value);
    bench_insert_k::<6, _, MultiGraded<6, _>>(&mut g, [0, 0, 0, 0, 0, 0], make_value);
    g.finish();
}

fn run_lookup_benchmark<
    const K: usize,
    T,
    B1: Benchable<K, T>,
    B2: Benchable<K, T>,
    B3: Benchable<K, T>,
>(
    c: &mut Criterion,
    min: [i32; K],
    make_value: &dyn Fn(usize) -> T,
) {
    let mut g = c.benchmark_group(format!("lookup_dim{K}_{}", std::any::type_name::<T>()));
    bench_lookup_k::<K, _, B1>(&mut g, min, make_value);
    // bench_lookup_k::<K, _, B2>(&mut g, min, make_value);
    bench_lookup_k::<K, _, B3>(&mut g, min, make_value);
    g.finish();
}

fn run_lookup_benchmarks<T>(c: &mut Criterion, make_value: &dyn Fn(usize) -> T) {
    // Dim 1
    let mut g = c.benchmark_group(format!("lookup_dim1_{}", std::any::type_name::<T>()));
    bench_lookup_k::<1, _, OnceBiVec<_>>(&mut g, [0], make_value);
    // bench_lookup_k::<1, _, SparseVec<_>>(&mut g, [0], make_value);
    // bench_lookup_k::<1, _, SparseKVec<_, 1>>(&mut g, [0], make_value);
    bench_lookup_k::<1, _, TwoEndedGrove<_>>(&mut g, [0], make_value);
    bench_lookup_k::<1, _, MultiGraded<1, _>>(&mut g, [0], make_value);
    g.finish();

    run_lookup_benchmark::<2, _, OnceBiVec<OnceBiVec<_>>, SparseKVec<_, 2>, MultiGraded<2, _>>(
        c,
        [0, 0],
        make_value,
    );
    run_lookup_benchmark::<
        3,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<_>>>,
        SparseKVec<_, 3>,
        MultiGraded<3, _>,
    >(c, [0, 0, 0], make_value);
    run_lookup_benchmark::<
        4,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>,
        SparseKVec<_, 4>,
        MultiGraded<4, _>,
    >(c, [0, 0, 0, 0], make_value);
    run_lookup_benchmark::<
        5,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>>,
        SparseKVec<_, 5>,
        MultiGraded<5, _>,
    >(c, [0, 0, 0, 0, 0], make_value);

    let mut g = c.benchmark_group(format!("lookup_dim6_{}", std::any::type_name::<T>()));
    // bench_lookup_k::<6, _, SparseKVec<_, 6>>(&mut g, [0, 0, 0, 0, 0, 0], make_value);
    bench_lookup_k::<6, _, MultiGraded<6, _>>(&mut g, [0, 0, 0, 0, 0, 0], make_value);
    g.finish();
}

// Helper functions to generate different types of ranges
fn hypercube_range<const K: usize>(min: [i32; K], size: i32) -> ([i32; K], [i32; K]) {
    let mut max = min;
    for i in 0..K {
        max[i] += size;
    }
    (min, max)
}

fn hyperplane_range<const K: usize>(
    min: [i32; K],
    fixed_dim: usize,
    fixed_val: i32,
    size: i32,
) -> ([i32; K], [i32; K]) {
    let mut range_min = min;
    let mut range_max = min;
    range_min[fixed_dim] = fixed_val;
    range_max[fixed_dim] = fixed_val;
    for i in 0..K {
        if i != fixed_dim {
            range_max[i] += size;
        }
    }
    (range_min, range_max)
}

fn codim2_range<const K: usize>(
    min: [i32; K],
    fixed_dim1: usize,
    fixed_val1: i32,
    fixed_dim2: usize,
    fixed_val2: i32,
    size: i32,
) -> ([i32; K], [i32; K]) {
    let mut range_min = min;
    let mut range_max = min;
    range_min[fixed_dim1] = fixed_val1;
    range_max[fixed_dim1] = fixed_val1;
    range_min[fixed_dim2] = fixed_val2;
    range_max[fixed_dim2] = fixed_val2;
    for i in 0..K {
        if i != fixed_dim1 && i != fixed_dim2 {
            range_max[i] += size;
        }
    }
    (range_min, range_max)
}

// Benchmark range queries for different dimensions and shapes
fn bench_range_k<const K: usize, T, B: Benchable<K, T>>(
    c: &mut BenchmarkGroup<'_, WallTime>,
    min: [i32; K],
    make_value: &dyn Fn(usize) -> T,
) {
    let vec = B::new(min);
    let coords = get_n_coords(NUM_ELEMENTS, min);

    // Insert data
    for (i, coord) in coords.iter().enumerate() {
        vec.push_checked(*coord, make_value(i));
    }

    // Test hypercubes of different sizes
    for size in [2, 5, 10] {
        let (range_min, range_max) = hypercube_range(min, size);
        c.bench_function(&format!("{}_range_k{}_cube_{}", B::name(), K, size), |b| {
            b.iter(|| {
                black_box(vec.range_query(range_min, range_max));
            })
        });
    }

    // Test hyperplanes in different dimensions
    if K > 1 {
        for fixed_dim in 0..K {
            let fixed_val = min[fixed_dim] + 5;
            let (range_min, range_max) = hyperplane_range(min, fixed_dim, fixed_val, 10);
            c.bench_function(
                &format!("{}_range_k{}_plane_dim{}", B::name(), K, fixed_dim),
                |b| {
                    b.iter(|| {
                        black_box(vec.range_query(range_min, range_max));
                    })
                },
            );
        }
    }

    // Test codimension 2 subspaces
    if K > 2 {
        for fixed_dim1 in 0..K {
            for fixed_dim2 in (fixed_dim1 + 1)..K {
                let fixed_val1 = min[fixed_dim1] + 5;
                let fixed_val2 = min[fixed_dim2] + 5;
                let (range_min, range_max) =
                    codim2_range(min, fixed_dim1, fixed_val1, fixed_dim2, fixed_val2, 10);
                c.bench_function(
                    &format!(
                        "{}_range_k{}_codim2_dim{}_{}",
                        B::name(),
                        K,
                        fixed_dim1,
                        fixed_dim2
                    ),
                    |b| {
                        b.iter(|| {
                            black_box(vec.range_query(range_min, range_max));
                        })
                    },
                );
            }
        }
    }
}

fn run_range_benchmark<const K: usize, T, B1: Benchable<K, T>, B2: Benchable<K, T>>(
    c: &mut Criterion,
    min: [i32; K],
    make_value: &dyn Fn(usize) -> T,
) {
    let mut g = c.benchmark_group(format!("range_dim{K}"));
    bench_range_k::<K, _, B1>(&mut g, min, make_value);
    bench_range_k::<K, _, B2>(&mut g, min, make_value);
    g.finish();
}

fn run_range_benchmarks<T>(c: &mut Criterion, make_value: &dyn Fn(usize) -> i32) {
    // Dim 1
    let mut g = c.benchmark_group("range_dim1");
    bench_range_k::<1, _, OnceBiVec<_>>(&mut g, [0], make_value);
    bench_range_k::<1, _, SparseVec<_>>(&mut g, [0], make_value);
    bench_range_k::<1, _, SparseKVec<_, 1>>(&mut g, [0], make_value);
    g.finish();

    run_range_benchmark::<2, _, OnceBiVec<OnceBiVec<_>>, SparseKVec<_, 2>>(c, [0, 0], make_value);
    run_range_benchmark::<3, _, OnceBiVec<OnceBiVec<OnceBiVec<_>>>, SparseKVec<_, 3>>(
        c,
        [0, 0, 0],
        make_value,
    );
    run_range_benchmark::<4, _, OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>, SparseKVec<_, 4>>(
        c,
        [0, 0, 0, 0],
        make_value,
    );
    run_range_benchmark::<
        5,
        _,
        OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<OnceBiVec<_>>>>>,
        SparseKVec<_, 5>,
    >(c, [0, 0, 0, 0, 0], make_value);

    let mut g = c.benchmark_group("range_dim6");
    bench_range_k::<6, _, SparseKVec<_, 6>>(&mut g, [0, 0, 0, 0, 0, 0], make_value);
    g.finish();
}

fn run_benchmarks(c: &mut Criterion) {
    run_insert_benchmarks(c, &|i| i as i32);
    run_insert_benchmarks(c, &|i| [i; 1000]);
    run_lookup_benchmarks(c, &|i| i as i32);
    run_lookup_benchmarks(c, &|i| [i; 1000]);
    // run_range_benchmarks(c);
}

use pprof::criterion::{Output, PProfProfiler};

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(30)).with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = run_benchmarks
}
criterion_main!(benches);
