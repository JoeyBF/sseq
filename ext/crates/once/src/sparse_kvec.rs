use std::sync::atomic::{AtomicI32, AtomicPtr, Ordering};

use dashmap::DashMap;
use fxhash::FxBuildHasher;
use nohash_hasher::BuildNoHashHasher;
use rand::{rng, seq::SliceRandom};

/// Helper function to recursively generate the tuples
fn generate_tuples<const K: usize>(
    tuple: &mut Vec<i32>,
    index: usize,
    sum: usize,
    result: &mut Vec<[i32; K]>,
) {
    if index == K - 1 {
        // The last element gets whatever is left to reach the sum
        tuple[index] = sum as i32;
        result.push(tuple.clone().try_into().unwrap()); // Convert to [i32; K]
        return;
    }

    for i in 0..=sum {
        tuple[index] = i as i32;
        generate_tuples::<K>(tuple, index + 1, sum - i, result);
    }
}

/// Generate all tuples of length K where the sum of coordinates equals n
/// ```
/// let result = once::get_nth_diagonal::<3>(4);
///
/// assert_eq!(result.len(), 15);
/// assert!(result.contains(&[0, 0, 4]));
/// assert!(result.contains(&[0, 1, 3]));
/// assert!(result.contains(&[0, 2, 2]));
/// assert!(result.contains(&[0, 3, 1]));
/// assert!(result.contains(&[0, 4, 0]));
/// assert!(result.contains(&[1, 0, 3]));
/// assert!(result.contains(&[1, 1, 2]));
/// assert!(result.contains(&[1, 2, 1]));
/// assert!(result.contains(&[1, 3, 0]));
/// assert!(result.contains(&[2, 0, 2]));
/// assert!(result.contains(&[2, 1, 1]));
/// assert!(result.contains(&[2, 2, 0]));
/// assert!(result.contains(&[3, 0, 1]));
/// assert!(result.contains(&[3, 1, 0]));
/// assert!(result.contains(&[4, 0, 0]));
/// ```
pub fn get_nth_diagonal<const K: usize>(n: usize) -> Vec<[i32; K]> {
    let mut result = Vec::new();
    let mut tuple = vec![0; K];

    // Generate all tuples where the sum of coordinates equals n
    generate_tuples::<K>(&mut tuple, 0, n, &mut result);

    // Shuffle the result to randomize the order
    let mut rng = rng();
    result.shuffle(&mut rng);

    result
}

/// A node in the coordinate list for range queries
#[derive(Debug)]
struct CoordList<const K: usize> {
    // Coordinates of an entry
    coords: [i32; K],
    // Next entry in the list
    next: AtomicPtr<CoordList<K>>,
}

/// A concurrent, lock-free sparse vector indexed by k-tuples of i32s.
/// Supports O(1) lookups and efficient range queries.
pub struct SparseKVec<T, const K: usize> {
    // Main storage for values
    data: DashMap<[i32; K], T, FxBuildHasher>,
    // Range indices for efficient range queries
    range_indices: [RangeIndex<K>; K],
}

/// Tracks entries along a single dimension for range queries
struct RangeIndex<const K: usize> {
    // Maps from coordinate value to list of entries with that value in this dimension
    point_map: DashMap<i32, AtomicPtr<CoordList<K>>, BuildNoHashHasher<i32>>,
    // Tracks min/max values seen in this dimension
    min_value: AtomicI32,
    max_value: AtomicI32,
}

impl<const K: usize, T> SparseKVec<T, K> {
    /// Creates a new empty SparseKVec
    pub fn new() -> Self {
        let range_indices = std::array::from_fn(|_| RangeIndex {
            point_map: DashMap::with_hasher(BuildNoHashHasher::default()),
            min_value: AtomicI32::new(i32::MAX),
            max_value: AtomicI32::new(i32::MIN),
        });

        Self {
            data: DashMap::with_hasher(FxBuildHasher::default()),
            range_indices,
        }
    }

    /// Returns the number of elements in the vector
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector contains no elements
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Gets a reference to the value at the given coordinates, if it exists
    pub fn get<'a>(&'a self, coords: [i32; K]) -> Option<&'a T> {
        self.data.get(&coords).map(|r| {
            // SAFETY: We do not allow deleting elements, so the reference is always valid. At
            // worst, the value may be modified through inner mutability, but this doesn't
            // invalidate the reference. dashmap::ReadOnlyView uses a similar technique.
            unsafe { std::mem::transmute::<&'_ T, &'a T>(&*r) }
        })
    }

    /// Inserts a value at the given coordinates
    /// Returns true if the value was inserted, false if the coordinates were already occupied
    pub fn insert(&self, coords: [i32; K], value: T) -> bool {
        // Try to insert into main storage
        // Use entry API to ensure atomicity
        if let dashmap::mapref::entry::Entry::Vacant(entry) = self.data.entry(coords) {
            entry.insert(value);
        } else {
            return false;
        }

        // Update range indices
        for dim in 0..K {
            let coord_value = coords[dim];
            let range_index = &self.range_indices[dim];

            // Update min/max values
            loop {
                let current_min = range_index.min_value.load(Ordering::Acquire);
                if coord_value >= current_min {
                    break;
                }
                if range_index
                    .min_value
                    .compare_exchange(
                        current_min,
                        coord_value,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    break;
                }
            }

            loop {
                let current_max = range_index.max_value.load(Ordering::Acquire);
                if coord_value <= current_max {
                    break;
                }
                if range_index
                    .max_value
                    .compare_exchange(
                        current_max,
                        coord_value,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    break;
                }
            }

            // Get or create the list for this coordinate value
            let entry = range_index
                .point_map
                .entry(coord_value)
                .or_insert_with(|| AtomicPtr::new(std::ptr::null_mut()));

            // Insert at head of list using compare_and_swap
            let mut current_head = entry.load(Ordering::Acquire);
            loop {
                // Create new coordinate list node for this attempt
                let new_node = Box::new(CoordList {
                    coords,
                    next: AtomicPtr::new(current_head),
                });

                match entry.compare_exchange(
                    current_head,
                    Box::into_raw(new_node),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(new_head) => current_head = new_head,
                }
            }
        }

        true
    }

    /// Returns an iterator over all values whose coordinates fall within the given range (inclusive)
    pub fn range<'a>(
        &'a self,
        min: [i32; K],
        max: [i32; K],
    ) -> impl Iterator<Item = ([i32; K], &'a T)> {
        // Find dimension with smallest range
        let best_dim = (0..K)
            .min_by_key(|&dim| {
                let dim_min = std::cmp::max(
                    min[dim],
                    self.range_indices[dim].min_value.load(Ordering::Acquire),
                );
                let dim_max = std::cmp::min(
                    max[dim],
                    self.range_indices[dim].max_value.load(Ordering::Acquire),
                );
                dim_max - dim_min + 1
            })
            .unwrap_or(0);

        let range_index = &self.range_indices[best_dim];
        let dim_min = std::cmp::max(min[best_dim], range_index.min_value.load(Ordering::Acquire));
        let dim_max = std::cmp::min(max[best_dim], range_index.max_value.load(Ordering::Acquire));

        // Collect candidate coordinates by scanning the chosen dimension
        let mut candidates = Vec::new();
        for value in dim_min..=dim_max {
            if let Some(list_head) = range_index.point_map.get(&value) {
                let mut current = list_head.load(Ordering::Acquire);
                while !current.is_null() {
                    let node = unsafe { &*current };

                    // Check if coordinates are within range
                    if (0..K)
                        .all(|dim| min[dim] <= node.coords[dim] && node.coords[dim] <= max[dim])
                    {
                        candidates.push(node.coords);
                    }

                    current = node.next.load(Ordering::Acquire);
                }
            }
        }

        // Return iterator over values
        candidates
            .into_iter()
            .filter_map(move |coords| self.get(coords).map(|r| (coords, r)))
    }
}

impl<const K: usize, T> Default for SparseKVec<T, K> {
    fn default() -> Self {
        Self::new()
    }
}

// Specialized implementation for K=1
pub struct SparseVec<T> {
    data: DashMap<i32, T, BuildNoHashHasher<i32>>,
    min_value: AtomicI32,
    max_value: AtomicI32,
}

impl<T> SparseVec<T> {
    pub fn new() -> Self {
        Self {
            data: DashMap::with_hasher(BuildNoHashHasher::default()),
            min_value: AtomicI32::new(i32::MAX),
            max_value: AtomicI32::new(i32::MIN),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get<'a>(&'a self, index: i32) -> Option<&'a T> {
        self.data.get(&index).map(|r| {
            // SAFETY: We do not allow deleting elements, so the reference is always valid. At
            // worst, the value may be modified through inner mutability, but this doesn't
            // invalidate the reference. dashmap::ReadOnlyView uses a similar technique.
            unsafe { std::mem::transmute::<&'_ T, &'a T>(&*r) }
        })
    }

    pub fn insert(&self, index: i32, value: T) -> bool {
        // Use entry API to ensure atomicity
        if let dashmap::mapref::entry::Entry::Vacant(entry) = self.data.entry(index) {
            entry.insert(value);
        } else {
            return false;
        }

        // Update min value
        loop {
            let current_min = self.min_value.load(Ordering::Acquire);
            if index >= current_min {
                break;
            }
            if self
                .min_value
                .compare_exchange(current_min, index, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }

        // Update max value
        loop {
            let current_max = self.max_value.load(Ordering::Acquire);
            if index <= current_max {
                break;
            }
            if self
                .max_value
                .compare_exchange(current_max, index, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }

        true
    }

    pub fn range(&self, min: i32, max: i32) -> impl Iterator<Item = (i32, &T)> {
        let min = std::cmp::max(min, self.min_value.load(Ordering::Acquire));
        let max = std::cmp::min(max, self.max_value.load(Ordering::Acquire));
        (min..=max).filter_map(move |i| self.get(i).map(|r| (i, r)))
    }
}

impl<T> Default for SparseVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

// #[cfg(not(miri))]
// #[cfg(test)]
// mod tests {
//     use proptest::prelude::*;

//     use super::*;

//     // Strategy for generating valid k-dimensional coordinates
//     fn coord_strategy<const K: usize>() -> impl Strategy<Value = [i32; K]> {
//         prop::array::uniform(-10i32..10i32)
//     }

//     // Strategy for generating a sequence of unique coordinates
//     fn unique_coords_strategy<const K: usize>(
//         count: usize,
//     ) -> impl Strategy<Value = Vec<([i32; K], i32)>> {
//         prop::collection::hash_set(coord_strategy::<K>(), count..=count).prop_map(|coords| {
//             coords
//                 .into_iter()
//                 .enumerate()
//                 .map(|(i, c)| (c, i as i32))
//                 .collect()
//         })
//     }

//     // Generic property tests
//     mod generic_props {
//         use super::*;

//         pub fn test_insert_get<const K: usize>() {
//             proptest!(|(coords in unique_coords_strategy::<K>(5))| {
//                 let vec = SparseKVec::<i32, K>::new();

//                 // Test insert and get
//                 for (coord, val) in &coords {
//                     prop_assert!(vec.insert(*coord, *val));
//                     prop_assert_eq!(*vec.get(*coord).unwrap(), *val);
//                 }

//                 // Test duplicate inserts
//                 for (coord, val) in &coords {
//                     prop_assert!(!vec.insert(*coord, *val + 1));
//                     prop_assert_eq!(*vec.get(*coord).unwrap(), *val);
//                 }

//                 // Test non-existent coordinates
//                 let mut missing_coord = [0i32; K];
//                 for i in 0..K {
//                     missing_coord[i] = coords.iter().map(|(c, _)| c[i]).max().unwrap_or(0) + 1;
//                 }
//                 prop_assert!(vec.get(missing_coord).is_none());
//             });
//         }

//         pub fn test_range_queries<const K: usize>() {
//             proptest!(|(
//                 coords in unique_coords_strategy::<K>(10),
//                 offset in -5i32..5i32,
//                 range_size in 1usize..5usize,
//             )| {
//                 let vec = SparseKVec::<i32, K>::new();

//                 // Insert values
//                 for (coord, val) in &coords {
//                     vec.insert(*coord, *val);
//                 }

//                 // Test empty range
//                 let empty_min = [i32::MAX / 2; K];
//                 let empty_max = [i32::MIN / 2; K];
//                 prop_assert_eq!(vec.range(empty_min, empty_max).count(), 0);

//                 // Test point range (min == max)
//                 for (coord, val) in &coords {
//                     let range_result: Vec<_> = vec.range(*coord, *coord).map(|r| *r.1).collect();
//                     prop_assert_eq!(range_result.len(), 1);
//                     prop_assert_eq!(range_result[0], *val);
//                 }

//                 // Test sliding window range
//                 if !coords.is_empty() {
//                     for dim in 0..K {
//                         let mut min = [0i32; K];
//                         let mut max = [0i32; K];
//                         for i in 0..K {
//                             min[i] = coords.iter().map(|(c, _)| c[i]).min().unwrap() + offset;
//                             max[i] = min[i] + if i == dim { range_size as i32 } else { 0 };
//                         }

//                         let range_result: Vec<_> = vec.range(min, max).map(|r| *r.1).collect();
//                         let expected: Vec<_> = coords.iter()
//                             .filter(|(c, _)| (0..K).all(|i| c[i] >= min[i] && c[i] <= max[i]))
//                             .map(|(_, v)| *v)
//                             .collect();

//                         prop_assert_eq!(range_result.len(), expected.len());
//                         for x in range_result {
//                             prop_assert!(expected.contains(&x));
//                         }
//                     }
//                 }
//             });
//         }

//         pub fn test_concurrent_operations<const K: usize>() {
//             proptest!(|(coords in unique_coords_strategy::<K>(20))| {
//                 use std::{sync::Arc, thread};

//                 let vec = Arc::new(SparseKVec::<i32, K>::new());
//                 let mut handles = Vec::new();

//                 // Split coordinates into chunks for concurrent insertion
//                 let chunk_size = coords.len() / 4;
//                 let chunks: Vec<_> = coords.chunks(chunk_size).collect();

//                 // Spawn threads for concurrent insertion
//                 for chunk in chunks.iter() {
//                     let vec = Arc::clone(&vec);
//                     let chunk = chunk.to_vec();
//                     handles.push(thread::spawn(move || {
//                         for (coord, val) in chunk {
//                             vec.insert(coord, val);
//                         }
//                     }));
//                 }

//                 // Wait for insertions to complete
//                 for handle in handles {
//                     handle.join().unwrap();
//                 }

//                 // Verify all values were inserted
//                 for (coord, val) in &coords {
//                     prop_assert_eq!(*vec.get(*coord).unwrap(), *val);
//                 }

//                 // Test concurrent range queries
//                 let mut handles = Vec::new();
//                 let min = [0i32; K];
//                 let max = [10i32; K];

//                 for _ in 0..4 {
//                     let vec = Arc::clone(&vec);
//                     handles.push(thread::spawn(move || {
//                         vec.range(min, max).count()
//                     }));
//                 }

//                 // All threads should see the same count
//                 let counts: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
//                 for i in 1..counts.len() {
//                     prop_assert_eq!(counts[0], counts[i]);
//                 }
//             });
//         }
//     }

//     // Macro to instantiate property tests for different dimensions
//     macro_rules! instantiate_dimension_tests {
//         ($($k:literal),*) => {
//             $(
//                 paste::paste! {
//                     #[test]
//                     fn [<test_insert_get_dim_ $k>]() {
//                         generic_props::test_insert_get::<$k>();
//                     }

//                     #[test]
//                     fn [<test_range_queries_dim_ $k>]() {
//                         generic_props::test_range_queries::<$k>();
//                     }

//                     #[test]
//                     fn [<test_concurrent_ops_dim_ $k>]() {
//                         generic_props::test_concurrent_operations::<$k>();
//                     }
//                 }
//             )*
//         };
//     }

//     instantiate_dimension_tests!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

//     proptest! {
//         #[test]
//         fn prop_sparse_vec_insert_get(
//             coords in prop::collection::hash_set(-100i32..100i32, 1..100)
//         ) {
//             let vec = SparseVec::<i32>::new();

//             // Insert all values
//             for &x in &coords {
//                 prop_assert!(vec.insert(x, x));
//                 prop_assert_eq!(*vec.get(x).unwrap(), x);
//             }

//             // Verify all values
//             for &x in &coords {
//                 prop_assert_eq!(*vec.get(x).unwrap(), x);
//             }

//             // Verify non-existent values return None
//             for x in -100..100 {
//                 if !coords.contains(&x) {
//                     prop_assert!(vec.get(x).is_none());
//                 }
//             }
//         }

//         #[test]
//         fn prop_sparse_vec_range_query(
//             coords in prop::collection::hash_set(-100i32..100i32, 1..100)
//         ) {
//             let vec = SparseVec::<i32>::new();

//             // Insert all values
//             for &x in &coords {
//                 vec.insert(x, x);
//             }

//             // Test various ranges
//             for min in -100..100 {
//                 for max in min..100 {
//                     let range_result: Vec<_> = vec.range(min, max).map(|r| *r.1).collect();
//                     let expected: Vec<_> = coords.iter()
//                         .filter(|&&x| x >= min && x <= max)
//                         .copied()
//                         .collect();
//                     prop_assert_eq!(range_result.len(), expected.len());
//                     for x in range_result {
//                         prop_assert!(expected.contains(&x));
//                     }
//                 }
//             }
//         }
//     }

//     #[test]
//     fn test_sparse_vec_basic() {
//         let vec = SparseVec::<i32>::new();

//         // Test insert and get
//         assert!(vec.insert(5, 10));
//         assert!(vec.insert(-3, 20));
//         assert!(!vec.insert(5, 30)); // Should fail, already exists

//         assert_eq!(*vec.get(5).unwrap(), 10);
//         assert_eq!(*vec.get(-3).unwrap(), 20);
//         assert!(vec.get(0).is_none());

//         // Test range query
//         let range: Vec<i32> = vec.range(-3, 5).map(|r| *r.1).collect();
//         assert_eq!(range, vec![20, 10]);
//     }

//     #[test]
//     fn test_sparse_kvec_basic() {
//         let vec = SparseKVec::<i32, 2>::new();

//         // Test insert and get
//         assert!(vec.insert([1, 2], 10));
//         assert!(vec.insert([0, -1], 20));
//         assert!(!vec.insert([1, 2], 30)); // Should fail, already exists

//         assert_eq!(*vec.get([1, 2]).unwrap(), 10);
//         assert_eq!(*vec.get([0, -1]).unwrap(), 20);
//         assert!(vec.get([0, 0]).is_none());

//         // Test range query
//         let range: Vec<i32> = vec.range([0, -1], [1, 2]).map(|r| *r.1).collect();
//         assert_eq!(range.len(), 2);
//         assert!(range.contains(&10));
//         assert!(range.contains(&20));
//     }

//     #[test]
//     fn test_concurrent_insert() {
//         use std::{sync::Arc, thread};

//         let vec = Arc::new(SparseVec::<i32>::new());
//         let mut handles = Vec::new();

//         // Spawn 10 threads that each insert 100 values
//         for t in 0..10 {
//             let vec = Arc::clone(&vec);
//             handles.push(thread::spawn(move || {
//                 for i in 0..100 {
//                     let value = t * 100 + i;
//                     vec.insert(value, value);
//                 }
//             }));
//         }

//         // Wait for all threads to complete
//         for handle in handles {
//             handle.join().unwrap();
//         }

//         // Verify all values were inserted
//         assert_eq!(vec.len(), 1000);
//         for value in 0..1000 {
//             assert_eq!(*vec.get(value).unwrap(), value);
//         }
//     }

//     #[test]
//     fn test_concurrent_range_query() {
//         use std::{sync::Arc, thread};

//         let vec = Arc::new(SparseKVec::<i32, 2>::new());

//         // Insert some test data
//         for x in 0..10 {
//             for y in 0..10 {
//                 vec.insert([x, y], x * 10 + y);
//             }
//         }

//         let mut handles = Vec::new();

//         // Spawn threads that perform overlapping range queries
//         for _ in 0..10 {
//             let vec = Arc::clone(&vec);
//             handles.push(thread::spawn(move || {
//                 let range1: Vec<_> = vec.range([0, 0], [5, 5]).map(|r| *r.1).collect();
//                 let range2: Vec<_> = vec.range([3, 3], [8, 8]).map(|r| *r.1).collect();
//                 (range1, range2)
//             }));
//         }

//         // Collect and verify results
//         for handle in handles {
//             let (range1, range2) = handle.join().unwrap();
//             assert_eq!(range1.len(), 36); // 6x6 grid
//             assert_eq!(range2.len(), 36); // 6x6 grid
//         }
//     }

//     #[test]
//     fn test_concurrent_mixed_operations() {
//         use std::{sync::Arc, thread};

//         let vec = Arc::new(SparseKVec::<i32, 3>::new());
//         let mut handles = Vec::new();

//         // First, insert all values
//         for t in 0..5 {
//             for i in 0..10 {
//                 vec.insert([t, i, 0], t * 100 + i);
//             }
//         }

//         // Then perform concurrent range queries
//         for _ in 0..5 {
//             let vec = Arc::clone(&vec);
//             handles.push(thread::spawn(move || {
//                 vec.range([0, 0, 0], [4, 9, 0]).count()
//             }));
//         }

//         // All threads should see the same number of values
//         for handle in handles {
//             assert_eq!(handle.join().unwrap(), 50); // 5 threads * 10 values
//         }

//         assert_eq!(vec.len(), 50); // 5 threads * 10 values
//     }
// }
