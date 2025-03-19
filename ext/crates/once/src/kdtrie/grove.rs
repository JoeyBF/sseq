use std::num::NonZero;

use super::block::Block;
use crate::std_or_loom::sync::atomic::{AtomicUsize, Ordering};

const MAX_NUM_BLOCKS: usize = 32;

/// An insert-only sparse vector with pinned elements and geometrically growing capacity. Pun on
/// "grow vec".
pub struct Grove<T> {
    blocks: [Block<T>; MAX_NUM_BLOCKS],
    /// The maximum index that has been inserted into the `Grove`.
    ///
    /// We actually store the maximum index plus one, so that we can use zero as a sentinel value
    /// for "empty".
    max: AtomicUsize,
}

impl<T> Grove<T> {
    /// Creates a new empty `GeoVec`.
    pub fn new() -> Self {
        let blocks = std::array::from_fn(|_| Block::new());
        Self {
            blocks,
            max: AtomicUsize::new(0),
        }
    }

    /// Find the block and offset within the block for index `index`.
    fn locate(&self, index: usize) -> (usize, usize) {
        let block_num = (usize::BITS - 1 - (index + 1).leading_zeros()) as usize;
        assert!(block_num < MAX_NUM_BLOCKS);
        let block_offset = (index + 1) - (1 << block_num);
        (block_num, block_offset)
    }

    fn ensure_init(&self, block_num: usize) {
        // Safety: `Block::init` is only ever called through this method, and every block has a
        // well-defined `block_num`, and therefore a well-defined size.
        unsafe { self.blocks[block_num].init(NonZero::new(1 << block_num).unwrap()) };
    }

    /// Insert a value at the given index
    pub fn insert(&self, index: usize, value: T) {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        // Safety: We just initialized the block, and `locate` only returns valid indices
        unsafe { self.blocks[block_num].insert(block_offset, value) };
        self.max.fetch_max(index + 1, Ordering::Release);
    }

    /// Return the value at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        // Safety: We just initialized the block, and `locate` only returns valid indices
        unsafe { self.blocks[block_num].get(block_offset) }
    }

    /// Return a mutable reference to the value at the given index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        self.blocks[block_num].get_mut(block_offset)
    }

    pub fn is_set(&self, index: usize) -> bool {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        self.blocks[block_num].is_set(block_offset)
    }

    // pub fn len(&self) -> Option<usize> {
    //     self.max.load(Ordering::Acquire).checked_sub(1)
    // }
}

// We implement the derives manually to avoid the bounds on `T`

impl<T> Default for Grove<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TwoEndedGrove<T> {
    non_neg: Grove<T>,
    neg: Grove<T>,
}

impl<T> TwoEndedGrove<T> {
    pub fn new() -> Self {
        Self {
            non_neg: Grove::new(),
            neg: Grove::new(),
        }
    }

    pub fn insert(&self, idx: i32, value: T) {
        if idx >= 0 {
            self.non_neg.insert(idx as usize, value);
        } else {
            self.neg.insert((-idx) as usize, value);
        }
    }

    pub fn get(&self, idx: i32) -> Option<&T> {
        if idx >= 0 {
            self.non_neg.get(idx as usize)
        } else {
            self.neg.get((-idx) as usize)
        }
    }

    pub fn get_mut(&mut self, idx: i32) -> Option<&mut T> {
        if idx >= 0 {
            self.non_neg.get_mut(idx as usize)
        } else {
            self.neg.get_mut((-idx) as usize)
        }
    }

    pub fn min(&self) -> i32 {
        -(self.neg.max.load(Ordering::Acquire).saturating_sub(1) as i32)
    }

    pub fn max(&self) -> i32 {
        self.non_neg.max.load(Ordering::Acquire).saturating_sub(1) as i32
    }

    pub fn is_set(&self, idx: i32) -> bool {
        if idx >= 0 {
            self.non_neg.is_set(idx as usize)
        } else {
            self.neg.is_set((-idx) as usize)
        }
    }

    pub fn iter_range(&self) -> impl Iterator<Item = i32> {
        (self.min()..=self.max())
            .filter(move |idx| self.is_set(*idx))
            .collect::<Vec<_>>() // Collect to Vec to avoid borrowing `self` in the iterator
            .into_iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZero,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
    };

    use super::Grove;

    #[test]
    fn test_locate() {
        let vec = Grove::<i32>::new();
        assert_eq!(vec.locate(0), (0, 0));
        assert_eq!(vec.locate(1), (1, 0));
        assert_eq!(vec.locate(2), (1, 1));
        assert_eq!(vec.locate(3), (2, 0));
        assert_eq!(vec.locate(4), (2, 1));
        assert_eq!(vec.locate(5), (2, 2));
        assert_eq!(vec.locate(6), (2, 3));
        assert_eq!(vec.locate(7), (3, 0));
        assert_eq!(vec.locate(8), (3, 1));
        assert_eq!(vec.locate(9), (3, 2));
        assert_eq!(vec.locate(10), (3, 3));
        assert_eq!(vec.locate(11), (3, 4));
        assert_eq!(vec.locate(12), (3, 5));
        assert_eq!(vec.locate(13), (3, 6));
        assert_eq!(vec.locate(14), (3, 7));
        assert_eq!(vec.locate(15), (4, 0));
        assert_eq!(vec.locate(16), (4, 1));
        assert_eq!(vec.locate(17), (4, 2));
        // This should be good enough
    }

    #[test]
    fn test_insert_get() {
        let v = Grove::<i32>::new();
        assert!(v.get(42).is_none());
        v.insert(42, 42);
        assert_eq!(v.get(42), Some(&42));
    }

    #[test]
    fn test_requires_drop() {
        use std::sync::Arc;

        static ACTIVE_ALLOCS: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;

        impl DropCounter {
            fn new() -> Self {
                ACTIVE_ALLOCS.fetch_add(1, Ordering::Relaxed);
                Self
            }
        }

        impl Drop for DropCounter {
            fn drop(&mut self) {
                ACTIVE_ALLOCS.fetch_sub(1, Ordering::Relaxed);
            }
        }

        let v = Arc::new(Grove::<DropCounter>::new());
        assert_eq!(ACTIVE_ALLOCS.load(Ordering::Relaxed), 0);

        let num_threads = 16;
        let inserts_per_thread = 1000;

        thread::scope(|s| {
            for thread_id in 0..num_threads {
                let v = Arc::clone(&v);
                s.spawn(move || {
                    for i in 0..inserts_per_thread {
                        v.insert(thread_id * inserts_per_thread + i, DropCounter::new());
                    }
                });
            }
        });

        assert_eq!(
            ACTIVE_ALLOCS.load(Ordering::Relaxed),
            num_threads * inserts_per_thread
        );

        drop(v);

        assert_eq!(ACTIVE_ALLOCS.load(Ordering::Relaxed), 0);
    }

    fn high_contention(num_threads: usize) {
        use crate::std_or_loom::{sync::Arc, thread};

        let vec = Arc::new(Grove::<usize>::new());

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let vec = Arc::clone(&vec);
                thread::spawn(move || {
                    for i in 0..10 {
                        vec.insert(thread_id * 10 + i, thread_id);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        for thread_id in 0..num_threads {
            for i in 0..10 {
                assert_eq!(
                    vec.get(thread_id * 10 + i),
                    Some(&thread_id),
                    "Value mismatch at index {}",
                    thread_id * 10 + i
                );
            }
        }

        // We don't care about memory issues when dropping
        #[cfg(feature = "loom")]
        loom::stop_exploring();
    }

    #[test]
    fn test_high_contention() {
        high_contention(
            std::thread::available_parallelism()
                .ok()
                .map(NonZero::get)
                .unwrap_or(4),
        );
    }

    // This test is only run with the `loom` feature enabled. Make sure not to run any other tests
    // in that case, as they will fail if not executed under `loom::model`.
    #[cfg(feature = "loom")]
    #[test]
    fn test_loom() {
        loom::model(|| high_contention(2));
    }
}
