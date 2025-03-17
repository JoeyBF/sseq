use std::{
    cell::UnsafeCell,
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZero,
};

use crate::std_or_loom::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

const MAX_NUM_BLOCKS: usize = 32;

/// An insert-only sparse vector with pinned elements and geometrically growing capacity.
pub struct GeoVec<V> {
    blocks: [Block<V>; MAX_NUM_BLOCKS],
}

impl<V> GeoVec<V> {
    /// Creates a new empty `GeoVec`.
    pub fn new() -> Self {
        let blocks = std::array::from_fn(|_| Block::new());
        Self { blocks }
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
    pub fn insert(&self, index: usize, value: V) {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        // Safety: We just initialized the block, and `locate` only returns valid indices
        unsafe { self.blocks[block_num].insert(block_offset, value) };
    }

    /// Return the value at the given index
    pub fn get(&self, index: usize) -> Option<&V> {
        let (block_num, block_offset) = self.locate(index);
        self.ensure_init(block_num);
        // Safety: We just initialized the block, and `locate` only returns valid indices
        unsafe { self.blocks[block_num].get(block_offset) }
    }
}

/// An allocation that can store a fixed number of elements.
pub struct Block<V> {
    /// The number of elements in the block.
    len: AtomicUsize,

    /// A pointer to the data buffer.
    ///
    /// If `size` is nonzero, this points to a slice of `WriteOnce<V>` of that size. If `size` is
    /// zero, this is a null pointer.
    data: AtomicPtr<WriteOnce<V>>,
}

impl<V> Block<V> {
    /// Create a new block.
    fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            data: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Initialize the block with a given size.
    ///
    /// # Safety
    ///
    /// For any given block, this method must always be called with the same size.
    unsafe fn init(&self, size: NonZero<usize>) {
        if self.data.load(Ordering::Acquire).is_null() {
            // We need to initialize the block
            let mut data_buffer = ManuallyDrop::new(Vec::with_capacity(size.get()));
            for _ in 0..size.get() {
                data_buffer.push(WriteOnce::none());
            }
            let data_ptr = data_buffer.as_mut_ptr();

            // We can use `Relaxed` here because we will release-store the data pointer, and so any
            // aquire-load of the data pointer will also see the instructions before it, in
            // particular this store.
            self.len.store(size.get(), Ordering::Relaxed);

            // `Release` means that any thread that sees the data pointer will also see the size. We
            // can use `Relaxed` for the failure case because we don't need to synchronize with any
            // other atomic operation.
            if self
                .data
                .compare_exchange(
                    std::ptr::null_mut(),
                    data_ptr,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                // Another thread initialized the block before us
                // Safety: the block has been initialized
                unsafe { ManuallyDrop::drop(&mut data_buffer) };
            }
        }
    }

    /// Insert a value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the block has been initialized.
    unsafe fn insert(&self, index: usize, value: V) {
        let data_ptr = self.data.load(Ordering::Acquire);
        let len = self.len.load(Ordering::Acquire);
        // Safety: the block has been initialized
        assert!(len > 0);
        assert!(!data_ptr.is_null());
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr, len) };
        data[index].set(value);
    }

    /// Return the value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the block has been initialized.
    unsafe fn get(&self, index: usize) -> Option<&V> {
        let len = self.len.load(Ordering::Acquire);
        let data_ptr = self.data.load(Ordering::Acquire);
        // Safety: the block has been initialized
        let data = std::slice::from_raw_parts_mut(data_ptr, len);
        // Safety: the index is within the allocation
        data.get(index).and_then(|w| w.get())
    }
}

impl<V> Drop for Block<V> {
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Relaxed);
        let data_ptr = self.data.load(Ordering::Relaxed);
        if !data_ptr.is_null() {
            // Safety: initialization stores a pointer that came from exactly such a vector
            unsafe { Vec::from_raw_parts(data_ptr, len, len) };
            // vector is dropped here
        }
    }
}

/// An atomic write-once cell.
struct WriteOnce<T> {
    is_some: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> WriteOnce<T> {
    /// Create a new `WriteOnce` with no value.
    fn none() -> Self {
        Self {
            is_some: AtomicU8::new(WriteOnceState::Uninit as u8),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Set the value of the `WriteOnce`.
    fn set(&self, value: T) {
        // Initially, `is_some` is `Uninit`, so it's impossible to observe anything else without a
        // prior `set`. Therefore, we will never panic if `set` was never called.
        //
        // However, we have no guarantee of observing `Init` if some other thread recently called
        // `set`. If so, the `Ok` branch will silently replace the value. This may be confusing if,
        // between the `compare_exchange` and the `write`, some other thread calls `get` and
        // receives a reference. The reference will not be dangling, but will instead point to the
        // value we just wrote. This is fine because the reference points to the contents of an
        // `UnsafeCell`, which explicitly allows mutation through shared references.
        match self.is_some.compare_exchange(
            WriteOnceState::Uninit as u8,
            WriteOnceState::Writing as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                unsafe { self.value.get().write(MaybeUninit::new(value)) }
                // This store creates a happens-before relationship with the load in `get`
                self.is_some
                    .store(WriteOnceState::Init as u8, Ordering::Release);
            }
            Err(_) => panic!("WriteOnce already set"),
        }
    }

    /// Get the value of the `WriteOnce`.
    fn get(&self) -> Option<&T> {
        if self.is_some.load(Ordering::Acquire) == WriteOnceState::Init as u8 {
            // Safety: the value is initialized
            let value = unsafe { (&*self.value.get()).assume_init_ref() };
            Some(&value)
        } else {
            None
        }
    }
}

impl<T> Drop for WriteOnce<T> {
    fn drop(&mut self) {
        // We have an exclusive reference to `self`, so we know that no other thread is accessing
        // it. Moreover, we also have a happens-before relationship with all other operations on
        // this `WriteOnce`, including a possible `set` that initialized the value. Therefore, the
        // following code will never lead to a memory leak.
        if self.is_some.load(Ordering::Relaxed) == WriteOnceState::Init as u8 {
            // Safety: the value is initialized
            unsafe { self.value.get_mut().assume_init_drop() };
        }
    }
}

/// The possible states of a `WriteOnce`.
///
/// We distinguish between `Uninit` and `Writing` so that we reach the `Err` branch of `set` if
/// `set` has been called by any thread before.
///
/// We distinguish between `Writing` and `Init` so that loading `Init` has a happens-before
/// relationship with the write in `set`.
#[repr(u8)]
enum WriteOnceState {
    Uninit = 0,
    Writing = 1,
    Init = 2,
}

#[cfg(test)]
mod tests {
    use std::{
        num::NonZero,
        sync::atomic::{AtomicUsize, Ordering},
        thread,
    };

    use super::GeoVec;

    #[test]
    fn test_locate() {
        let vec = GeoVec::<i32>::new();
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
        let v = GeoVec::<i32>::new();
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

        let v = Arc::new(GeoVec::<DropCounter>::new());
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

        drop(v);

        assert_eq!(ACTIVE_ALLOCS.load(Ordering::Relaxed), 0);
    }

    fn high_contention(num_threads: usize) {
        use crate::std_or_loom::{sync::Arc, thread};

        let vec = Arc::new(GeoVec::<usize>::new());

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
        loom::model(|| high_contention(3));
    }
}
