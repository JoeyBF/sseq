use std::{mem::ManuallyDrop, num::NonZero, sync::atomic::Ordering};

use crate::{
    std_or_loom::{
        sync::atomic::{AtomicPtr, AtomicUsize},
        GetMut,
    },
    write_once::WriteOnce,
};

/// An allocation that can store a fixed number of elements.
#[derive(Debug)]
pub struct Block<T> {
    /// The number of elements in the block.
    len: AtomicUsize,

    /// A pointer to the data buffer.
    ///
    /// If `size` is nonzero, this points to a slice of `WriteOnce<V>` of that size. If `size` is
    /// zero, this is a null pointer.
    data: AtomicPtr<WriteOnce<T>>,
}

impl<T> Block<T> {
    /// Create a new block.
    pub(super) fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            data: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub(super) fn is_init(&self) -> bool {
        !self.data.load(Ordering::Acquire).is_null()
    }

    pub(super) fn data(&self) -> &AtomicPtr<WriteOnce<T>> {
        &self.data
    }

    /// Initialize the block with a given size.
    ///
    /// # Safety
    ///
    /// For any given block, this method must always be called with the same size.
    pub(super) unsafe fn init(&self, size: NonZero<usize>) {
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
    pub(super) unsafe fn insert(&self, index: usize, value: T) {
        let data_ptr = self.data.load(Ordering::Acquire);
        let len = self.len.load(Ordering::Acquire);
        // Safety: the block has been initialized
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        data[index].set(value);
    }

    /// Attempt to insert a value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the block has been initialized.
    pub(super) unsafe fn try_insert(&self, index: usize, value: T) -> Result<(), T> {
        let data_ptr = self.data.load(Ordering::Acquire);
        let len = self.len.load(Ordering::Acquire);
        // Safety: the block has been initialized
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        data[index].try_set(value)
    }

    /// Return the value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the block has been initialized.
    pub(super) unsafe fn get(&self, index: usize) -> Option<&T> {
        let len = self.len.load(Ordering::Acquire);
        let data_ptr = self.data.load(Ordering::Acquire);
        // Safety: the block has been initialized
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        // Safety: the index is within the allocation
        data.get(index).and_then(|w| w.get())
    }

    /// Return a mutable reference to the value at the given index.
    ///
    /// This is safe because we hold an exclusive reference.
    pub(super) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let len = self.len.get_by_mut();
        let data_ptr = self.data.get_by_mut();
        // Safety: the block has been initialized
        let data = unsafe { std::slice::from_raw_parts_mut(data_ptr, len) };
        // Safety: the index is within the allocation
        data.get_mut(index).and_then(|w| w.get_mut())
    }

    /// Return the value at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the block has been initialized.
    pub(super) fn is_set(&self, index: usize) -> bool {
        let len = self.len.load(Ordering::Acquire);
        let data_ptr = self.data.load(Ordering::Acquire);
        // Safety: the block has been initialized
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        // Safety: the index is within the allocation
        data.get(index).is_some_and(|w| w.is_set())
    }
}

impl<T> Drop for Block<T> {
    fn drop(&mut self) {
        let len = self.len.get_by_mut();
        let data_ptr = self.data.get_by_mut();
        if !data_ptr.is_null() {
            // Safety: initialization stores a pointer that came from exactly such a vector
            unsafe { Vec::from_raw_parts(data_ptr, len, len) };
            // vector is dropped here
        }
    }
}
