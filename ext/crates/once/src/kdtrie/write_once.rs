use std::{cell::UnsafeCell, mem::MaybeUninit, sync::atomic::Ordering};

use crate::std_or_loom::sync::atomic::AtomicU8;

/// An atomic write-once cell.
#[derive(Debug)]
pub struct WriteOnce<T> {
    is_some: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> WriteOnce<T> {
    /// Create a new `WriteOnce` with no value.
    pub const fn none() -> Self {
        Self {
            is_some: AtomicU8::new(WriteOnceState::Uninit as u8),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Set the value of the `WriteOnce`.
    pub fn set(&self, value: T) {
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
            Ordering::Relaxed,
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
    pub fn get(&self) -> Option<&T> {
        if self.is_set() {
            // Safety: the value is initialized
            let value = unsafe { (&*self.value.get()).assume_init_ref() };
            Some(&value)
        } else {
            None
        }
    }

    pub fn is_set(&self) -> bool {
        self.is_some.load(Ordering::Acquire) == WriteOnceState::Init as u8
    }

    /// Get a mutable reference to the value of the `WriteOnce`.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if *self.is_some.get_mut() == WriteOnceState::Init as u8 {
            // Safety: the value is initialized
            let value = unsafe { (&mut *self.value.get()).assume_init_mut() };
            Some(value)
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
        if *self.is_some.get_mut() == WriteOnceState::Init as u8 {
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
