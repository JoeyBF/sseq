// use std::sync::Arc;

// use crate::OnceVec;

use std::mem::ManuallyDrop;

use grove::TwoEndedGrove;

mod block;
pub mod grove;
mod write_once;

pub struct MultiGraded<const K: usize, V>(KdTrie<V>);

impl<const K: usize, V> MultiGraded<K, V> {
    pub fn new() -> Self {
        Self(KdTrie::new(K))
    }

    pub fn insert(&self, coords: [i32; K], value: V) {
        self.0.insert(&coords, value);
    }

    pub fn get(&self, coords: [i32; K]) -> Option<&V> {
        self.0.get(&coords)
    }
}

pub struct KdTrie<V> {
    root: Node<V>,
    dimensions: usize,
}

impl<V> KdTrie<V> {
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0);

        let root = if dimensions == 1 {
            Node::new_leaf()
        } else {
            Node::new_inner()
        };

        Self { root, dimensions }
    }

    pub fn insert(&self, coords: &[i32], value: V) {
        assert!(coords.len() == self.dimensions);

        // When's the last time you saw a mutable shared reference?
        let mut node = &self.root;

        for &coord in coords.iter().take(self.dimensions.saturating_sub(2)) {
            node = unsafe { node.ensure_child(coord, Node::new_inner()) };
        }
        if self.dimensions > 1 {
            node = unsafe { node.ensure_child(coords[self.dimensions - 2], Node::new_leaf()) };
        }

        unsafe { node.set_value(coords[self.dimensions - 1], value) };
    }

    pub fn get(&self, coords: &[i32]) -> Option<&V> {
        assert!(coords.len() == self.dimensions);

        let mut node = &self.root;

        for &coord in coords.iter().take(self.dimensions - 1) {
            node = unsafe { node.get_child(coord)? };
        }

        unsafe { node.get_value(coords[self.dimensions - 1]) }
    }
}

impl<V> Drop for KdTrie<V> {
    fn drop(&mut self) {
        self.root.drop_level(self.dimensions, 0);
    }
}

union Node<V> {
    inner: ManuallyDrop<TwoEndedGrove<Node<V>>>,
    leaf: ManuallyDrop<TwoEndedGrove<V>>,
}

impl<V> Node<V> {
    fn new_inner() -> Self {
        Self {
            inner: ManuallyDrop::new(TwoEndedGrove::new()),
        }
    }

    fn new_leaf() -> Self {
        Self {
            leaf: ManuallyDrop::new(TwoEndedGrove::new()),
        }
    }

    /// # Safety
    ///
    /// Can only be called on an inner node.
    unsafe fn ensure_child(&self, idx: i32, to_insert: Node<V>) -> &Node<V> {
        if let Some(child) = self.get_child(idx) {
            child
        } else {
            unsafe { self.inner.insert(idx, to_insert) };
            self.get_child(idx).unwrap()
        }
    }

    /// # Safety
    ///
    /// Can only be called on an inner node.
    unsafe fn get_child(&self, idx: i32) -> Option<&Node<V>> {
        unsafe { self.inner.get(idx) }
    }

    /// # Safety
    ///
    /// Can only be called on a leaf node.
    unsafe fn get_value(&self, idx: i32) -> Option<&V> {
        unsafe { self.leaf.get(idx) }
    }

    /// # Safety
    ///
    /// Can only be called on a leaf node.
    unsafe fn set_value(&self, idx: i32, value: V) {
        unsafe { self.leaf.insert(idx, value) }
    }

    fn drop_level(&mut self, dimensions: usize, level: usize) {
        if level == dimensions {
            return;
        }

        if level == dimensions - 1 {
            // This is a leaf node
            unsafe { ManuallyDrop::drop(&mut self.leaf) };
        } else {
            // This is an inner node
            unsafe {
                for idx in self.inner.iter_range() {
                    self.inner
                        .get_mut(idx)
                        .unwrap()
                        .drop_level(dimensions, level + 1);
                }
                ManuallyDrop::drop(&mut self.inner);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic() {
        let arr = KdTrie::new(3);

        arr.insert(&[1, 2, 3], 42);
        arr.insert(&[1, 2, 4], 43);
        arr.insert(&[1, 3, 3], 44);
        arr.insert(&[1, 3, 4], 45);

        assert_eq!(arr.get(&[1, 2, 3]), Some(&42));
        assert_eq!(arr.get(&[1, 2, 4]), Some(&43));
        assert_eq!(arr.get(&[1, 3, 3]), Some(&44));
        assert_eq!(arr.get(&[1, 3, 4]), Some(&45));
    }

    fn get_n_coords<const K: usize>(n: usize, min: [i32; K]) -> Vec<[i32; K]> {
        (0..)
            .flat_map(|i| crate::get_nth_diagonal::<K>(i))
            .map(|mut v| {
                for (xi, mi) in v.iter_mut().zip(min.iter()) {
                    *xi += mi;
                }
                v
            })
            .take(n)
            .collect()
    }

    #[test]
    fn test_large() {
        let arr = KdTrie::new(8);
        for (idx, coord) in get_n_coords(10_000, [0, 0, 0, 0, 0, 0, 0, 0])
            .iter()
            .enumerate()
        {
            arr.insert(coord, idx);
        }
    }

    #[test]
    fn test_requires_drop() {
        use std::{
            sync::{
                atomic::{AtomicUsize, Ordering},
                Arc,
            },
            thread,
        };

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

        let v = Arc::new(KdTrie::<DropCounter>::new(3));
        assert_eq!(ACTIVE_ALLOCS.load(Ordering::Relaxed), 0);

        let num_threads = 16;
        let inserts_per_thread = 1000;

        thread::scope(|s| {
            for thread_id in 0..num_threads {
                let v = Arc::clone(&v);
                s.spawn(move || {
                    for i in 0..inserts_per_thread {
                        // for j in 0..inserts_per_thread {
                        v.insert(&[thread_id, i, 4], DropCounter::new());
                        // }
                    }
                });
            }
        });

        assert_eq!(
            ACTIVE_ALLOCS.load(Ordering::Relaxed),
            (num_threads * inserts_per_thread.pow(2)) as usize
        );

        drop(v);

        assert_eq!(ACTIVE_ALLOCS.load(Ordering::Relaxed), 0);
    }
}
