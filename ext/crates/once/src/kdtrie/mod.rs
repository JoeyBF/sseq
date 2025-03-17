// use std::sync::Arc;

// use crate::OnceVec;

use grove::{Grove, WriteOnce};

pub mod grove;

// // /// In our trie, each node either has children or values. Therefore, we can encode both at the same
// // /// time with a list of values of this union type.
// // union ChildOrValue<V> {
// //     child: *mut Node<V>,
// //     // We need to use `ManuallyDrop` because, when this union is dropped, we don't know which
// //     // variant is present, and so we can't know if we need to drop V.
// //     value: ManuallyDrop<V>,
// // }

struct TwoEndedGrove<T> {
    non_neg: Grove<T>,
    neg: Grove<T>,
}

impl<T> TwoEndedGrove<T> {
    fn new() -> Self {
        Self {
            non_neg: Grove::new(),
            neg: Grove::new(),
        }
    }

    fn insert(&self, idx: i32, value: T) {
        if idx >= 0 {
            self.non_neg.insert(idx as usize, value);
        } else {
            self.neg.insert((-idx) as usize, value);
        }
    }

    fn get(&self, idx: i32) -> Option<&T> {
        if idx >= 0 {
            self.non_neg.get(idx as usize)
        } else {
            self.neg.get((-idx) as usize)
        }
    }
}

struct Node<V> {
    children: TwoEndedGrove<Node<V>>,
    value: WriteOnce<V>,
}

impl<V> Node<V> {
    fn new() -> Self {
        Self {
            children: TwoEndedGrove::new(),
            value: WriteOnce::none(),
        }
    }

    fn ensure_child(&self, idx: i32) -> &Node<V> {
        if let Some(child) = self.get_child(idx) {
            child
        } else {
            self.children.insert(idx, Node::new());
            self.get_child(idx).unwrap()
        }
    }

    fn get_child(&self, idx: i32) -> Option<&Node<V>> {
        self.children.get(idx)
    }

    fn get_value(&self) -> Option<&V> {
        self.value.get()
    }

    fn set_value(&self, value: V) {
        self.value.set(value);
    }
}

pub struct ConcurrentNDArray<V> {
    root: Node<V>,
    dimensions: usize,
}

impl<V> ConcurrentNDArray<V> {
    pub fn new(dimensions: usize) -> Self {
        Self {
            root: Node::new(),
            dimensions,
        }
    }

    pub fn insert(&self, coords: &[i32], value: V) {
        assert!(coords.len() == self.dimensions);

        // When's the last time you saw a mutable shared reference?
        let mut node = &self.root;

        for &coord in coords.iter() {
            node = node.ensure_child(coord);
        }

        node.set_value(value);
    }

    pub fn get(&self, coords: &[i32]) -> Option<&V> {
        assert!(coords.len() == self.dimensions);

        let mut node = &self.root;
        for &coord in coords.iter() {
            node = node.get_child(coord)?;
        }

        node.get_value()
    }
}
