// use std::sync::Arc;

// use crate::OnceVec;

pub mod geometric_vector;

// // /// In our trie, each node either has children or values. Therefore, we can encode both at the same
// // /// time with a list of values of this union type.
// // union ChildOrValue<V> {
// //     child: *mut Node<V>,
// //     // We need to use `ManuallyDrop` because, when this union is dropped, we don't know which
// //     // variant is present, and so we can't know if we need to drop V.
// //     value: ManuallyDrop<V>,
// // }

// struct Node<V> {
//     // Geometrically growing vector of optional pointers to child nodes
//     children: OnceVec<Option<Arc<Node<V>>>>,
//     value: Option<V>,
// }

// impl<V> Node<V> {
//     fn new() -> Self {
//         Self {
//             children: OnceVec::new(),
//             value: None,
//         }
//     }

//     fn ensure_child(&self, idx: u32) -> Arc<Node<V>> {
//         let children = self.children;
//         let idx_usize = idx as usize;

//         // if idx_usize >= children.len() {
//         //     let mut new_len = children.len().max(1);
//         //     while new_len <= idx_usize {
//         //         new_len *= 2; // geometric growth
//         //     }
//         //     children.resize_with(new_len, || None);
//         // }

//         if let Some(child) = &children[idx_usize] {
//             child.clone()
//         } else {
//             let new_child = Arc::new(Node::new());
//             children.[idx_usize] = Some(new_child.clone());
//             new_child
//         }
//     }

//     fn get_child(&self, idx: u32) -> Option<Arc<Node<V>>> {
//         let children = self.children.read().unwrap();
//         children.get(idx as usize).and_then(|c| c.clone())
//     }
// }

// struct ConcurrentNDArray<V> {
//     root: Arc<Node<V>>,
//     dimensions: usize,
// }

// impl<V> ConcurrentNDArray<V> {
//     fn new(dimensions: usize) -> Self {
//         Self {
//             root: Arc::new(Node::new()),
//             dimensions,
//         }
//     }

//     fn insert(&self, coords: &[i32], value: V) {
//         assert!(coords.len() == self.dimensions);
//         let mut node = self.root.clone();
//         for &coord in coords.iter() {
//             let idx = encode_coord(coord);
//             node = node.ensure_child(idx);
//         }
//         let mut node_value = node.value.write().unwrap();
//         *node_value = Some(Arc::new(value));
//     }

//     fn get(&self, coords: &[i32]) -> Option<Arc<V>> {
//         assert!(coords.len() == self.dimensions);
//         let mut node = self.root.clone();
//         for &coord in coords.iter() {
//             let idx = encode_coord(coord);
//             match node.get_child(idx) {
//                 Some(next_node) => node = next_node,
//                 None => return None,
//             }
//         }
//         let node_value = node.value.read().unwrap();
//         node_value.clone()
//     }
// }
