//! Main HAMPTree container with online insertion, running centroids, and rayon batch evaluation.

use crate::matrix::compute_centroid;
use crate::node::{HAMPNode, LeafNode};
use crate::router::{query_tree, QueryResult};
use crate::split::{split_leaf_node, split_leaf_pair};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Summary statistics describing the tree topology and balance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStats {
    pub num_nodes: usize,
    pub num_leaves: usize,
    pub num_items: usize,
    pub depth: usize,
    pub avg_leaf_size: f32,
    pub min_leaf_size: usize,
    pub max_leaf_size: usize,
}

/// Dynamic Hierarchical Partitioned Associative Memory Tree (HAM-P).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HAMPTree {
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub max_leaf_capacity: usize,
    pub root: HAMPNode,
    pub next_node_id: usize,
    pub num_items: usize,
}

impl HAMPTree {
    /// Create a new HAM-P Tree with configured MRL dimensions ($d_0, d_1, d_2$) and leaf capacity ($M_{\max}$).
    pub fn new(d0: usize, d1: usize, d2: usize, max_leaf_capacity: usize) -> Self {
        let root = HAMPNode::Leaf(LeafNode::new(0, d2, max_leaf_capacity));
        Self {
            d0,
            d1,
            d2,
            max_leaf_capacity,
            root,
            next_node_id: 1,
            num_items: 0,
        }
    }

    /// Insert a single project vector into the HAM-P tree online.
    ///
    /// Dynamically routes prefix slices, updates running-mean centroids in $O(\text{depth})$,
    /// and triggers spherical 2-means dynamic splitting if leaf capacity is exceeded.
    pub fn insert(&mut self, item_id: &str, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.d2,
            "Inserted vector dimension {} does not match tree d2 {}",
            vector.len(),
            self.d2
        );

        let d0 = self.d0;
        let d1 = self.d1;
        let max_cap = self.max_leaf_capacity;
        let mut id_counter = self.next_node_id;

        match self.root {
            HAMPNode::Leaf(ref mut leaf) => {
                leaf.push(item_id, vector);
                if leaf.len() > max_cap {
                    let routing = split_leaf_node(leaf.clone(), d0, &mut || {
                        let id = id_counter;
                        id_counter += 1;
                        id
                    });
                    self.root = HAMPNode::Routing(routing);
                }
            }
            HAMPNode::Routing(ref mut root_routing) => {
                let vec_d0 = &vector[..d0];
                let best_child_idx = root_routing.best_child(vec_d0);
                root_routing.update_running_centroid(best_child_idx, vec_d0);

                let child_is_leaf = root_routing.children[best_child_idx].is_leaf();
                if child_is_leaf {
                    let should_split = if let HAMPNode::Leaf(ref mut leaf) =
                        root_routing.children[best_child_idx]
                    {
                        leaf.push(item_id, vector);
                        leaf.len() > max_cap
                    } else {
                        false
                    };

                    if should_split {
                        if let HAMPNode::Leaf(leaf) =
                            root_routing.children.remove(best_child_idx)
                        {
                            let l1_routing = split_leaf_node(leaf, d1, &mut || {
                                let id = id_counter;
                                id_counter += 1;
                                id
                            });
                            root_routing
                                .children
                                .insert(best_child_idx, HAMPNode::Routing(l1_routing));
                        }
                    }
                } else if let HAMPNode::Routing(ref mut l1_routing) =
                    root_routing.children[best_child_idx]
                {
                    let vec_d1 = &vector[..d1];
                    let best_leaf_idx = l1_routing.best_child(vec_d1);
                    l1_routing.update_running_centroid(best_leaf_idx, vec_d1);

                    let should_split = if let HAMPNode::Leaf(ref mut leaf) =
                        l1_routing.children[best_leaf_idx]
                    {
                        leaf.push(item_id, vector);
                        leaf.len() > max_cap
                    } else {
                        false
                    };

                    if should_split {
                        if let HAMPNode::Leaf(leaf) = l1_routing.children.remove(best_leaf_idx)
                        {
                            let (leaf_a, leaf_b) = split_leaf_pair(leaf, &mut || {
                                let id = id_counter;
                                id_counter += 1;
                                id
                            });
                            let sum_a = leaf_a.compute_sum_vector(d1);
                            let sum_b = leaf_b.compute_sum_vector(d1);
                            let count_a = leaf_a.len();
                            let count_b = leaf_b.len();

                            l1_routing
                                .children
                                .insert(best_leaf_idx, HAMPNode::Leaf(leaf_a));
                            l1_routing.set_child_prototype(best_leaf_idx, &sum_a, count_a);
                            l1_routing.add_child(HAMPNode::Leaf(leaf_b), &sum_b, count_b);
                        }
                    }
                }
            }
        }

        self.next_node_id = id_counter;
        self.num_items += 1;
    }

    /// Query the tree for Top-K nearest neighbors using beam search.
    pub fn query(&self, query_vector: &[f32], top_k: usize, beam_width: usize) -> Vec<QueryResult> {
        assert_eq!(
            query_vector.len(),
            self.d2,
            "Query vector dimension {} does not match tree d2 {}",
            query_vector.len(),
            self.d2
        );
        query_tree(&self.root, query_vector, top_k, beam_width)
    }

    /// Batch parallel query evaluation across CPU cores using `rayon`.
    pub fn query_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        beam_width: usize,
    ) -> Vec<Vec<QueryResult>> {
        queries
            .par_iter()
            .map(|q| self.query(q, top_k, beam_width))
            .collect()
    }

    /// Collect leaf sizes to evaluate tree balance.
    pub fn get_leaf_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        self.root.collect_leaf_sizes(&mut sizes);
        sizes
    }

    /// Collect comprehensive tree statistics.
    pub fn stats(&self) -> TreeStats {
        let leaf_sizes = self.get_leaf_sizes();
        let num_leaves = leaf_sizes.len();
        let num_items = self.num_items;
        let avg_leaf_size = if num_leaves > 0 {
            num_items as f32 / num_leaves as f32
        } else {
            0.0
        };
        let min_leaf_size = leaf_sizes.iter().copied().min().unwrap_or(0);
        let max_leaf_size = leaf_sizes.iter().copied().max().unwrap_or(0);
        let depth = self.root.max_depth();

        TreeStats {
            num_nodes: num_leaves + (if depth > 1 { depth } else { 0 }),
            num_leaves,
            num_items,
            depth,
            avg_leaf_size,
            min_leaf_size,
            max_leaf_size,
        }
    }

    /// Validate that running-mean centroids in routing nodes match exact batch centroids.
    /// Returns maximum L2 deviation across all child prototypes.
    pub fn validate_running_centroids(&self) -> f32 {
        let mut max_dev = 0.0f32;
        Self::validate_centroids_recursive(&self.root, &mut max_dev);
        max_dev
    }

    fn validate_centroids_recursive(node: &HAMPNode, max_dev: &mut f32) {
        if let HAMPNode::Routing(routing) = node {
            let dim = routing.dim;
            for (child_idx, child) in routing.children.iter().enumerate() {
                let running_proto = routing.prototypes.get_col(child_idx);

                // Collect all vectors under this subtree
                let mut subtree_vecs: Vec<Vec<f32>> = Vec::new();
                Self::collect_all_vectors(child, &mut subtree_vecs);

                if !subtree_vecs.is_empty() {
                    let slices: Vec<&[f32]> = subtree_vecs.iter().map(|v| &v[..dim]).collect();
                    let exact_proto = compute_centroid(&slices, dim);

                    // Compute L2 difference
                    let mut diff_sq = 0.0f32;
                    for (&a, &b) in running_proto.iter().zip(exact_proto.iter()) {
                        let d = a - b;
                        diff_sq += d * d;
                    }
                    let dev = diff_sq.sqrt();
                    if dev > *max_dev {
                        *max_dev = dev;
                    }
                }

                Self::validate_centroids_recursive(child, max_dev);
            }
        }
    }

    fn collect_all_vectors(node: &HAMPNode, out: &mut Vec<Vec<f32>>) {
        match node {
            HAMPNode::Leaf(leaf) => {
                for i in 0..leaf.len() {
                    out.push(leaf.memory_matrix.get_col(i).to_vec());
                }
            }
            HAMPNode::Routing(routing) => {
                for child in &routing.children {
                    Self::collect_all_vectors(child, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::normalize;

    #[test]
    fn test_tree_insert_and_split() {
        let mut tree = HAMPTree::new(4, 8, 16, 5);

        for i in 0..20 {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            v[(i + 1) % 16] = 0.5;
            let norm_v = normalize(&v);
            tree.insert(&format!("proj_{}", i), &norm_v);
        }

        assert_eq!(tree.num_items, 20);
        let stats = tree.stats();
        assert!(stats.num_leaves > 1, "Tree should have split into multiple leaves");
        assert!(stats.depth <= 3, "Tree depth should be <= 3");
    }

    #[test]
    fn test_tree_query_retrieval() {
        let mut tree = HAMPTree::new(4, 8, 16, 5);

        for i in 0..30 {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            let norm_v = normalize(&v);
            tree.insert(&format!("item_{}", i), &norm_v);
        }

        let mut target = vec![0.0f32; 16];
        target[0] = 1.0;
        let target_norm = normalize(&target);

        let results = tree.query(&target_norm, 5, 2);
        assert!(!results.is_empty());
        assert_eq!(results[0].item_id, "item_0");
        assert!((results[0].score - 1.0).abs() < 1e-4);
    }
}
