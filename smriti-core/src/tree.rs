//! Main HAMPTree container with online insertion, running centroids, multi-probe routing, and Hopfield dynamics.

use crate::hopfield::{adaptive_density_beta, compute_lyapunov_energy, continuous_hopfield_retrieve, resolution_to_beta, HopfieldResult};
use crate::matrix::{compute_centroid, normalized_prefix_slice, normalize_in_place, ColumnMatrix};
use crate::node::{HAMPNode, LeafNode};
use crate::router::{query_tree, query_tree_multi_probe, route_multi_probe, QueryResult};
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
    /// Strictly guarantees unit-normalized prefix slicing for routing and running centroid updates.
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
                let vec_d0 = normalized_prefix_slice(vector, d0);
                let best_child_idx = root_routing.best_child(&vec_d0);
                root_routing.update_running_centroid(best_child_idx, &vec_d0);

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
                    let vec_d1 = normalized_prefix_slice(vector, d1);
                    let best_leaf_idx = l1_routing.best_child(&vec_d1);
                    l1_routing.update_running_centroid(best_leaf_idx, &vec_d1);

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

    /// Query the tree using multi-probe beam search with delta-margin thresholding.
    pub fn query_multi_probe(
        &self,
        query_vector: &[f32],
        top_k: usize,
        beam_width: usize,
        score_delta: f32,
    ) -> Vec<QueryResult> {
        assert_eq!(
            query_vector.len(),
            self.d2,
            "Query vector dimension {} does not match tree d2 {}",
            query_vector.len(),
            self.d2
        );
        query_tree_multi_probe(&self.root, query_vector, top_k, beam_width, score_delta)
    }

    /// Execute Continuous Hopfield associative retrieval using the Resolution Dial R in [0.0, 1.0].
    pub fn query_hopfield(
        &self,
        query_vector: &[f32],
        resolution: f32,
        beam_width: usize,
        score_delta: f32,
        max_steps: usize,
    ) -> HopfieldResult {
        assert_eq!(
            query_vector.len(),
            self.d2,
            "Query vector dimension {} does not match tree d2 {}",
            query_vector.len(),
            self.d2
        );

        let leaves = route_multi_probe(&self.root, query_vector, beam_width, score_delta);
        if leaves.is_empty() {
            return HopfieldResult {
                retrieved_vector: query_vector.to_vec(),
                final_energy: 0.0,
                energy_trajectory: vec![0.0],
                weights: Vec::new(),
                candidate_ids: Vec::new(),
            };
        }

        let total_cols: usize = leaves.iter().map(|l| l.len()).sum();
        let mut candidate_pool = ColumnMatrix::new(self.d2, total_cols);
        let mut candidate_ids = Vec::with_capacity(total_cols);
        let mut avg_dispersion = 0.0f32;

        for leaf in &leaves {
            avg_dispersion += leaf.compute_dispersion();
            for i in 0..leaf.len() {
                candidate_pool.push_col(leaf.memory_matrix.get_col(i));
                candidate_ids.push(leaf.item_ids[i].clone());
            }
        }
        avg_dispersion /= leaves.len() as f32;

        let base_beta = resolution_to_beta(resolution, 0.5, 30.0);
        let effective_beta = adaptive_density_beta(base_beta, avg_dispersion, 0.1);

        continuous_hopfield_retrieve(
            &candidate_pool,
            &candidate_ids,
            query_vector,
            effective_beta,
            max_steps,
            1e-5,
        )
    }

    /// Compute the Lyapunov energy of a continuous state vector across the memory space.
    pub fn compute_lyapunov_energy(&self, state: &[f32], beta: f32) -> f32 {
        let mut all_vecs = Vec::new();
        Self::collect_all_vectors(&self.root, &mut all_vecs);
        let mut mat = ColumnMatrix::new(self.d2, all_vecs.len());
        for v in &all_vecs {
            mat.push_col(v);
        }
        compute_lyapunov_energy(&mat, state, beta)
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

    /// Batch parallel Hopfield query evaluation across CPU cores using `rayon`.
    pub fn query_hopfield_batch(
        &self,
        queries: &[Vec<f32>],
        resolution: f32,
        beam_width: usize,
        score_delta: f32,
        max_steps: usize,
    ) -> Vec<HopfieldResult> {
        queries
            .par_iter()
            .map(|q| self.query_hopfield(q, resolution, beam_width, score_delta, max_steps))
            .collect()
    }

    /// Lazy re-centering task: recalculates exact sum vectors and normalized prototypes
    /// for all routing nodes directly from child leaf memories.
    pub fn recenter(&mut self) {
        Self::recenter_recursive(&mut self.root);
    }

    fn recenter_recursive(node: &mut HAMPNode) {
        if let HAMPNode::Routing(routing) = node {
            let dim = routing.dim;
            for child_idx in 0..routing.children.len() {
                let child = &mut routing.children[child_idx];
                Self::recenter_recursive(child);

                let mut subtree_vecs = Vec::new();
                Self::collect_all_vectors(child, &mut subtree_vecs);

                let count = subtree_vecs.len();
                let mut exact_sum = vec![0.0f32; dim];
                for v in &subtree_vecs {
                    let mut prefix = v[..dim].to_vec();
                    normalize_in_place(&mut prefix);
                    for (s, &x) in exact_sum.iter_mut().zip(prefix.iter()) {
                        *s += x;
                    }
                }
                routing.set_child_prototype(child_idx, &exact_sum, count);
            }
        }
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
                    let normalized_prefixes: Vec<Vec<f32>> = subtree_vecs
                        .iter()
                        .map(|v| normalized_prefix_slice(v, dim))
                        .collect();
                    let slices: Vec<&[f32]> = normalized_prefixes.iter().map(|v| v.as_slice()).collect();
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

    #[test]
    fn test_tree_hopfield_energy_decay() {
        let mut tree = HAMPTree::new(4, 8, 16, 5);

        for i in 0..20 {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            let norm_v = normalize(&v);
            tree.insert(&format!("item_{}", i), &norm_v);
        }

        let mut query = vec![0.0f32; 16];
        query[0] = 0.9;
        query[1] = 0.4;
        let query_norm = normalize(&query);

        let res = tree.query_hopfield(&query_norm, 0.7, 3, 0.1, 10);
        assert_eq!(res.retrieved_vector.len(), 16);
        assert!(!res.energy_trajectory.is_empty());

        for step in 1..res.energy_trajectory.len() {
            assert!(
                res.energy_trajectory[step] <= res.energy_trajectory[step - 1] + 1e-5,
                "Energy non-monotonic: step {} ({}) > step {} ({})",
                step,
                res.energy_trajectory[step],
                step - 1,
                res.energy_trajectory[step - 1]
            );
        }
    }

    #[test]
    fn test_tree_recenter() {
        let mut tree = HAMPTree::new(4, 8, 16, 5);

        for i in 0..30 {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            let norm_v = normalize(&v);
            tree.insert(&format!("item_{}", i), &norm_v);
        }

        let dev_before = tree.validate_running_centroids();
        tree.recenter();
        let dev_after = tree.validate_running_centroids();

        assert!(dev_before < 1e-4);
        assert!(dev_after < 1e-6);
    }
}
