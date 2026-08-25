//! Node definitions for the HAM-P multi-resolution tree.

use crate::matrix::{dot_product, normalize, normalize_in_place, ColumnMatrix};
use serde::{Deserialize, Serialize};

/// Terminal Leaf Node containing raw high-fidelity memory vectors ($d_2 = 768$).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafNode {
    pub id: usize,
    pub dim: usize,
    pub capacity: usize,
    pub memory_matrix: ColumnMatrix,
    pub item_ids: Vec<String>,
}

impl LeafNode {
    pub fn new(id: usize, dim: usize, capacity: usize) -> Self {
        Self {
            id,
            dim,
            capacity,
            memory_matrix: ColumnMatrix::new(dim, capacity),
            item_ids: Vec::with_capacity(capacity),
        }
    }

    /// Append a new item and vector to the leaf memory matrix.
    #[inline]
    pub fn push(&mut self, id: &str, vector: &[f32]) {
        self.memory_matrix.push_col(vector);
        self.item_ids.push(id.to_string());
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.memory_matrix.len() > self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.memory_matrix.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.memory_matrix.is_empty()
    }

    /// Compute sum vector across all columns up to dim.
    pub fn compute_sum_vector(&self, dim: usize) -> Vec<f32> {
        let mut sum = vec![0.0f32; dim];
        for i in 0..self.len() {
            let col = self.memory_matrix.get_col_slice(i, dim);
            for (s, &x) in sum.iter_mut().zip(col.iter()) {
                *s += x;
            }
        }
        sum
    }

    /// Compute local cluster dispersion (variance of cosine distance to mean centroid).
    pub fn compute_dispersion(&self) -> f32 {
        let count = self.len();
        if count <= 1 {
            return 0.0;
        }
        // Compute mean
        let mut mean = vec![0.0f32; self.dim];
        for i in 0..count {
            let col = self.memory_matrix.get_col(i);
            for (m, &x) in mean.iter_mut().zip(col.iter()) {
                *m += x;
            }
        }
        normalize_in_place(&mut mean);

        // Compute average cosine distance (1 - cos_sim)
        let mut sum_dist = 0.0f32;
        for i in 0..count {
            let col = self.memory_matrix.get_col(i);
            let sim = dot_product(col, &mean);
            sum_dist += (1.0 - sim).max(0.0);
        }
        sum_dist / (count as f32)
    }
}

/// Intermediate Routing Node containing child prototype vectors ($d_0 = 64$ or $d_1 = 128$).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingNode {
    pub id: usize,
    pub dim: usize,
    pub prototypes: ColumnMatrix,     // Unit-normalized for SIMD dot products: dim * num_children
    pub sum_prototypes: ColumnMatrix, // Raw sum vector: dim * num_children (for exact running centroids)
    pub children: Vec<HAMPNode>,
    pub child_counts: Vec<usize>,     // Subtree item counts
}

impl RoutingNode {
    pub fn new(id: usize, dim: usize) -> Self {
        Self {
            id,
            dim,
            prototypes: ColumnMatrix::new(dim, 4),
            sum_prototypes: ColumnMatrix::new(dim, 4),
            children: Vec::with_capacity(4),
            child_counts: Vec::with_capacity(4),
        }
    }

    /// Add a child node with an initial prototype vector / sum vector.
    pub fn add_child(&mut self, child: HAMPNode, sum_vec: &[f32], initial_count: usize) {
        assert_eq!(sum_vec.len(), self.dim);
        let normalized = normalize(sum_vec);
        self.prototypes.push_col(&normalized);
        self.sum_prototypes.push_col(sum_vec);
        self.children.push(child);
        self.child_counts.push(initial_count);
    }

    /// Find the index of the child with highest inner product to `query_slice`.
    #[inline]
    pub fn best_child(&self, query_slice: &[f32]) -> usize {
        let n_children = self.children.len();
        debug_assert!(n_children > 0, "RoutingNode has no children");
        if n_children == 1 {
            return 0;
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for col_idx in 0..n_children {
            let proto = self.prototypes.get_col(col_idx);
            let score = dot_product(proto, query_slice);
            if score > best_score {
                best_score = score;
                best_idx = col_idx;
            }
        }
        best_idx
    }

    /// Update child prototype using exact running sum:
    /// sum_new = sum_old + x_new
    /// proto_new = norm(sum_new)
    #[inline]
    pub fn update_running_centroid(&mut self, child_idx: usize, new_vec_slice: &[f32]) {
        let sum_col = self.sum_prototypes.get_col_mut(child_idx);
        for (s, &x) in sum_col.iter_mut().zip(new_vec_slice.iter()) {
            *s += x;
        }
        self.child_counts[child_idx] += 1;

        // Synchronize normalized prototype
        let proto_col = self.prototypes.get_col_mut(child_idx);
        proto_col.copy_from_slice(sum_col);
        normalize_in_place(proto_col);
    }

    /// Replace child prototype directly (e.g. after a re-clustering or split).
    pub fn set_child_prototype(&mut self, child_idx: usize, new_sum: &[f32], new_count: usize) {
        assert_eq!(new_sum.len(), self.dim);
        let sum_col = self.sum_prototypes.get_col_mut(child_idx);
        sum_col.copy_from_slice(new_sum);

        let proto_col = self.prototypes.get_col_mut(child_idx);
        proto_col.copy_from_slice(new_sum);
        normalize_in_place(proto_col);

        self.child_counts[child_idx] = new_count;
    }

    /// Count all items stored in subtree children.
    pub fn count_items(&self) -> usize {
        self.children.iter().map(|c| c.count_items()).sum()
    }
}

/// Generic node in the HAM-P hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HAMPNode {
    Routing(RoutingNode),
    Leaf(LeafNode),
}

impl HAMPNode {
    pub fn is_leaf(&self) -> bool {
        matches!(self, HAMPNode::Leaf(_))
    }

    pub fn as_leaf(&self) -> Option<&LeafNode> {
        match self {
            HAMPNode::Leaf(ref leaf) => Some(leaf),
            _ => None,
        }
    }

    pub fn as_leaf_mut(&mut self) -> Option<&mut LeafNode> {
        match self {
            HAMPNode::Leaf(ref mut leaf) => Some(leaf),
            _ => None,
        }
    }

    pub fn as_routing(&self) -> Option<&RoutingNode> {
        match self {
            HAMPNode::Routing(ref routing) => Some(routing),
            _ => None,
        }
    }

    pub fn as_routing_mut(&mut self) -> Option<&mut RoutingNode> {
        match self {
            HAMPNode::Routing(ref mut routing) => Some(routing),
            _ => None,
        }
    }

    pub fn count_items(&self) -> usize {
        match self {
            HAMPNode::Leaf(leaf) => leaf.len(),
            HAMPNode::Routing(routing) => routing.children.iter().map(|c| c.count_items()).sum(),
        }
    }

    pub fn count_leaves(&self) -> usize {
        match self {
            HAMPNode::Leaf(_) => 1,
            HAMPNode::Routing(routing) => routing.children.iter().map(|c| c.count_leaves()).sum(),
        }
    }

    pub fn max_depth(&self) -> usize {
        match self {
            HAMPNode::Leaf(_) => 1,
            HAMPNode::Routing(routing) => {
                1 + routing.children.iter().map(|c| c.max_depth()).max().unwrap_or(0)
            }
        }
    }

    pub fn collect_leaf_sizes(&self, sizes: &mut Vec<usize>) {
        match self {
            HAMPNode::Leaf(leaf) => sizes.push(leaf.len()),
            HAMPNode::Routing(routing) => {
                for c in &routing.children {
                    c.collect_leaf_sizes(sizes);
                }
            }
        }
    }
}
