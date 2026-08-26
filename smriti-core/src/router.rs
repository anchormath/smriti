//! Multi-resolution prefix router and nearest-neighbor search with multi-probe beam routing.

use crate::matrix::{dot_product, normalized_prefix_slice};
use crate::node::{HAMPNode, LeafNode};
use serde::{Deserialize, Serialize};

/// Query retrieval result containing project item ID and cosine similarity score.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryResult {
    pub item_id: String,
    pub score: f32,
}

/// Search a LeafNode memory matrix and accumulate top candidates.
#[inline]
pub fn search_leaf(leaf: &LeafNode, query_d2: &[f32], candidates: &mut Vec<QueryResult>) {
    let count = leaf.len();
    for i in 0..count {
        let col = leaf.memory_matrix.get_col(i);
        let score = dot_product(col, query_d2);
        candidates.push(QueryResult {
            item_id: leaf.item_ids[i].clone(),
            score,
        });
    }
}

/// Top-down greedy prefix routing through the HAMP tree.
///
/// `query`: Full high-dimensional query vector ($d_2 = 768$).
/// Guarantees that query slices are strictly unit-normalized before inner product evaluation.
pub fn route_greedy<'a>(root: &'a HAMPNode, query: &[f32]) -> Option<&'a LeafNode> {
    let mut current = root;

    loop {
        match current {
            HAMPNode::Leaf(leaf) => return Some(leaf),
            HAMPNode::Routing(routing) => {
                if routing.children.is_empty() {
                    return None;
                }
                let dim = routing.dim;
                let query_slice = normalized_prefix_slice(query, dim);
                let best_idx = routing.best_child(&query_slice);
                current = &routing.children[best_idx];
            }
        }
    }
}

/// Top-down Beam Search prefix routing through the HAMP tree.
pub fn route_beam<'a>(
    root: &'a HAMPNode,
    query: &[f32],
    beam_width: usize,
) -> Vec<&'a LeafNode> {
    route_multi_probe(root, query, beam_width, 0.0)
}

/// Top-down Multi-Probe Beam Search prefix routing with delta-margin thresholding.
///
/// Parameters:
/// - `beam_width`: Maximum number of branches to keep at each routing level.
/// - `score_delta`: Score distance margin relative to the winning branch. All branches with
///   `score >= max_score - score_delta` are explored (up to `beam_width`), eliminating
///   boundary recall drops on cluster edges.
pub fn route_multi_probe<'a>(
    root: &'a HAMPNode,
    query: &[f32],
    beam_width: usize,
    score_delta: f32,
) -> Vec<&'a LeafNode> {
    if beam_width <= 1 && score_delta <= 0.0 {
        return route_greedy(root, query).into_iter().collect();
    }

    let mut current_nodes: Vec<(&'a HAMPNode, f32)> = vec![(root, 1.0)];

    loop {
        let mut all_leaves = true;
        let mut next_candidates: Vec<(&'a HAMPNode, f32)> =
            Vec::with_capacity(current_nodes.len() * 4);

        for (node, parent_score) in current_nodes {
            match node {
                HAMPNode::Leaf(_) => {
                    next_candidates.push((node, parent_score));
                }
                HAMPNode::Routing(routing) => {
                    all_leaves = false;
                    let dim = routing.dim;
                    let query_slice = normalized_prefix_slice(query, dim);
                    let mut child_scores = Vec::with_capacity(routing.children.len());
                    let mut max_score = f32::NEG_INFINITY;

                    for (child_idx, child) in routing.children.iter().enumerate() {
                        let proto = routing.prototypes.get_col(child_idx);
                        let score = dot_product(proto, &query_slice);
                        if score > max_score {
                            max_score = score;
                        }
                        child_scores.push((child, score));
                    }

                    // Sort children descending by similarity
                    child_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let cutoff = max_score - score_delta.max(0.0);

                    for (rank, (child, score)) in child_scores.into_iter().enumerate() {
                        if rank < beam_width || score >= cutoff {
                            next_candidates.push((child, score));
                        }
                    }
                }
            }
        }

        if all_leaves || next_candidates.is_empty() {
            // Sort by branch score descending and take top beam_width leaves
            next_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut leaves = Vec::new();
            for (node, _) in next_candidates.into_iter().take(beam_width.max(1)) {
                if let Some(leaf) = node.as_leaf() {
                    // Deduplicate leaves by id pointer
                    if !leaves.iter().any(|&l: &&LeafNode| l.id == leaf.id) {
                        leaves.push(leaf);
                    }
                }
            }
            return leaves;
        }

        // Keep top beam_width candidate nodes for next depth level
        next_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        next_candidates.truncate(beam_width.max(1));
        current_nodes = next_candidates;
    }
}

/// Retrieve Top-K nearest neighbors from the tree given a query vector using multi-probe beam search.
pub fn query_tree(
    root: &HAMPNode,
    query: &[f32],
    top_k: usize,
    beam_width: usize,
) -> Vec<QueryResult> {
    query_tree_multi_probe(root, query, top_k, beam_width, 0.0)
}

/// Retrieve Top-K nearest neighbors using configurable multi-probe delta margin.
pub fn query_tree_multi_probe(
    root: &HAMPNode,
    query: &[f32],
    top_k: usize,
    beam_width: usize,
    score_delta: f32,
) -> Vec<QueryResult> {
    let leaves = route_multi_probe(root, query, beam_width, score_delta);
    if leaves.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for leaf in leaves {
        search_leaf(leaf, query, &mut candidates);
    }

    // Sort candidates descending by cosine score
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate if multiple beam paths reached the same item
    candidates.dedup_by(|a, b| a.item_id == b.item_id);

    candidates.truncate(top_k);
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::RoutingNode;

    #[test]
    fn test_multi_probe_delta_routing() {
        // Create 2 leaves with prototypes near [1, 0] and [0.95, 0.31] (close boundary)
        let mut leaf1 = LeafNode::new(1, 2, 10);
        leaf1.push("item_1a", &[1.0, 0.0]);

        let mut leaf2 = LeafNode::new(2, 2, 10);
        leaf2.push("item_2a", &[0.95, 0.31]);

        let mut routing = RoutingNode::new(0, 2);
        routing.add_child(HAMPNode::Leaf(leaf1), &[1.0, 0.0], 1);
        routing.add_child(HAMPNode::Leaf(leaf2), &[0.95, 0.31], 1);

        let root = HAMPNode::Routing(routing);

        // Boundary query
        let q = vec![0.98, 0.15];

        // Greedy routing only finds 1 leaf
        let single = route_greedy(&root, &q);
        assert!(single.is_some());

        // Multi-probe with delta = 0.1 finds BOTH leaves
        let multi = route_multi_probe(&root, &q, 2, 0.1);
        assert_eq!(multi.len(), 2);
    }
}
