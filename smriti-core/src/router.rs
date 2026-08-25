//! Multi-resolution prefix router and nearest-neighbor search.

use crate::matrix::dot_product;
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
/// Prefix slices $x[:d_0]$ and $x[:d_1]$ are zero-copy references.
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
                let query_slice = &query[..dim.min(query.len())];
                let best_idx = routing.best_child(query_slice);
                current = &routing.children[best_idx];
            }
        }
    }
}

/// Top-down Beam Search prefix routing through the HAMP tree.
///
/// Keeps top `beam_width` branches at each level for higher recall on boundary queries.
pub fn route_beam<'a>(
    root: &'a HAMPNode,
    query: &[f32],
    beam_width: usize,
) -> Vec<&'a LeafNode> {
    if beam_width <= 1 {
        return route_greedy(root, query).into_iter().collect();
    }

    let mut current_nodes: Vec<(&'a HAMPNode, f32)> = vec![(root, 1.0)];

    loop {
        let mut all_leaves = true;
        let mut next_candidates: Vec<(&'a HAMPNode, f32)> = Vec::with_capacity(current_nodes.len() * 4);

        for (node, parent_score) in current_nodes {
            match node {
                HAMPNode::Leaf(_) => {
                    next_candidates.push((node, parent_score));
                }
                HAMPNode::Routing(routing) => {
                    all_leaves = false;
                    let dim = routing.dim;
                    let query_slice = &query[..dim.min(query.len())];

                    for (child_idx, child) in routing.children.iter().enumerate() {
                        let proto = routing.prototypes.get_col(child_idx);
                        let score = dot_product(proto, query_slice);
                        next_candidates.push((child, score));
                    }
                }
            }
        }

        if all_leaves || next_candidates.is_empty() {
            // Sort by branch score descending and take top beam_width leaves
            next_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return next_candidates
                .into_iter()
                .take(beam_width)
                .filter_map(|(node, _)| node.as_leaf())
                .collect();
        }

        // Keep top beam_width nodes for next depth level
        next_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        next_candidates.truncate(beam_width);
        current_nodes = next_candidates;
    }
}

/// Retrieve Top-K nearest neighbors from the tree given a query vector.
pub fn query_tree(
    root: &HAMPNode,
    query: &[f32],
    top_k: usize,
    beam_width: usize,
) -> Vec<QueryResult> {
    let leaves = route_beam(root, query, beam_width);
    if leaves.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for leaf in leaves {
        search_leaf(leaf, query, &mut candidates);
    }

    // Sort candidates descending by cosine score
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate if multiple beam paths reached the same leaf
    candidates.dedup_by(|a, b| a.item_id == b.item_id);

    candidates.truncate(top_k);
    candidates
}
