//! Dynamic leaf node splitting and transformation into intermediate routing nodes.

use crate::kmeans::spherical_2means;
use crate::node::{HAMPNode, LeafNode, RoutingNode};

/// Split an overflowing LeafNode into two child LeafNodes.
pub fn split_leaf_pair(
    leaf: LeafNode,
    next_node_id: &mut dyn FnMut() -> usize,
) -> (LeafNode, LeafNode) {
    assert!(
        leaf.len() >= 2,
        "Cannot split leaf with fewer than 2 items"
    );

    let split_res = spherical_2means(&leaf.memory_matrix, 25);

    let id_a = next_node_id();
    let id_b = next_node_id();
    let dim = leaf.dim;
    let capacity = leaf.capacity;

    let mut leaf_a = LeafNode::new(id_a, dim, capacity);
    let mut leaf_b = LeafNode::new(id_b, dim, capacity);

    for (i, &assignment) in split_res.assignments.iter().enumerate() {
        let item_id = &leaf.item_ids[i];
        let vec = leaf.memory_matrix.get_col(i);
        if assignment == 0 {
            leaf_a.push(item_id, vec);
        } else {
            leaf_b.push(item_id, vec);
        }
    }

    (leaf_a, leaf_b)
}

/// Split an overflowing LeafNode into a RoutingNode with two child LeafNodes.
pub fn split_leaf_node(
    leaf: LeafNode,
    route_dim: usize,
    next_node_id: &mut dyn FnMut() -> usize,
) -> RoutingNode {
    let parent_id = leaf.id;
    let (leaf_a, leaf_b) = split_leaf_pair(leaf, next_node_id);

    let sum_a = leaf_a.compute_sum_vector(route_dim);
    let sum_b = leaf_b.compute_sum_vector(route_dim);
    let count_a = leaf_a.len();
    let count_b = leaf_b.len();

    let mut routing = RoutingNode::new(parent_id, route_dim);
    routing.add_child(HAMPNode::Leaf(leaf_a), &sum_a, count_a);
    routing.add_child(HAMPNode::Leaf(leaf_b), &sum_b, count_b);

    routing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_leaf_node() {
        let mut leaf = LeafNode::new(1, 4, 10);
        leaf.push("p1", &[1.0, 0.0, 0.0, 0.0]);
        leaf.push("p2", &[0.9, 0.1, 0.0, 0.0]);
        leaf.push("p3", &[0.0, 0.0, 1.0, 0.0]);
        leaf.push("p4", &[0.0, 0.0, 0.9, 0.1]);

        let mut id_counter = 10;
        let routing = split_leaf_node(leaf, 2, &mut || {
            id_counter += 1;
            id_counter
        });

        assert_eq!(routing.children.len(), 2);
        assert_eq!(routing.prototypes.cols, 2);
        assert_eq!(routing.prototypes.rows, 2);
        assert_eq!(routing.count_items(), 4);
    }
}
