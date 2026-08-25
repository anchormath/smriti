//! Spherical 2-Means clustering for dynamic node splitting and bisection.

use crate::matrix::{dot_product, normalize_in_place, ColumnMatrix};

/// Result of spherical 2-means bisection.
#[derive(Debug, Clone)]
pub struct SplitResult {
    /// 0 for cluster A, 1 for cluster B for each item in the matrix
    pub assignments: Vec<usize>,
    /// Unit-normalized centroid for cluster A (dim = matrix.rows)
    pub centroid_a: Vec<f32>,
    /// Unit-normalized centroid for cluster B (dim = matrix.rows)
    pub centroid_b: Vec<f32>,
    /// Number of items in cluster A
    pub count_a: usize,
    /// Number of items in cluster B
    pub count_b: usize,
}

/// Perform spherical 2-means clustering on columns of a ColumnMatrix.
///
/// Uses cosine similarity / dot products on unit-normalized vectors.
/// Guarantees that neither cluster is empty and both centroids are unit-normalized.
pub fn spherical_2means(matrix: &ColumnMatrix, max_iter: usize) -> SplitResult {
    let n = matrix.len();
    let dim = matrix.rows;
    assert!(n >= 2, "Cannot split matrix with fewer than 2 vectors");

    // 1. Deterministic initialization:
    // Centroid A = first vector
    let mut c_a = matrix.get_col(0).to_vec();
    normalize_in_place(&mut c_a);

    // Centroid B = vector with lowest dot product to Centroid A (furthest point)
    let mut min_sim = f32::INFINITY;
    let mut min_idx = 1;
    for i in 1..n {
        let v = matrix.get_col(i);
        let sim = dot_product(v, &c_a);
        if sim < min_sim {
            min_sim = sim;
            min_idx = i;
        }
    }
    let mut c_b = matrix.get_col(min_idx).to_vec();
    normalize_in_place(&mut c_b);

    // If initial points are identical (degenerate), perturb c_b
    if dot_product(&c_a, &c_b) > 0.9999 {
        c_b.reverse();
        normalize_in_place(&mut c_b);
    }

    let mut assignments = vec![0usize; n];
    let mut prev_assignments = vec![usize::MAX; n];

    for _iter in 0..max_iter {
        // --- Assignment step ---
        let mut count_a = 0usize;
        let mut count_b = 0usize;

        for i in 0..n {
            let v = matrix.get_col(i);
            let sim_a = dot_product(v, &c_a);
            let sim_b = dot_product(v, &c_b);

            if sim_a >= sim_b {
                assignments[i] = 0;
                count_a += 1;
            } else {
                assignments[i] = 1;
                count_b += 1;
            }
        }

        // Handle edge case: empty cluster by reassigning furthest item
        if count_a == 0 {
            let mut worst_sim = f32::INFINITY;
            let mut worst_idx = 0;
            for i in 0..n {
                let sim = dot_product(matrix.get_col(i), &c_b);
                if sim < worst_sim {
                    worst_sim = sim;
                    worst_idx = i;
                }
            }
            assignments[worst_idx] = 0;
        } else if count_b == 0 {
            let mut worst_sim = f32::INFINITY;
            let mut worst_idx = 0;
            for i in 0..n {
                let sim = dot_product(matrix.get_col(i), &c_a);
                if sim < worst_sim {
                    worst_sim = sim;
                    worst_idx = i;
                }
            }
            assignments[worst_idx] = 1;
        }

        // Check for convergence
        if assignments == prev_assignments {
            break;
        }
        prev_assignments.copy_from_slice(&assignments);

        // --- Centroid Update step ---
        let mut new_c_a = vec![0.0f32; dim];
        let mut new_c_b = vec![0.0f32; dim];

        for i in 0..n {
            let v = matrix.get_col(i);
            if assignments[i] == 0 {
                for (s, &x) in new_c_a.iter_mut().zip(v.iter()) {
                    *s += x;
                }
            } else {
                for (s, &x) in new_c_b.iter_mut().zip(v.iter()) {
                    *s += x;
                }
            }
        }

        normalize_in_place(&mut new_c_a);
        normalize_in_place(&mut new_c_b);

        c_a = new_c_a;
        c_b = new_c_b;
    }

    let count_a = assignments.iter().filter(|&&a| a == 0).count();
    let count_b = n - count_a;

    SplitResult {
        assignments,
        centroid_a: c_a,
        centroid_b: c_b,
        count_a,
        count_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_2means_partition() {
        let mut mat = ColumnMatrix::new(4, 4);
        // Two points near [1, 0, 0, 0]
        mat.push_col(&[0.99, 0.05, 0.0, 0.0]);
        mat.push_col(&[0.95, 0.10, 0.0, 0.0]);
        // Two points near [0, 0, 1, 0]
        mat.push_col(&[0.0, 0.0, 0.99, 0.05]);
        mat.push_col(&[0.0, 0.0, 0.95, 0.10]);

        let res = spherical_2means(&mat, 20);
        assert_eq!(res.count_a, 2);
        assert_eq!(res.count_b, 2);
        assert_eq!(res.assignments[0], res.assignments[1]);
        assert_eq!(res.assignments[2], res.assignments[3]);
        assert_ne!(res.assignments[0], res.assignments[2]);
    }
}
