//! Continuous Hopfield Dynamics, Lyapunov Energy Evaluation & Resolution Dial (HAM-R).

use crate::matrix::{dot_product, logsumexp, normalize_in_place, softmax_stable, ColumnMatrix};
use serde::{Deserialize, Serialize};

/// Result of continuous Hopfield associative retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopfieldResult {
    /// Continuous output state vector (dim = memory_matrix.rows)
    pub retrieved_vector: Vec<f32>,
    /// Final Lyapunov energy E(x_out)
    pub final_energy: f32,
    /// Energy decay trajectory [E(x^(0)), E(x^(1)), ..., E(x^(T))]
    pub energy_trajectory: Vec<f32>,
    /// Softmax attention weights over candidate memory vectors
    pub weights: Vec<f32>,
    /// Candidate memory item IDs
    pub candidate_ids: Vec<String>,
}

/// Convert continuous Resolution Dial R in [0.0, 1.0] to inverse temperature beta.
///
/// Uses geometric/exponential interpolation:
/// beta(R) = beta_min * (beta_max / beta_min)^R
///
/// At R = 0.0 -> beta_min (smooth cluster blend)
/// At R = 1.0 -> beta_max (exact isolated precedent recall)
#[inline]
pub fn resolution_to_beta(resolution: f32, beta_min: f32, beta_max: f32) -> f32 {
    let r_clamped = resolution.clamp(0.0, 1.0);
    let ratio = beta_max / beta_min.max(1e-6);
    beta_min * ratio.powf(r_clamped)
}

/// Scale inverse temperature beta based on local cluster dispersion.
///
/// Denser clusters (low dispersion sigma^2) receive higher effective beta to prevent cross-talk,
/// while sparse clusters receive lower beta for smoother blending.
#[inline]
pub fn adaptive_density_beta(base_beta: f32, dispersion: f32, alpha: f32) -> f32 {
    let disp = dispersion.max(0.0);
    base_beta * (1.0 + alpha / (disp + 0.05))
}

/// Compute the exact continuous Lyapunov energy of state `x` relative to memory matrix Xi:
/// E(x) = -1/beta * LSE(beta * Xi^T * x) + 1/2 * ||x||_2^2
pub fn compute_lyapunov_energy(memory_matrix: &ColumnMatrix, state: &[f32], beta: f32) -> f32 {
    let n = memory_matrix.len();
    if n == 0 {
        return 0.5 * dot_product(state, state);
    }

    let mut logits = vec![0.0f32; n];
    for (i, logit) in logits.iter_mut().enumerate() {
        let mem = memory_matrix.get_col(i);
        *logit = dot_product(mem, state);
    }

    let lse = logsumexp(&logits, beta);
    let norm_term = 0.5 * dot_product(state, state);

    - (1.0 / beta.max(1e-6)) * lse + norm_term
}

/// Execute continuous Hopfield associative retrieval on a candidate memory pool.
///
/// Performs iterative energy minimization updates:
/// a^(t) = softmax(beta * Xi^T * x^(t))
/// x^(t+1) = normalize(Xi * a^(t))
///
/// Tracks the Lyapunov energy at each step to guarantee monotonic decay.
pub fn continuous_hopfield_retrieve(
    memory_matrix: &ColumnMatrix,
    item_ids: &[String],
    query: &[f32],
    beta: f32,
    max_steps: usize,
    tol: f32,
) -> HopfieldResult {
    let n_memories = memory_matrix.len();
    let dim = memory_matrix.rows;
    assert_eq!(item_ids.len(), n_memories);
    assert_eq!(query.len(), dim);

    if n_memories == 0 {
        return HopfieldResult {
            retrieved_vector: query.to_vec(),
            final_energy: 0.0,
            energy_trajectory: vec![0.0],
            weights: Vec::new(),
            candidate_ids: Vec::new(),
        };
    }

    let mut current_state = query.to_vec();
    normalize_in_place(&mut current_state);

    let initial_energy = compute_lyapunov_energy(memory_matrix, &current_state, beta);
    let mut energy_trajectory = vec![initial_energy];
    let mut logits = vec![0.0f32; n_memories];
    let mut weights = vec![0.0f32; n_memories];

    let max_iter = max_steps.max(1);

    for _step in 0..max_iter {
        // 1. Compute inner product logits: S_i = <xi_i, x^(t)>
        for (i, logit) in logits.iter_mut().enumerate() {
            let mem = memory_matrix.get_col(i);
            *logit = dot_product(mem, &current_state);
        }

        // 2. Numerically stable softmax: a = softmax(beta * S)
        softmax_stable(&logits, beta, &mut weights);

        // 3. Matrix-vector product: x_next = sum_i (a_i * xi_i)
        let mut next_state = vec![0.0f32; dim];
        for (i, &w) in weights.iter().enumerate() {
            if w > 1e-12 {
                let mem = memory_matrix.get_col(i);
                for (s, &m) in next_state.iter_mut().zip(mem.iter()) {
                    *s += w * m;
                }
            }
        }
        normalize_in_place(&mut next_state);

        // 4. Compute Lyapunov energy of new state
        let energy = compute_lyapunov_energy(memory_matrix, &next_state, beta);
        energy_trajectory.push(energy);

        // 5. Check step convergence ||x^(t+1) - x^(t)||_2 < tol
        let mut diff_sq = 0.0f32;
        for (&a, &b) in current_state.iter().zip(next_state.iter()) {
            let d = a - b;
            diff_sq += d * d;
        }

        current_state = next_state;

        if diff_sq.sqrt() < tol {
            break;
        }
    }

    let final_energy = *energy_trajectory.last().unwrap_or(&initial_energy);

    HopfieldResult {
        retrieved_vector: current_state,
        final_energy,
        energy_trajectory,
        weights,
        candidate_ids: item_ids.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_to_beta_interpolation() {
        let b0 = resolution_to_beta(0.0, 0.5, 30.0);
        let b1 = resolution_to_beta(1.0, 0.5, 30.0);
        let b_mid = resolution_to_beta(0.5, 0.5, 30.0);

        assert!((b0 - 0.5).abs() < 1e-5);
        assert!((b1 - 30.0).abs() < 1e-5);
        assert!(b_mid > b0 && b_mid < b1);
        // Geometric mean of 0.5 and 30 is sqrt(15) ~ 3.87298
        assert!((b_mid - 15.0f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_monotonic_energy_decay() {
        let mut mat = ColumnMatrix::new(4, 3);
        mat.push_col(&[1.0, 0.0, 0.0, 0.0]);
        mat.push_col(&[0.8, 0.6, 0.0, 0.0]);
        mat.push_col(&[0.0, 0.0, 1.0, 0.0]);

        let ids = vec!["m1".to_string(), "m2".to_string(), "m3".to_string()];
        let query = vec![0.9, 0.1, 0.0, 0.0];

        let result = continuous_hopfield_retrieve(&mat, &ids, &query, 5.0, 10, 1e-6);

        // Verify strictly non-increasing Lyapunov energy
        for i in 1..result.energy_trajectory.len() {
            let prev = result.energy_trajectory[i - 1];
            let curr = result.energy_trajectory[i];
            assert!(
                curr <= prev + 1e-6,
                "Energy increased from step {} ({}) to step {} ({})",
                i - 1,
                prev,
                i,
                curr
            );
        }
    }
}
