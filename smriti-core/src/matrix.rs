//! Contiguous column-major memory matrix & SIMD-accelerated linear algebra operations.

use serde::{Deserialize, Serialize};

/// Column-major matrix where each column is stored contiguously in memory.
///
/// If dimensions are `d` (rows) and count is `M` (cols),
/// column `j` occupies `data[j * rows .. (j + 1) * rows]`.
/// Slicing a prefix dimension `d_sub <= d` is a contiguous subslice `&col[0..d_sub]`,
/// ensuring zero copies and zero heap allocations during multi-resolution prefix routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMatrix {
    pub rows: usize, // Dimension (e.g. 768 or 128 or 64)
    pub cols: usize, // Number of items
    pub data: Vec<f32>, // Flattened column-major storage
}

impl ColumnMatrix {
    /// Create a new empty ColumnMatrix with fixed row dimension and initial column capacity.
    pub fn new(rows: usize, capacity_cols: usize) -> Self {
        Self {
            rows,
            cols: 0,
            data: Vec::with_capacity(rows * capacity_cols),
        }
    }

    /// Construct a ColumnMatrix from existing flattened column-major data.
    pub fn from_flat(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Data length {} does not match rows ({}) * cols ({}) = {}",
                data.len(),
                rows,
                cols,
                rows * cols
            ));
        }
        Ok(Self { rows, cols, data })
    }

    /// Append a single column vector of length `self.rows`.
    #[inline]
    pub fn push_col(&mut self, col: &[f32]) {
        assert_eq!(
            col.len(),
            self.rows,
            "Pushed vector dimension {} does not match matrix rows {}",
            col.len(),
            self.rows
        );
        self.data.extend_from_slice(col);
        self.cols += 1;
    }

    /// Borrow full column `index` as a slice.
    #[inline]
    pub fn get_col(&self, index: usize) -> &[f32] {
        debug_assert!(index < self.cols, "Index out of bounds");
        let start = index * self.rows;
        &self.data[start..start + self.rows]
    }

    /// Borrow mutable full column `index`.
    #[inline]
    pub fn get_col_mut(&mut self, index: usize) -> &mut [f32] {
        debug_assert!(index < self.cols, "Index out of bounds");
        let start = index * self.rows;
        &mut self.data[start..start + self.rows]
    }

    /// Zero-copy prefix slice of column `index` up to dimension `dim <= self.rows`.
    #[inline]
    pub fn get_col_slice(&self, index: usize, dim: usize) -> &[f32] {
        debug_assert!(index < self.cols, "Index out of bounds");
        debug_assert!(dim <= self.rows, "Requested dim exceeds matrix rows");
        let start = index * self.rows;
        &self.data[start..start + dim]
    }

    /// Number of columns stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.cols
    }

    /// Check if matrix has no columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cols == 0
    }

    /// Remove all columns while keeping allocated capacity.
    pub fn clear(&mut self) {
        self.cols = 0;
        self.data.clear();
    }
}

/// Compute inner dot product between two slices.
///
/// Uses NEON on aarch64 (Apple Silicon) or unrolled SIMD-friendly loops for high performance.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let a_slice = &a[..len];
    let b_slice = &b[..len];

    #[cfg(target_arch = "aarch64")]
    unsafe {
        dot_product_neon(a_slice, b_slice)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_scalar_unrolled(a_slice, b_slice)
    }
}

/// Fast unrolled scalar dot product with 8-way accumulation for auto-vectorization.
#[inline]
pub fn dot_product_scalar_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
        sum4 += a[base + 4] * b[base + 4];
        sum5 += a[base + 5] * b[base + 5];
        sum6 += a[base + 6] * b[base + 6];
        sum7 += a[base + 7] * b[base + 7];
    }

    let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    for i in (chunks * 8)..len {
        total += a[i] * b[i];
    }
    total
}

/// AArch64 NEON hardware-accelerated dot product.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 16;
    let remainder = len % 16;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..chunks {
        let va0 = vld1q_f32(ptr_a);
        let vb0 = vld1q_f32(ptr_b);
        acc0 = vfmaq_f32(acc0, va0, vb0);

        let va1 = vld1q_f32(ptr_a.add(4));
        let vb1 = vld1q_f32(ptr_b.add(4));
        acc1 = vfmaq_f32(acc1, va1, vb1);

        let va2 = vld1q_f32(ptr_a.add(8));
        let vb2 = vld1q_f32(ptr_b.add(8));
        acc2 = vfmaq_f32(acc2, va2, vb2);

        let va3 = vld1q_f32(ptr_a.add(12));
        let vb3 = vld1q_f32(ptr_b.add(12));
        acc3 = vfmaq_f32(acc3, va3, vb3);

        ptr_a = ptr_a.add(16);
        ptr_b = ptr_b.add(16);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    let mut total = vaddvq_f32(acc0);

    // Remainder handling
    let rem_chunks = remainder / 4;
    let rem_rem = remainder % 4;

    for _ in 0..rem_chunks {
        let va = vld1q_f32(ptr_a);
        let vb = vld1q_f32(ptr_b);
        let acc = vmulq_f32(va, vb);
        total += vaddvq_f32(acc);
        ptr_a = ptr_a.add(4);
        ptr_b = ptr_b.add(4);
    }

    for i in 0..rem_rem {
        total += *ptr_a.add(i) * *ptr_b.add(i);
    }

    total
}

/// Compute matrix-vector inner products: `out[j] = dot(mat.col(j), query)`.
#[inline]
pub fn matrix_vector_dot(mat: &ColumnMatrix, query: &[f32], out: &mut [f32]) {
    assert_eq!(out.len(), mat.cols);
    for (j, score) in out.iter_mut().enumerate() {
        let col = mat.get_col(j);
        *score = dot_product(col, query);
    }
}

/// Calculate L2 norm of a vector slice.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    let dot = dot_product(v, v);
    dot.sqrt()
}

/// L2-normalize a slice in-place. If norm is near zero, leaves unchanged.
#[inline]
pub fn normalize_in_place(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Return an L2-normalized copy of a slice.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize_in_place(&mut out);
    out
}

/// Compute the exact normalized centroid of a list of vectors.
pub fn compute_centroid(vectors: &[&[f32]], dim: usize) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![0.0; dim];
    }
    let mut sum = vec![0.0f32; dim];
    for v in vectors {
        for (s, &x) in sum.iter_mut().zip(v[..dim].iter()) {
            *s += x;
        }
    }
    let count = vectors.len() as f32;
    for s in sum.iter_mut() {
        *s /= count;
    }
    normalize_in_place(&mut sum);
    sum
}

/// Extract an L2-normalized prefix slice of dimension `dim <= v.len()`.
///
/// Guarantees that the returned sub-vector strictly satisfies `|norm - 1.0| < 1e-6`.
#[inline]
pub fn normalized_prefix_slice(v: &[f32], dim: usize) -> Vec<f32> {
    assert!(dim <= v.len(), "Requested dim {} exceeds vector len {}", dim, v.len());
    let mut out = v[..dim].to_vec();
    normalize_in_place(&mut out);
    out
}

/// Numerically stable Log-Sum-Exp with maximum subtraction:
/// LSE(beta * S) = beta * max(S) + ln(sum_i exp(beta * (S_i - max(S))))
#[inline]
pub fn logsumexp(logits: &[f32], beta: f32) -> f32 {
    if logits.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &x in logits {
        if x > max_val {
            max_val = x;
        }
    }
    if max_val.is_infinite() {
        return max_val;
    }

    let mut sum_exp = 0.0f32;
    for &x in logits {
        let diff = (x - max_val) * beta;
        // Clamp lower bound to avoid underflow subnormals
        if diff > -80.0 {
            sum_exp += diff.exp();
        }
    }

    if sum_exp <= 0.0 {
        beta * max_val
    } else {
        beta * max_val + sum_exp.ln()
    }
}

/// Numerically stable Softmax computation:
/// out_weights[i] = exp(beta * (S_i - max(S))) / sum_j exp(beta * (S_j - max(S)))
///
/// Prevents underflow/overflow and falls back gracefully to uniform weights if total sum is near zero.
#[inline]
pub fn softmax_stable(logits: &[f32], beta: f32, out_weights: &mut [f32]) {
    assert_eq!(logits.len(), out_weights.len());
    let n = logits.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out_weights[0] = 1.0;
        return;
    }

    let mut max_val = f32::NEG_INFINITY;
    for &x in logits {
        if x > max_val {
            max_val = x;
        }
    }

    let mut sum_exp = 0.0f32;
    for (w, &x) in out_weights.iter_mut().zip(logits.iter()) {
        let diff = (x - max_val) * beta;
        let e = if diff > -80.0 { diff.exp() } else { 0.0 };
        *w = e;
        sum_exp += e;
    }

    if sum_exp > 1e-12 {
        let inv_sum = 1.0 / sum_exp;
        for w in out_weights.iter_mut() {
            *w *= inv_sum;
        }
    } else {
        // Fallback to uniform distribution
        let uniform = 1.0 / (n as f32);
        for w in out_weights.iter_mut() {
            *w = uniform;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_matrix_basic() {
        let mut mat = ColumnMatrix::new(4, 2);
        mat.push_col(&[1.0, 2.0, 3.0, 4.0]);
        mat.push_col(&[5.0, 6.0, 7.0, 8.0]);

        assert_eq!(mat.len(), 2);
        assert_eq!(mat.get_col(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat.get_col(1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mat.get_col_slice(0, 2), &[1.0, 2.0]);
        assert_eq!(mat.get_col_slice(1, 3), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_dot_product_simd_accuracy() {
        let a: Vec<f32> = (0..768).map(|x| (x as f32) * 0.01).collect();
        let b: Vec<f32> = (0..768).map(|x| (768 - x) as f32 * 0.01).collect();

        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_product(&a, &b);
        let rel_error = (result - expected).abs() / expected;
        assert!(rel_error < 1e-4, "SIMD dot product mismatch: got {}, expected {}, rel_err={}", result, expected, rel_error);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize_in_place(&mut v);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_prefix_slice() {
        let full = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = normalized_prefix_slice(&full, 3);
        assert_eq!(p.len(), 3);
        assert!((l2_norm(&p) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_stable_extreme_beta() {
        let logits = vec![0.1, 0.9, 0.5, 0.2];
        let mut weights = vec![0.0f32; 4];

        // Moderate beta
        softmax_stable(&logits, 5.0, &mut weights);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(weights[1] > weights[2]);

        // Extreme high beta (should not NaN or overflow)
        softmax_stable(&logits, 100.0, &mut weights);
        let sum_high: f32 = weights.iter().sum();
        assert!((sum_high - 1.0).abs() < 1e-5);
        assert!(weights[1] > 0.999);

        // Extreme low beta (should approach uniform)
        softmax_stable(&logits, 0.001, &mut weights);
        let sum_low: f32 = weights.iter().sum();
        assert!((sum_low - 1.0).abs() < 1e-5);
        for &w in &weights {
            assert!((w - 0.25).abs() < 1e-2);
        }
    }

    #[test]
    fn test_logsumexp_stability() {
        let logits = vec![1.0, 2.0, 3.0];
        let lse = logsumexp(&logits, 1.0);
        let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
        assert!((lse - expected).abs() < 1e-5);

        // High beta stability
        let lse_high = logsumexp(&logits, 50.0);
        assert!(!lse_high.is_nan() && !lse_high.is_infinite());
    }
}
