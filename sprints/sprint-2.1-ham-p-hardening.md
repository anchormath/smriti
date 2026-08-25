# Sprint 2.1: HAM-P Engine Hardening, Recursive Scaling & Continuous Hopfield Dynamics

**Timeline:** Mid-Sprint / Weeks 4–5  
**Epic:** Production Hardening & Mathematical Integrity of the `smriti-core` Rust Engine  
**Goal:** Harden the HAM-P tree against numerical edge cases, bucket-boundary routing mis-allocations, and prototype drift, while implementing arbitrary recursive fan-out ($K_{\max}$), under-filled leaf merging ($M_{\min}$), zero-allocation router workspaces, and verifying continuous Hopfield Lyapunov energy decay ($E(x)$).

---

## 1. Objectives & Scope

- [x] **MRL $L_2$ Prefix Normalization Audit:** Enforce strict in-flight $L_2$ normalization on all prefix slices ($x[:d_0]$, $x[:d_1]$) across routing, prototype calculation, and centroid updates to guarantee true cosine attention.
- [x] **Log-Sum-Exp & Numerically Stable Softmax:** Implement max-subtracted Log-Sum-Exp and softmax underflow protection across extreme inverse temperatures $\beta \in [0.01, 100.0]$.
- [x] **Multi-Probe $\delta$-Threshold Routing:** Implement soft beam search ($S_i \ge S_{\max} - \delta$) to eliminate recall drop-offs on cluster decision boundaries.
- [x] **Recursive Fan-Out Scaling ($K_{\max} \approx 16$):** Generalize `insert()` from a fixed 3-layer assumption to recursive multi-way tree branching, promoting overflowing routing nodes dynamically.
- [x] **Under-filled Leaf Merging & Pruning ($M_{\text{leaf}} < M_{\min} \approx 10$):** Implement sibling leaf merging to prevent degenerate sparse leaves (e.g. 2 items) from asymmetric 2-means splits.
- [x] **Zero-Allocation Router Workspace:** Replace dynamic heap vector allocations in beam search with stack-allocated small vectors or a reusable `RouterWorkspace`.
- [x] **Periodic Lazy Re-centering:** Add a bottom-up `recenter()` pass to eliminate centroid drift from asymmetric insertion sequences.
- [x] **Lyapunov Energy Monotonicity Test Suite:** Implement the continuous Lyapunov energy function $E(x)$ and verify strictly non-increasing decay ($E(x^{(t+1)}) \le E(x^{(t)})$) during iterative Hopfield updates.
- [x] **PyO3 Zero-Copy NumPy Buffer Interface:** Support zero-copy array views (`&[f32]` / `PyReadonlyArray1<f32>`) across the Python $\leftrightarrow$ Rust boundary.

---

## 2. Mathematical & Systems Hardening Specifications

### 2.1 The MRL $L_2$ Re-Normalization Invariant

When slicing an embedding $x \in \mathbb{R}^{d_{\text{full}}}$:
$$\Vert{}x[:d_k]\Vert{}_2 \le 1.0$$
If prefix slices are not immediately re-normalized, inner products $\langle \xi_i, x \rangle$ are no longer cosine similarities, and vectors with naturally larger prefix norms dominate routing weights regardless of semantic relevance.

**Invariant Rule:**
```rust
#[inline]
pub fn normalized_prefix_slice(v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = v[..dim].to_vec();
    normalize_in_place(&mut out);
    out
}
```

---

### 2.2 Log-Sum-Exp & Softmax Stability at Extreme $\beta$

For inverse temperatures $\beta \in [0.01, 100.0]$:
$$S_{\max} = \max_{j} S_j$$
$$\text{LSE}(\beta S) = \beta S_{\max} + \log \sum_{i=1}^M \exp(\beta (S_i - S_{\max}))$$
$$a_i = \frac{\exp(\beta (S_i - S_{\max}))}{\sum_j \exp(\beta (S_j - S_{\max}))}$$

If $\sum \exp(\dots)$ underflows to `0.0` due to floating-point precision at $\beta > 50$, the engine falls back to a one-hot vector on $\operatorname{argmax}(S)$, preventing `NaN` or zero-division errors.

---

### 2.3 Multi-Probe $\delta$-Threshold Beam Search (Eliminating Boundary Drops)

Instead of hard routing to a single branch ($\operatorname{argmax}$), multi-probe beam search explores all branches within a margin $\delta$ of the top candidate:

$$\text{Branch Selection:} \quad \{ i \mid S_i \ge S_{\max} - \delta \} \quad \text{capped at } \text{beam\_width}$$

```
                Incoming Query x (Lies near boundary between Domain A & B)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             Domain A (Score = 0.88)             Domain B (Score = 0.86)
             [Selected: Diff <= delta=0.05]      [Selected: Diff <= delta=0.05]
                        │                                   │
                  ┌─────┴─────┐                       ┌─────┴─────┐
                  ▼           ▼                       ▼           ▼
               Leaf A1     Leaf A2                 Leaf B1     Leaf B2
```

Candidate leaf memories are merged into a unified contiguous pool before Hopfield retrieval, boosting boundary recall to $\ge 96.5\%$.

---

### 2.4 Recursive Fan-Out & Depth Management ($K_{\max} \approx 16$)

To scale beyond $M = 100,000$ without wide flat routing nodes:
* When a leaf node under a routing node splits, if the parent router already has $K_{\text{children}} \ge K_{\max}$ (e.g. 16 children), the router itself splits into two child routers using spherical 2-means on its prototype centroids.
* This maintains bounded fan-out $O(K_{\max})$ and bounded tree depth $O(\log_{K_{\max}} M)$.

---

### 2.5 Under-filled Leaf Merging & Pruning ($M_{\min} \approx 10$)

* **Trigger:** If a dynamic split results in an asymmetric partition where $M_{\text{leaf}} < M_{\min}$ (e.g. 2 items), the sparse leaf is merged into its closest sibling leaf node.
* **Maintenance:** Parent routing prototypes and counts are updated atomically.

---

### 2.6 Zero-Allocation Router Workspace

To achieve zero heap allocation per query, routing buffers use preallocated or stack-allocated arrays:

```rust
pub struct RouterWorkspace {
    pub candidate_nodes: smallvec::SmallVec<[(*const HAMPNode, f32); 16]>,
    pub next_candidates: smallvec::SmallVec<[(*const HAMPNode, f32); 16]>,
}
```

---

### 2.7 Lyapunov Energy Decay & Physics Verification

The Lyapunov energy function governing continuous Hopfield associative recall is:

$$E(x) = -\frac{1}{\beta} \log \sum_{i=1}^M \exp\left(\beta \cdot \xi_i^\top x\right) + \frac{1}{2} \Vert{}x\Vert{}_2^2$$

During iterative Hopfield updates:
$$x^{(t+1)} = \Xi_{\text{leaf}} \operatorname{softmax}\left(\beta \cdot \Xi_{\text{leaf}}^\top x^{(t)}\right)$$

**Physical Verification Requirement:**
$$E(x^{(t+1)}) \le E(x^{(t)}) + 10^{-7} \quad \forall t$$
Energy must strictly decrease or stay constant until converging to an attractor fixed point.

---

## 3. Sprint Tasks & Deliverables

| Task ID | Task Description | Est. Days | Module / Tooling | Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **S2.1-T1** | Implement `logsumexp`, `softmax_stable`, and `normalized_prefix_slice` | 1 | Rust (`matrix.rs`) | `smriti-core/src/matrix.rs` |
| **S2.1-T2** | Implement `HopfieldEngine` with Lyapunov energy and Resolution Dial $\beta(R)$ | 2 | Rust (`hopfield.rs`) | `smriti-core/src/hopfield.rs` |
| **S2.1-T3** | Implement Multi-Probe $\delta$-beam search and zero-alloc `RouterWorkspace` | 2 | Rust (`router.rs`) | `smriti-core/src/router.rs` |
| **S2.1-T4** | Implement recursive fan-out promotion ($K_{\max}$) and leaf pruning ($M_{\min}$) | 2 | Rust (`tree.rs`, `split.rs`) | `smriti-core/src/tree.rs` |
| **S2.1-T5** | Implement periodic lazy re-centering (`recenter()`) | 1 | Rust (`tree.rs`) | `smriti-core/src/tree.rs` |
| **S2.1-T6** | Expose Hopfield retrieval, energy, and zero-copy NumPy buffers via PyO3 | 2 | PyO3 / Rust (`lib.rs`) | `smriti-core/src/lib.rs` |
| **S2.1-T7** | Build hardening test suite (Lyapunov decay, $\delta$-probe recall, drift audit) | 2 | PyTest / Python | `tests/test_hamp_hardening.py` |
| **S2.1-T8** | Build comprehensive hardening benchmark script (Energy, Dial, Concurrency) | 2 | Python / Rich | `scripts/benchmark_hardening.py` |

---

## 4. Acceptance Criteria Checklist

1. **MRL Norm Invariant:** $100\%$ of vector slices evaluated in routing or stored as prototypes are strictly $L_2$-normalized ($|\|v\|_2 - 1.0| < 10^{-6}$).
2. **Softmax Numerical Stability:** Zero `NaN` or `Inf` values across extreme temperatures $\beta \in [0.001, 100.0]$.
3. **Lyapunov Energy Monotonicity:** $100\%$ of iterative Hopfield trajectories exhibit monotonic non-increasing energy decay across 50 iterations on 100 test queries.
4. **Boundary Recall Boost:** Multi-probe $\delta$-routing ($\delta = 0.05$) achieves $\ge 96.5\%$ Top-5 recall on boundary/cross-domain queries.
5. **Zero-Drift Re-centering:** Running-mean centroids deviate by $< 10^{-6}$ from exact batch re-computed centroids after `recenter()`.
6. **Concurrent Write Stability:** Read queries maintain latency $< 0.10\text{ ms}$ while a background thread continuously inserts items and triggers dynamic splits.