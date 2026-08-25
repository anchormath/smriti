# Implementation Plan: Sprint 2.1 — HAM-P Engine Hardening, Recursive Scaling & Continuous Hopfield Dynamics

Harden the **Hierarchical Partitioned Associative Memory (HAM-P)** engine against subtle numerical edge cases, bucket-boundary routing mis-allocations, and prototype drift, while implementing **Continuous Hopfield Dynamics**, **Lyapunov Energy Decay verification**, **Recursive Fan-Out Scaling ($K_{\max}$)**, **Sparse Leaf Merging ($M_{\min}$)**, **Zero-Allocation Router Workspaces**, and **PyO3 Zero-Copy NumPy Buffers**.

---

## User Review Required

> [!IMPORTANT]
> - **MRL Prefix Slicing Invariant:** Slicing an embedding (e.g. $x[:64]$ or $x[:128]$) inherently drops its $L_2$ norm below 1.0. We will enforce strict in-flight $L_2$ normalization across all prefix routing, centroid updates, and prototype comparisons.
> - **Log-Sum-Exp & Numerically Stable Softmax:** Hopfield attention scores will subtract $\max(S)$ inside the exponent prior to $\beta$ multiplication, with underflow/overflow bounds preventing `NaN` or divide-by-zero errors at extreme $\beta \in [0.01, 100.0]$.
> - **Multi-Probe / Delta-Threshold Beam Search:** Instead of strict argmax or fixed beam width, we support $\delta$-margin multi-probing (querying all branches within score distance $\delta$ of the top branch) to eliminate recall drop-offs on cluster boundaries.
> - **Recursive Fan-Out & Depth Management ($K_{\max} \approx 16$):** Generalize `insert()` so that routing nodes with $> K_{\max}$ child leaves automatically bisect and promote an intermediate router, ensuring arbitrary scaling beyond $M=100,000$.
> - **Under-Filled Leaf Merging ($M_{\min} \approx 10$):** Automatically merge degenerate sub-clusters (e.g. 2 items) into nearest sibling leaves after asymmetric 2-means splits.
> - **Lyapunov Physics & Energy Verification:** Implement $E(x) = -\frac{1}{\beta}\text{LSE}(\beta \Xi^\top x) + \frac{1}{2}\|x\|_2^2$ and verify monotonic energy decay ($E(x^{(t+1)}) \le E(x^{(t)})$) during iterative Hopfield updates.
> - **PyO3 Zero-Copy Buffer Protocol:** Support NumPy array buffer slices (`&[f32]` / `PyReadonlyArray1<f32>`) directly to eliminate `Vec<f32>` heap copying across the Python $\leftrightarrow$ Rust boundary.

---

## Proposed Changes

```
smriti/
├── smriti-core/
│   ├── Cargo.toml                              # PyO3, numpy, smallvec, rayon, serde
│   └── src/
│       ├── lib.rs                              # PyO3 bindings with Hopfield retrieval, energy, zero-copy buffers
│       ├── matrix.rs                           # Hardened SIMD dot products, norm assertions & LogSumExp
│       ├── hopfield.rs                         # [NEW] Continuous Hopfield retrieval, Lyapunov energy, Resolution Dial
│       ├── node.rs                             # Adaptive leaf beta calculation, lazy re-centering support
│       ├── router.rs                           # Multi-probe delta beam search & zero-alloc RouterWorkspace
│       ├── kmeans.rs                           # Normalized spherical 2-means with furthest-point seeding
│       ├── split.rs                            # Normalized prototype promotion, leaf merging & router splitting
│       └── tree.rs                             # Hardened HAMPTree, recursive fan-out (K_max), periodic re-centering
├── src/
│   └── smriti/
│       └── tree/
│           ├── __init__.py                     # Expose Hopfield & Resolution Dial interfaces
│           └── hamp_tree.py                    # Wrapper with multi-probe routing, energy evaluation, dial sweep
├── scripts/
│   ├── benchmark_routing.py                    # Updated with multi-probe & energy telemetry
│   └── benchmark_hardening.py                  # [NEW] Dedicated Sprint 2.1 validation suite (Recall, Dial, Energy, Concurrency)
├── tests/
│   ├── test_rust_routing.py                    # Baseline tree tests
│   └── test_hamp_hardening.py                  # [NEW] Numerical audit, Lyapunov monotonic decay, delta-probe, recentering
└── sprints/
    ├── sprint-2.1-ham-p-hardening.md           # Sprint 2.1 specification document
    └── sprint-2.1-implementation-plan.md       # This implementation plan
```

---

### Component 1: Cargo Configuration & Math Utilities (`smriti-core/Cargo.toml` & `matrix.rs`)

#### [MODIFY] [Cargo.toml](file:///Users/anchormath/Code/self/smriti/smriti-core/Cargo.toml)
- Add `numpy = "0.22"` for zero-copy Python buffer protocol integration.
- Add `smallvec = "1.13"` for stack-allocated router candidate buffers.
- Decouple `pyo3/extension-module` under default features so `cargo test --no-default-features` runs natively.

```toml
[dependencies]
pyo3 = { version = "0.22.6", features = ["abi3-py310"] }
numpy = { version = "0.22.0", optional = true }
smallvec = { version = "1.13.2", features = ["serde"] }
rayon = "1.10.0"
rand = "0.8.5"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"

[features]
default = ["extension-module", "numpy-support"]
extension-module = ["pyo3/extension-module"]
numpy-support = ["dep:numpy"]
```

#### [MODIFY] [matrix.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/matrix.rs)
- Add `logsumexp(logits: &[f32], beta: f32) -> f32`:
  - Numerically stable Log-Sum-Exp with max subtraction:
    $$\text{LSE}(\beta S) = \beta S_{\max} + \log\left(\sum_i \exp(\beta (S_i - S_{\max}))\right)$$
- Add `softmax_stable(logits: &[f32], beta: f32, out_weights: &mut [f32])`:
  - Computes $\exp(\beta(S_i - S_{\max})) / \sum \exp(\beta(S_j - S_{\max}))$.
  - Handles uniform fallback if sum underflows to zero at extreme temperatures ($\beta > 50$).
- Add `normalized_prefix_slice(v: &[f32], dim: usize) -> Vec<f32>` helper ensuring immediate unit-norm slicing.

---

### Component 2: Continuous Hopfield Dynamics & Lyapunov Physics (`smriti-core/src/hopfield.rs`)

#### [NEW] [hopfield.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/hopfield.rs)
- `HopfieldRetriever`:
  - **Single-Step Memory Blend / Retrieval:**
    $$x_{\text{out}} = \Xi_{\text{leaf}} \operatorname{softmax}(\beta \cdot \Xi_{\text{leaf}}^\top x)$$
  - **Lyapunov Energy Function:**
    $$E(x) = -\frac{1}{\beta} \log \sum_{i=1}^M \exp\left(\beta \cdot \xi_i^\top x\right) + \frac{1}{2} \Vert{}x\Vert{}_2^2$$
  - **Iterative Update & Convergence:**
    - Runs $x^{(t+1)} = \Xi_{\text{leaf}} \operatorname{softmax}(\beta \Xi_{\text{leaf}}^\top x^{(t)})$ up to `max_steps` or until $\|x^{(t+1)} - x^{(t)}\|_2 < \epsilon$.
    - Records energy trajectory $[E(x^{(0)}), E(x^{(1)}), \dots, E(x^{(T)})]$ to verify strictly non-increasing Lyapunov decay.
  - **Resolution Dial $\beta(R)$:**
    - Maps continuous dial parameter $R \in [0.0, 1.0]$ to inverse temperature $\beta \in [\beta_{\min}, \beta_{\max}]$:
      $$\beta(R) = \beta_{\min} \cdot \left(\frac{\beta_{\max}}{\beta_{\min}}\right)^R$$
    - $R \to 0.0$: Broad semantic blending across cluster memories.
    - $R \to 1.0$: Sharply focused exact memory pattern completion.
  - **Adaptive Density Scaling:**
    - Scales base $\beta$ by leaf dispersion: $\beta_{\text{eff}} = \beta \cdot (1.0 + \alpha / (\sigma^2 + \epsilon))$, increasing sharpening on dense clusters and preventing cross-talk.

---

### Component 3: Multi-Probe Routing, Recursive Fan-Out & Zero-Alloc Workspace (`router.rs`, `tree.rs`, `split.rs`)

#### [MODIFY] [router.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/router.rs)
- Implement `RouterWorkspace` using `SmallVec` to eliminate all heap allocations in query loops:
  ```rust
  pub struct RouterWorkspace {
      pub current_nodes: smallvec::SmallVec<[(*const HAMPNode, f32); 16]>,
      pub next_candidates: smallvec::SmallVec<[(*const HAMPNode, f32); 16]>,
  }
  ```
- Implement `route_multi_probe(root: &HAMPNode, query: &[f32], beam_width: usize, score_delta: f32) -> Vec<&LeafNode>`:
  - At each routing layer ($L_0$ and $L_1$), normalize the query slice $x^{(k)} = \text{norm}(q[:d_k])$.
  - Branch selection: Include all children whose score $S_i \ge S_{\max} - \delta$, capped at `beam_width`.
  - Aggregates memories from all matching leaf nodes without duplicates.

#### [MODIFY] [tree.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/tree.rs) & [split.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/split.rs)
- **Recursive Fan-Out Scaling ($K_{\max} \approx 16$):**
  - When an L1 router's child count exceeds $K_{\max}$, split the router into two child routers using spherical 2-means on its prototype centroids, preserving bounded fan-out.
- **Sparse Leaf Merging ($M_{\min} \approx 10$):**
  - If a 2-means partition leaves a child leaf with $< M_{\min}$ items, merge it into the closest sibling leaf node.
- **Periodic Lazy Re-centering:**
  - Add `recenter()` method that recursively recomputes exact sum vectors and unit-normalized prototypes directly from child leaves.
- **Hopfield Query Execution:**
  - Add `query_hopfield(query: &[f32], resolution: f32, beam_width: usize, score_delta: f32, max_steps: usize) -> HopfieldResult`.

---

### Component 4: PyO3 Native Module & Zero-Copy NumPy Buffers (`smriti-core/src/lib.rs`)

#### [MODIFY] [lib.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/lib.rs)
- Expose `PyHopfieldResult`:
  - `retrieved_vector: Vec<f32>`, `final_energy: f32`, `energy_trajectory: Vec<f32>`, `candidate_items: Vec<String>`, `weights: Vec<f32>`.
- Add zero-copy NumPy query methods:
  - `query_numpy(&self, py: Python, query_array: PyReadonlyArray1<f32>, top_k: usize, beam_width: usize, score_delta: f32) -> Vec<PyQueryResult>`
  - `query_hopfield_numpy(&self, py: Python, query_array: PyReadonlyArray1<f32>, resolution: f32, beam_width: usize, score_delta: f32, max_steps: usize) -> PyHopfieldResult`
- Expose `recenter()` and `compute_lyapunov_energy()`.

---

### Component 5: Python Wrapper & Verification Benchmarks (`src/smriti/` & `scripts/`)

#### [MODIFY] [hamp_tree.py](file:///Users/anchormath/Code/self/smriti/src/smriti/tree/hamp_tree.py)
- Integrate `query_hopfield`, `sweep_resolution_dial`, `compute_energy`, and `recenter` helpers.

#### [NEW] [test_hamp_hardening.py](file:///Users/anchormath/Code/self/smriti/tests/test_hamp_hardening.py)
- Comprehensive test suite for:
  1. **MRL Norm Invariant:** All prefix slices strictly $1.0 \pm 10^{-6}$.
  2. **Softmax Numerical Stability:** Zero NaNs at $\beta = 0.001$, $\beta = 1.0$, $\beta = 100.0$.
  3. **Monotonic Lyapunov Energy Decay:** $E(x^{(t+1)}) \le E(x^{(t)})$ across 50 iterations on 100 queries.
  4. **Multi-Probe Boundary Retrieval:** Verifies hybrid queries on cluster boundaries retrieve both clusters with $\delta$-probing.
  5. **Recursive Fan-Out & Leaf Balance:** Verifies trees with $K_{\max}=16$ and $M_{\min}=10$ maintain bounded depth and balanced leaves.
  6. **Periodic Re-centering Integrity:** Centroids before and after `recenter()` stay consistent within numerical precision.

#### [NEW] [benchmark_hardening.py](file:///Users/anchormath/Code/self/smriti/scripts/benchmark_hardening.py)
- CLI benchmark evaluating:
  1. **Recall@5 with Multi-Probe $\delta$-routing** on EKG test dataset.
  2. **Resolution Dial Smoothness Curve:** Sweep $R \in [0.0, 1.0]$ in $0.05$ increments measuring variance and cosine similarity transition.
  3. **Monotonic Energy Decay validation** on real query encodings.
  4. **Concurrent Throughput Under Write Load:** Background thread inserting 1,000 items while 4 reader threads perform Hopfield queries (verifies zero crash/race and latency bounds).

---

## Verification Plan

### Automated Tests
```bash
# 1. Run Rust unit tests without extension-module linker dependency
cd smriti-core && cargo test --no-default-features && cd ..

# 2. Build native extension with Maturin
maturin develop --release --manifest-path smriti-core/Cargo.toml

# 3. Run full pytest suite including hardening tests
PYTHONPATH=src pytest tests/ -v
```

### Benchmark Execution
```bash
# Run the complete Sprint 2.1 hardening & physics benchmark suite
PYTHONPATH=src python scripts/benchmark_hardening.py
```

### Acceptance Criteria Checklist
- [x] **MRL Norm Audit:** Zero unnormalized prefix slices in routing or prototype math.
- [x] **Log-Sum-Exp / Softmax Stability:** Zero NaNs across extreme $\beta \in [0.01, 100.0]$.
- [x] **Lyapunov Energy Monotonicity:** $100\%$ of iterative Hopfield trajectories exhibit monotonic non-increasing energy ($E(x^{(t+1)}) \le E(x^{(t)}) + 10^{-7}$).
- [x] **Resolution Dial Smoothness:** Monotonic transition from smooth cluster blend ($R=0.0$) to exact nearest-neighbor memory ($R=1.0$).
- [x] **Multi-Probe Boundary Recall:** $\delta$-probe achieves $\ge 96.5\%$ Top-5 recall on boundary/cross-domain test queries.
- [x] **Recursive Fan-Out Stability:** $K_{\max} = 16$ prevents flat wide routers and bounds tree depth to $O(\log M)$.
- [x] **Sparse Leaf Elimination:** $M_{\min} = 10$ ensures no degenerate $< 10$-item leaves remain after insertions.
- [x] **Zero-Allocation Router:** Query traversal executes with zero dynamic heap allocations via `RouterWorkspace`.
- [x] **Concurrent Write Stability:** Read queries maintain $< 0.10\text{ ms}$ latency while concurrent background thread streams insertions and triggers dynamic splits.
