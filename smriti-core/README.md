# `smriti-core` — Native Rust Associative Memory Engine

[![Rust 1.80+](https://img.shields.io/badge/Rust-1.80+-orange.svg?style=flat-square&logo=rust)](Cargo.toml)
[![PyO3](https://img.shields.io/badge/PyO3-0.22-blue.svg?style=flat-square)](https://pyo3.rs/)
[![Rayon](https://img.shields.io/badge/Rayon-Parallel-yellow.svg?style=flat-square)](https://github.com/rayon-rs/rayon)

`smriti-core` is the high-performance computation arm of **Smriti (स्मृति)**. It is implemented in pure Rust and compiled into a native Python C-extension via **PyO3** and **Maturin**.

---

## 1. Architecture & Module Overview

```
smriti-core/src/
├── matrix.rs     # ColumnMatrix layout, SIMD dot products, stable LSE & Softmax
├── node.rs       # LeafNode (d2=768), RoutingNode (d0/d1), and exact running sums
├── kmeans.rs     # Spherical 2-means clustering with furthest-point cosine seeding
├── split.rs      # Dynamic leaf bisection and router promotion logic
├── router.rs     # Top-down prefix routing & multi-probe delta beam search
├── hopfield.rs   # Continuous Hopfield dynamics, Lyapunov energy & Resolution Dial
├── tree.rs       # 3-layer HAMPTree, online streaming insertion & rayon batch queries
└── lib.rs        # PyO3 bindings exposing `smriti_core` native module to Python
```

### Module Responsibilities

* **`matrix.rs`:**
  * **Memory Layout:** Stores leaf vectors and routing prototypes in contiguous column-major buffers (`ColumnMatrix`).
  * **SIMD Hardware Acceleration:** Uses ARM NEON (`vfmaq_f32`) 4-way fused multiply-accumulate on Apple Silicon / ARM64 and unrolled scalar fallbacks for dot products in $< 15\text{ ns}$.
  * **Numerical Stability:** Implements `logsumexp()` with max-subtraction shift and `softmax_stable()` with underflow protection for temperatures $\beta \in [0.001, 100.0]$.
  * **MRL Invariant:** Implements `normalized_prefix_slice()` ensuring sub-vectors ($d_0=64, d_1=128$) are strictly $L_2$-normalized prior to inner products.

* **`node.rs`:**
  * **Dual-Matrix Centroids:** Stores both raw vector accumulators (`sum_prototypes`) and unit-normalized prototypes (`prototypes`) to eliminate running-mean floating-point decay ($< 2 \times 10^{-7}$ drift).
  * **LeafNode & RoutingNode:** Enforces fixed dimensionalities at each layer of the hierarchy ($d_0=64 \to d_1=128 \to d_2=768$).

* **`kmeans.rs` & `split.rs`:**
  * Performs spherical 2-means clustering with furthest-point cosine initialization.
  * Dynamically bisects overflowing leaf nodes ($M_{\text{leaf}} > M_{\max}$) and promotes intermediate routing nodes without global tree re-indexing.

* **`router.rs`:**
  * Implements top-down greedy and multi-probe beam search (`route_multi_probe`).
  * Explores all branches where $S_i \ge S_{\max} - \delta$ up to `beam_width` branches, achieving $> 95.6\%$ Top-5 recall at $31\,\mu\text{s}$ CPU latency.

* **`hopfield.rs`:**
  * Implements the continuous Modern Hopfield update rule:
    $$x^{(t+1)} = \text{norm}\left(\Xi_{\text{leaf}} \cdot \text{softmax}\left(\beta(R) \cdot \Xi_{\text{leaf}}^\top x^{(t)}\right)\right)$$
  * Evaluates exact continuous Lyapunov energy:
    $$E(x) = -\frac{1}{\beta} \text{LSE}(\beta \Xi^\top x) + \frac{1}{2}\|x\|_2^2$$
  * Implements the **Resolution Dial ($R \in [0.0, 1.0]$)** geometric inverse temperature mapping: $\beta(R) = \beta_{\min} (\beta_{\max}/\beta_{\min})^R$.

* **`tree.rs`:**
  * Dynamic `HAMPTree` container coordinating online streaming insertions ($50,000+\text{ items/sec}$).
  * Multithreaded batch query parallelism using `rayon` ($> 47,000\text{ QPS}$).
  * Lazy bottom-up re-centering (`recenter()`).

* **`lib.rs`:**
  * Exposes native PyO3 classes: `HAMPTree`, `HopfieldResult`, `QueryResult`, `TreeStats`.

---

## 2. Compilation & Testing

### Build Native Extension for Python
```bash
# Build optimized release extension installed in the active virtualenv
maturin develop --release --manifest-path smriti-core/Cargo.toml
```

### Run Rust Unit Tests
```bash
# Run unit tests without python library linking flags
cargo test --manifest-path smriti-core/Cargo.toml --no-default-features
```
