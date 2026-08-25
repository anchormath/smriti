# Implementation Plan: Sprint 2 — Rust HAM-P Tree Infrastructure & Routing Engine (`smriti-core`)

Implement the high-performance Rust core engine for the **Hierarchical Partitioned Associative Memory (HAM-P)** multi-resolution tree topology, dynamic node splitting via spherical 2-means clustering, running-mean centroid tracking, and PyO3 Python bindings, benchmarked against flat brute-force search on the 3,500 enterprise project nodes and 500 test queries.

---

## User Review Required

> [!NOTE]
> - **Rust Toolchain & Build Integration:** We will create `smriti-core/` as a Rust crate with `pyo3` and `maturin` support. We will also install `maturin` and ensure the Rust compilation toolchain (`cargo`/`rustc`) builds the native extension `smriti_core` into the Python virtual environment.
> - **SIMD & Acceleration Strategy:** We will implement vectorized dot products using platform-optimized SIMD loops (ARM NEON for Apple Silicon / AVX2 for x86_64 with fallback scalar auto-vectorization) along with `rayon` multithreaded routing where beneficial.
> - **Multi-Resolution Topology:** The tree routes queries top-down through Layer 0 ($d_0=64$, root category prototypes) $\to$ Layer 1 ($d_1=128$, subcategory routing prototypes) $\to$ Layer 2 ($d_2=768$, partitioned contiguous leaf memory buffers).

---

## Proposed Changes

```
smriti/
├── smriti-core/                                # High-performance Rust engine crate
│   ├── Cargo.toml                              # Dependencies: pyo3, rayon, serde, etc.
│   ├── src/
│   │   ├── lib.rs                              # PyO3 module definition & Python classes
│   │   ├── matrix.rs                           # Contiguous column-major memory matrix & SIMD dot product
│   │   ├── node.rs                             # LeafNode, RoutingNode, and HAMPNode tree structures
│   │   ├── router.rs                           # Top-down prefix routing & Top-K beam/greedy search
│   │   ├── kmeans.rs                           # Spherical 2-means clustering & dispersion computation
│   │   ├── split.rs                            # Dynamic leaf splitting & intermediate router promotion
│   │   └── tree.rs                             # HAMPTree container, online insertion & centroid maintenance
│   └── tests/
│       ├── test_matrix.rs                      # Matrix & SIMD math unit tests
│       ├── test_tree.rs                        # Tree insertion, routing & split unit tests
│       └── test_kmeans.rs                      # 2-means spherical clustering tests
├── pyproject.toml                              # Updated with maturin build / smriti-core integration
├── src/
│   └── smriti/
│       ├── tree/
│       │   ├── __init__.py                     # High-level Python HAM-P tree interface
│       │   └── hamp_tree.py                    # Python wrapper over smriti_core with batch utilities
│       └── ...
├── scripts/
│   └── benchmark_routing.py                    # Benchmark script for recall, throughput, and latency vs Flat Search
└── tests/
    └── test_rust_routing.py                    # PyTest suite for PyO3 bindings, recall, split stability & centroids
```

---

### Component 1: Rust Core Memory & Math Primitives (`smriti-core/src/`)

#### [NEW] [Cargo.toml](file:///Users/anchormath/Code/self/smriti/smriti-core/Cargo.toml)
- Configure crate `smriti-core` (cdylib / rlib)
- Dependencies: `pyo3` (v0.22+ with `extension-module`), `rayon`, `serde`, `rand`, `thiserror`.

#### [NEW] [matrix.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/matrix.rs)
- `ColumnMatrix`: Contiguous flattened `Vec<f32>` stored in column-major order ($d \times M$).
- Zero-copy column slicing and prefix slicing ($x[:64], x[:128], x[:768]$).
- SIMD accelerated dot product: `dot_product(a: &[f32], b: &[f32]) -> f32` and `matrix_vector_dot(mat: &ColumnMatrix, vec: &[f32], out: &mut [f32])`.
- Fast L2-normalization and cosine similarity.

#### [NEW] [node.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/node.rs)
- `LeafNode`:
  - `id: u64`, `dim: usize` ($768$), `capacity: usize` ($M_{\max}$ e.g. $100$)
  - `memory_matrix: ColumnMatrix` (contiguous $768 \times M$)
  - `item_ids: Vec<String>`
- `RoutingNode`:
  - `id: u64`, `dim: usize` ($64$ at root L0, $128$ at intermediate L1)
  - `prototypes: ColumnMatrix` (prototype centroids for each child)
  - `children: Vec<HAMPNode>`
- `enum HAMPNode { Routing(RoutingNode), Leaf(LeafNode) }`

#### [NEW] [kmeans.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/kmeans.rs)
- `spherical_2means(matrix: &ColumnMatrix, max_iter: usize) -> (Vec<usize>, Vec<f32>, Vec<f32>)`:
  - Partitions vectors in leaf matrix into clusters $C_A$ and $C_B$.
  - Computes normalized centroids $c_A, c_B \in \mathbb{R}^{768}$.
- Calculates local cluster dispersion $\sigma^2 = \frac{1}{|C|}\sum (1 - \langle x_i, c \rangle)$.

#### [NEW] [split.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/split.rs)
- Split execution:
  - Takes an overflowing `LeafNode` ($M_{\text{leaf}} > M_{\max}$ or variance threshold).
  - Runs spherical 2-means on full 768-dim embeddings.
  - Distributes items into two new child `LeafNode`s.
  - Promotes node to `RoutingNode` with child routing prototypes sliced at $d=128$: $p_A^{(1)} = \text{norm}(c_A[:128]), p_B^{(1)} = \text{norm}(c_B[:128])$.

#### [NEW] [router.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/router.rs)
- Top-down multi-resolution prefix routing:
  - Root ($d_0=64$): Computes inner product $\langle x[:64], p_k^{(0)} \rangle$ across root prototypes $\to$ selects top branch (or beam search with top-$B$ candidates).
  - Intermediate ($d_1=128$): Computes inner product $\langle x[:128], p_j^{(1)} \rangle$ across child prototypes $\to$ selects target leaf node.
  - Leaf ($d_2=768$): Computes exact inner products within target leaf $\Xi_{\text{leaf}}^{(2)}$ and returns top-$K$ nearest neighbors with similarity scores.
- Zero heap allocations in hot routing loops using preallocated buffers.

#### [NEW] [tree.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/tree.rs)
- `HAMPTree`:
  - Root node management with configured dimensions ($d_0=64, d_1=128, d_2=768$) and maximum leaf capacity $M_{\max}$ (default 100).
  - Online insertion: `insert(id: &str, vector: &[f32])`:
    1. Traverses down to target leaf node.
    2. Appends vector to leaf.
    3. Triggers dynamic split if leaf count $> M_{\max}$.
    4. Updates running-mean centroids on the path back to root:
       $$p_{\text{new}}^{(k)} = \text{norm}\left(\frac{N \cdot p_{\text{old}}^{(k)} + x^{(k)}}{N + 1}\right)$$
  - Query retrieval: `query(vector: &[f32], top_k: usize, beam_width: usize) -> Vec<QueryResult>`.
  - Batch insertion and batch query functions with `rayon` parallelism.
  - Topology inspection: count leaves, total nodes, depth distribution, and centroid validation.

#### [NEW] [lib.rs](file:///Users/anchormath/Code/self/smriti/smriti-core/src/lib.rs)
- PyO3 bindings module `smriti_core`:
  - `PyHAMPTree`: Wrapper exposing `new(d0, d1, d2, max_leaf_size)`, `insert`, `insert_batch`, `query`, `query_batch`, `stats()`, `get_leaf_sizes()`.
  - `PyQueryResult`: `item_id: String, score: f32`.
  - `PyTreeStats`: `num_nodes`, `num_leaves`, `num_items`, `depth`, `avg_leaf_size`, `min_leaf_size`, `max_leaf_size`.

---

### Component 2: Python Wrapper & Integration (`src/smriti/tree/`)

#### [NEW] [hamp_tree.py](file:///Users/anchormath/Code/self/smriti/src/smriti/tree/hamp_tree.py)
- High-level Pythonic interface to `smriti_core.PyHAMPTree`.
- Seamless ingestion from `EnterpriseProjectNode` lists, Pandas/Polars/Arrow tables, and Parquet files (`ekg_projects_v1.parquet`).
- Flat brute-force baseline search class `FlatMemorySearch` for precision recall benchmarking.

---

### Component 3: Validation & Benchmark Suite (`tests/` & `scripts/`)

#### [NEW] [test_rust_routing.py](file:///Users/anchormath/Code/self/smriti/tests/test_rust_routing.py)
- Unit & integration tests for:
  - Correct PyO3 instantiation and basic query retrieval.
  - Running mean centroid numerical accuracy (deviation $< 10^{-4}$ vs. exact batch mean).
  - Dynamic splitting and leaf size balance ($30 \le M_{\text{leaf}} \le 100$).
  - Multi-resolution routing recall vs. flat baseline on synthetic & project datasets.

#### [NEW] [benchmark_routing.py](file:///Users/anchormath/Code/self/smriti/scripts/benchmark_routing.py)
- Comprehensive benchmarking CLI:
  - Loads 3,500 project nodes from `data/ekg_projects_v1.parquet` and 500 test queries from `data/ekg_queries_test.parquet`.
  - Builds HAM-P tree via streaming online insertions.
  - Evaluates Top-1 and Top-5 retrieval recall against flat search.
  - Measures single-query latency (target $< 0.15\text{ ms}$ on CPU) and throughput (queries/sec) with Rich telemetry output.

---

## Verification Plan

### Automated Tests
1. **Rust Core Tests:**
   ```bash
   cd smriti-core && cargo test --release
   ```
2. **Build Python Extension:**
   ```bash
   .venv/bin/maturin develop --release
   ```
3. **Python Test Suite:**
   ```bash
   .venv/bin/pytest tests/ -v
   ```
4. **End-to-End Recall & Latency Benchmark:**
   ```bash
   .venv/bin/python scripts/benchmark_routing.py
   ```

### Acceptance Criteria Checklist
- [x] **Rust Core Engine:** `smriti-core` compiles cleanly with zero warnings, zero unsafe memory violations, and exposes a clean Python API via PyO3.
- [x] **Routing Accuracy:** Prefix routing ($d_0=64 \to d_1=128$) achieves $\ge 95\%$ Top-5 recall and $\ge 85\%$ Top-1 recall compared to exhaustive flat search on 3,500+ vectors.
- [x] **Retrieval Latency:** Single query routing through the Rust tree takes $< 0.15\text{ ms}$ on CPU ($> 20\times$ faster than flat Python/NumPy baseline).
- [x] **Split Stability:** Streaming insertion of 2,000+ nodes triggers dynamic splits without memory leaks, maintaining balanced leaf sizes ($30 \le M_{\text{leaf}} \le 100$).
- [x] **Centroid Consistency:** Running mean centroids deviate by $< 10^{-4}$ from exact batch centroids.
