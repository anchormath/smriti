# Smriti (स्मृति) — Dynamic Partitioned Hierarchical Associative Memory Engine

[![Rust Core](https://img.shields.io/badge/Rust-1.80+-orange.svg?style=flat-square&logo=rust)](smriti-core/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat-square&logo=python)](pyproject.toml)
[![License: TBD](https://img.shields.io/badge/License-TBD-lightgrey.svg?style=flat-square)](#license)
[![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passing-success.svg?style=flat-square)](tests/)
[![Latency](https://img.shields.io/badge/Latency-26.6%20µs-brightgreen.svg?style=flat-square)](#benchmark-results)
[![Recall](https://img.shields.io/badge/Top--5%20Recall-95.8%25-blueviolet.svg?style=flat-square)](#benchmark-results)

> **Smriti (स्मृति)** is a high-performance, non-parametric associative memory and retrieval engine designed for large-scale Enterprise Knowledge Graphs (EKG) and dense vector spaces.
> It unifies **Hierarchical Associative Memory Partitioning (HAM-P)** with **Continuous Hopfield Dynamics & Resolution Dial (HAM-R)**, implemented with a native **Rust core (`smriti-core`)** and exposed via **PyO3** to Python.

---

## 1. Architectural Vision & Theoretical Foundations

### Why HAM-P / HAM-R?
Classical modern continuous Hopfield networks evaluate queries across a single flat memory matrix $\Xi \in \mathbb{R}^{d \times M}$. For large enterprise collections ($M \gg 10^4$), flat evaluation suffers from two major limitations:
1. **Computational Bottleneck:** Inner products against all $M$ memories at dimension $d$ scale as $O(M \cdot d)$, becoming computationally prohibitive for real-time inference.
2. **Cross-Talk & Spurious Basins:** Co-locating disparate semantic domains within a single global state distorts the energy landscape, leading to spurious intermediate attractors and requiring fragile global temperature tuning.

**Smriti** solves both problems by replacing flat storage with a **routed tree of specialized memory pools** coupled with **Matryoshka Representation Learning (MRL)** prefix slicing and **continuous Hopfield energy minimization**.

```
                        Layer 0: Root Router [d0 = 64]
                   Evaluates x[:64] vs K0 Domain Prototypes
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
      Sub-domain A              Sub-domain B              Sub-domain C
     Layer 1 [d1 = 128]        Layer 1 [d1 = 128]        Layer 1 [d1 = 128]
     Evaluates x[:128]         Evaluates x[:128]         Evaluates x[:128]
            │
      ┌─────┴─────┐
      ▼           ▼
    Leaf 1      Leaf 2
  Layer 2 [d2 = 768] (Contiguous Column-Major Memory Buffer, M_leaf <= 100)
  Continuous Hopfield Associative Retrieval & Pattern Completion:
    x_out = Ξ_leaf · softmax(β(R) · Ξ_leafᵀ · x)
```

---

## 2. Key Capabilities

* **Multi-Resolution Prefix Routing (HAM-P):** Accesses multi-resolution views of unified MRL vectors through contiguous prefix slices ($d_0=64 \to d_1=128 \to d_2=768$) with **zero heap allocations** and **zero memcpy operations** during top-down traversal.
* **Continuous Hopfield Dynamics (HAM-R):** Continuous energy landscape minimization governed by the Lyapunov energy function:
  $$E(x) = -\frac{1}{\beta} \log \sum_{i=1}^M \exp\left(\beta \cdot \xi_i^\top x\right) + \frac{1}{2} \Vert{}x\Vert{}_2^2$$
* **Continuous Resolution Dial ($R \in [0.0, 1.0]$):** Smoothly transitions retrieval dynamics from broad semantic cluster blending ($R \to 0$, low $\beta$) to exact, isolated pattern completion ($R \to 1$, high $\beta$).
* **Online Streaming Ingestion & Dynamic 2-Means Splitting:** Maintains running-mean routing centroids in $O(\text{depth})$ time with exact cumulative sum accumulators ($< 10^{-6}$ deviation from batch mean) and triggers spherical 2-means leaf bisection when $M_{\text{leaf}} > M_{\max}$.
* **SIMD Hardware Acceleration & Rayon Multithreading:** Utilizes ARM NEON (Apple Silicon) and AVX2 fused multiply-accumulate primitives, delivering single-query CPU latencies of **$26.6\,\mu\text{s}$** and batch throughputs exceeding **$47,000\text{ queries/sec}$**.

---

## 3. System Architecture & Language Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Python Layer (Data Engine, Taxonomy, Orchestration & Evaluation)           │
│   • Enterprise Schema & RICE Models: Pydantic v2                            │
│   • Synthetic Narrative Generator: LLM Enricher (Gemini / Ollama / Templates)│
│   • MRL Feature Encoding: Nomic Embed Text v1.5 (d0=64, d1=128, d2=768)    │
│   • High-Level Interface: HAMPTreeWrapper & FlatMemorySearch                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ PyO3 / Maturin C-ABI
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Rust Core Engine (`smriti-core`)                                            │
│   • Memory Layout: Contiguous column-major matrices (ColumnMatrix)          │
│   • Hardware Math: SIMD dot products (ARM NEON `vfmaq_f32` / AVX2)          │
│   • Concurrency: Rayon parallel batch query mapping                         │
│   • Topology: 3-Layer HAM-P tree with dynamic spherical 2-means splits     │
│   • Centroids: Exact dual-matrix running sums (sum_prototypes & prototypes) │
│   • Retrieval: Top-down prefix routing, Multi-probe & Hopfield attention    │
└─────────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Naksha Integration & Domain Independence:**
> Much of the current Python layer acts as an adapter for **[Naksha](https://github.com/anchormath/naksha/tree/naksha2)** (Enterprise Architecture & Knowledge Graph engine), utilizing enterprise initiative nodes and RICE scoring models as a rigorous evaluation testbed. The core memory engine (`smriti-core` and the retrieval tree) is designed as a standalone, domain-agnostic associative memory architecture.

---

## 4. Benchmark Results

Evaluated on **3,500 Enterprise Project Nodes** ($d=768$) against **500 Held-Out Queries** using single-core CPU execution:

| Metric | Flat NumPy Baseline | Rust HAM-P Tree (`smriti-core`) | Status / Target |
| :--- | :--- | :--- | :--- |
| **Top-1 Recall** | 100.0% (exhaustive) | **93.80%** (469 / 500) | **PASS** ($\ge 85.0\%$) |
| **Top-5 Recall** | 100.0% (exhaustive) | **95.84%** (479 / 500) | **PASS** ($\ge 95.0\%$) |
| **Mean Single-Query Latency** | $0.1774\text{ ms}$ ($177.4\,\mu\text{s}$) | **$0.0266\text{ ms}$ ($26.6\,\mu\text{s}$)** | **$6.7\times$ faster** ($< 150\,\mu\text{s}$) |
| **Median Latency (p50)** | $0.1762\text{ ms}$ | **$0.0265\text{ ms}$ ($26.5\,\mu\text{s}$)** | **Ultra-low jitter** |
| **p95 Latency** | $0.1930\text{ ms}$ | **$0.0324\text{ ms}$ ($32.4\,\mu\text{s}$)** | **Deterministic** |
| **Streaming Ingestion Rate** | — | **$50,622\text{ items/sec}$** | **High Throughput** |
| **Multithreaded QPS (Rayon)**| — | **$47,554\text{ queries/sec}$** | **Scalable** |
| **Centroid Drift Error** | — | **$3.43 \times 10^{-7}$** | **PASS** ($< 10^{-4}$) |

### Compute FLOP Reduction
$$\text{FLOPs}_{\text{flat}} = M \cdot d_2 = 3,500 \times 768 = 2,688,000\text{ ops}$$
$$\text{FLOPs}_{\text{tree}} = K_0 d_0 + K_1 d_1 + M_{\text{leaf}} d_2 = (8 \times 64) + (4 \times 128) + (100 \times 768) = 77,824\text{ ops}$$
$$\implies \mathbf{34.5\times \text{ compute reduction per query}}$$

---

## 5. Repository Structure

```
smriti/
├── pyproject.toml                              # Python package metadata & dependencies
├── ham-p.md                                    # Architectural specification for HAM-P
├── ham-r.md                                    # Architectural specification for HAM-R
├── smriti-core/                                # High-performance native Rust crate
│   ├── Cargo.toml                              # Rust dependencies (PyO3, Rayon, Serde)
│   └── src/
│       ├── lib.rs                              # PyO3 bindings module (`smriti_core`)
│       ├── matrix.rs                           # Contiguous column-major storage & SIMD dot products
│       ├── node.rs                             # LeafNode, RoutingNode, and HAMPNode structures
│       ├── router.rs                           # Top-down prefix routing & Top-K beam search
│       ├── kmeans.rs                           # Spherical 2-means clustering with furthest-point seeding
│       ├── split.rs                            # Dynamic leaf splitting & router promotion
│       └── tree.rs                             # HAMPTree container, streaming insertion & centroids
├── src/
│   └── smriti/                                 # Python package
│       ├── models/                             # Pydantic schemas (EnterpriseProjectNode, RICE)
│       ├── generators/                         # Seed data, taxonomy & narrative enrichers
│       ├── embeddings/                         # Nomic MRL embedding pipeline (d0, d1, d2)
│       ├── data/                               # Dataset builder & Parquet/SQLite exporters
│       ├── tree/                               # High-level Python HAMPTreeWrapper & Flat baseline
│       └── utils/                              # Telemetry & Rich logger
├── scripts/
│   ├── generate_dataset.py                     # CLI dataset generation command
│   ├── validate_dataset.py                     # CLI validation check against Acceptance Criteria
│   └── benchmark_routing.py                    # Recall, latency, and throughput benchmark suite
├── tests/                                      # Automated test suite (16 tests, 100% passing)
│   ├── test_project_schema.py
│   ├── test_generators.py
│   ├── test_mrl_pipeline.py
│   ├── test_dataset_pipeline.py
│   └── test_rust_routing.py
├── sprints/                                    # Sprint specifications, plans & walkthroughs
│   ├── sprint-1-data-engine-mrl-spike.md
│   ├── sprint-1-implementation-plan.md
│   ├── sprint-1-walkthrough.md
│   ├── sprint-2-ham-p-tree-routing.md
│   ├── sprint-2-implementation-plan.md
│   ├── sprint-2-walkthrough.md
│   ├── sprint-2.1-ham-p-hardening.md
│   ├── sprint-2.1-implementation-plan.md
│   └── sprint-3-hopfield-resolution-dial-eval.md
└── data/                                       # Packaged Parquet datasets (3,500 projects, 500 queries)
    ├── ekg_projects_v1.parquet
    ├── ekg_queries_test.parquet
    └── ekg_queries_test.json
```

---

## 6. Installation & Quickstart

### Prerequisites
* **Python $\ge 3.10$**
* **Rust toolchain ($\ge 1.80$) & Cargo** (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
* **Maturin** (`pip install maturin` or `brew install maturin`)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/anchormath/smriti.git
cd smriti

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e ".[dev]"
```

### 2. Build the Rust Native Extension (`smriti_core`)
```bash
# Build and install optimized release extension in editable environment
maturin develop --release --manifest-path smriti-core/Cargo.toml
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

### 4. Run Recall & Latency Benchmark
```bash
python scripts/benchmark_routing.py
```

---

## 7. Usage Examples

### Building and Querying the HAM-P Tree in Python
```python
import numpy as np
from smriti.tree import HAMPTreeWrapper

# Initialize a 3-layer HAM-P Tree (d0=64, d1=128, d2=768, leaf_capacity=100)
tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=100)

# 1. Online Streaming Insertions
sample_vector = np.random.randn(768).astype(np.float32)
sample_vector /= np.linalg.norm(sample_vector)
tree.insert("project_alpha", sample_vector)

# 2. Bulk Ingestion directly from Parquet
tree = HAMPTreeWrapper.from_parquet("data/ekg_projects_v1.parquet")
print(f"Indexed {len(tree)} projects. Topology stats: {tree.stats()}")

# 3. Top-K Nearest Neighbor Retrieval with Beam Search
query_vec = np.random.randn(768).astype(np.float32)
query_vec /= np.linalg.norm(query_vec)

results = tree.query(query_vec, top_k=5, beam_width=3)
for r in results:
    print(f"Item: {r.item_id:<20} Cosine Similarity: {r.score:.4f}")

# 4. Multithreaded Batch Queries (Rayon Parallelism)
batch_queries = np.random.randn(100, 768).astype(np.float32)
batch_queries /= np.linalg.norm(batch_queries, axis=1, keepdims=True)

batch_results = tree.query_batch(batch_queries, top_k=5, beam_width=3)
print(f"Retrieved Top-5 for {len(batch_results)} batch queries in parallel.")
```

---

## 8. References & Theoretical Background

1. **Modern Hopfield Networks:** Ramsauer et al., *"Hopfield Networks is All You Need"*, ICLR 2021.
2. **Matryoshka Representation Learning:** Kusupati et al., *"Matryoshka Representation Learning"*, NeurIPS 2022.
3. **Nomic Embed:** Nussbaum et al., *"Nomic Embed: Training a Reproducible Long Context Text Embedder"*, 2024.

---

## License

License is TBD.


