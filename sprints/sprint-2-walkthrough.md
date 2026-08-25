# Walkthrough: Sprint 2 — Rust HAM-P Tree Infrastructure & Routing Engine (`smriti-core`)

## Summary of Accomplishments

Sprint 2 objectives have been completed, establishing the high-performance **Rust `smriti-core` Engine** for the **Hierarchical Partitioned Associative Memory (HAM-P)** multi-resolution tree topology ($d_0=64 \to d_1=128 \to d_2=768$), online streaming ingestion, spherical 2-means dynamic leaf splitting, running-mean centroid maintenance, and PyO3 Python bindings.

---

## Key Deliverables & Code Structure

```
smriti/
├── smriti-core/                                # High-performance Rust engine crate
│   ├── Cargo.toml                              # PyO3, Rayon, Serde, Rand dependencies
│   └── src/
│       ├── lib.rs                              # PyO3 module exposing HAMPTree, QueryResult, TreeStats
│       ├── matrix.rs                           # Contiguous column-major memory matrices & SIMD dot products (ARM NEON)
│       ├── node.rs                             # LeafNode, RoutingNode, and HAMPNode tree structures
│       ├── router.rs                           # Top-down prefix routing & Top-K beam search
│       ├── kmeans.rs                           # Spherical 2-means clustering with cosine similarity
│       ├── split.rs                            # Dynamic leaf splitting & intermediate router promotion
│       └── tree.rs                             # HAMPTree container, online insertion & centroid maintenance
├── pyproject.toml                              # Pytest and module configuration
├── src/
│   └── smriti/
│       └── tree/
│           ├── __init__.py                     # Tree routing module entry point
│           └── hamp_tree.py                    # HAMPTreeWrapper & FlatMemorySearch baseline
├── scripts/
│   └── benchmark_routing.py                    # Recall, latency, and throughput benchmark suite
└── tests/
    └── test_rust_routing.py                    # PyTest suite for PyO3 bindings, recall, split stability & centroids
```

---

## Acceptance Criteria Verification

All five Acceptance Criteria from the Sprint 2 specification are verified:

| Acceptance Criterion | Specification | Benchmark Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Rust Core Engine** | Compiles cleanly with zero warnings, zero unsafe memory violations, PyO3 API | **0 warnings**, clean PyO3 module `smriti_core` | **PASS** |
| **2. Routing Accuracy (Top-1)** | $\ge 85.0\%$ Top-1 recall vs exhaustive flat search on 3,500 projects | **93.80%** (469 / 500 queries matched Top-1) | **PASS** |
| **3. Routing Accuracy (Top-5)** | $\ge 95.0\%$ Top-5 recall vs exhaustive flat search on 3,500 projects | **95.84%** Top-5 overlap across 500 test queries | **PASS** |
| **4. Retrieval Latency** | Single-query latency $< 0.15\text{ ms}$ ($< 150\,\mu\text{s}$) on CPU | **0.0266 ms** ($26.6\,\mu\text{s}$) mean latency | **PASS** |
| **5. Split Stability & Balance** | Streaming insertion of 2,000+ nodes triggers splits with balanced leaves | **65 leaves**, mean size $53.8$, max size $100$ | **PASS** |
| **6. Centroid Consistency** | Running mean centroids deviate by $< 10^{-4}$ from exact batch centroids | **$3.43 \times 10^{-7}$** maximum L2 deviation | **PASS** |

---

## Benchmark Results

```text
╭──────────────────────────────────────────────────────────────────────────────╮
│ Hierarchical Partitioned Associative Memory (HAM-P) Rust Engine Benchmark    │
│ Evaluating Rust smriti-core tree routing vs. Flat NumPy baseline             │
╰──────────────────────────────────────────────────────────────────────────────╯
• Loaded 3,500 project nodes (d=768)
• Loaded 500 evaluation queries (d=768)

1. Building HAM-P Tree via Online Streaming Insertions...
           HAM-P Tree Topology Statistics            
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Metric                         ┃ Value            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Total Items Indexed            │ 3,500            │
│ Total Leaf Nodes               │ 65               │
│ Tree Max Depth                 │ 3                │
│ Average Leaf Size              │ 53.8 items       │
│ Min Leaf Size                  │ 2 items          │
│ Max Leaf Size                  │ 100 items        │
│ Total Build Time               │ 69.14 ms         │
│ Streaming Ingestion Throughput │ 50,622 items/sec │
└────────────────────────────────┴──────────────────┘
• Maximum Running Centroid Deviation: 3.4262e-07 (Threshold: < 1e-4) -> PASS

2. Evaluating Retrieval Recall vs. Flat Brute-Force Baseline...
     Retrieval Recall vs Flat Baseline (500 Queries)      
┏━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Recall Metric ┃ Result ┃ Acceptance Criterion ┃ Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┓
│ Top-1 Recall  │ 93.80% │ >= 85.0%             │ PASS   │
│ Top-5 Recall  │ 95.84% │ >= 95.0%             │ PASS   │
└───────────────┴────────┴──────────────────────┴────────┘

3. Benchmarking Single-Query Latency...
                      Latency Benchmark (CPU Single-Core)                       
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Engine     ┃ Mean       ┃ Median     ┃ p95        ┃ p99 Latency ┃ Speedup vs ┃
┃            ┃ Latency    ┃ (p50)      ┃ Latency    ┃             ┃ Flat       ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Flat NumPy │ 0.1774 ms  │ 0.1762 ms  │ 0.1930 ms  │ 0.2075 ms   │ 1.0x       │
│ Baseline   │            │            │            │             │            │
│ Rust HAM-P │ 0.0266 ms  │ 0.0265 ms  │ 0.0324 ms  │ 0.0364 ms   │ 6.7x       │
│ Tree       │ (26.6 µs)  │ (26.5 µs)  │ (32.4 µs)  │ (36.4 µs)   │ faster     │
│ (beam=3)   │            │            │            │             │            │
└────────────┴────────────┴────────────┴────────────┴─────────────┴────────────┘

4. Benchmarking Multithreaded Batch Throughput (Rayon)...
• Evaluated 500 batch queries in 10.51 ms
• Multithreaded Throughput: 47,554 queries/sec
```

---

## Test Execution Summary

### Automated Test Suite (16/16 Passed)
```bash
.venv/bin/pytest tests/ -v
```
Output:
```text
tests/test_dataset_pipeline.py::test_dataset_generation_and_export PASSED [  6%]
tests/test_generators.py::test_taxonomy_coverage PASSED                  [ 12%]
tests/test_generators.py::test_seed_generator_rice_sampling PASSED       [ 18%]
tests/test_generators.py::test_dag_relations_synthesis PASSED            [ 25%]
tests/test_generators.py::test_llm_enricher_deterministic_fallback PASSED [ 31%]
tests/test_mrl_pipeline.py::test_normalize_l2 PASSED                     [ 37%]
tests/test_mrl_pipeline.py::test_extract_mrl_slices PASSED               [ 43%]
tests/test_mrl_pipeline.py::test_mrl_pipeline_batch_encoding PASSED      [ 50%]
tests/test_project_schema.py::test_rice_factors_calculation_and_validation PASSED [ 56%]
tests/test_project_schema.py::test_mrl_embeddings_unit_normalization PASSED [ 62%]
tests/test_project_schema.py::test_enterprise_project_node_serialization PASSED [ 68%]
tests/test_rust_routing.py::test_smriti_core_module_import PASSED        [ 75%]
tests/test_rust_routing.py::test_hamp_tree_basic_insertion_and_query PASSED [ 81%]
tests/test_rust_routing.py::test_running_centroid_consistency PASSED     [ 87%]
tests/test_rust_routing.py::test_split_stability_and_leaf_balance PASSED [ 93%]
tests/test_rust_routing.py::test_prefix_routing_recall_vs_flat_search PASSED [100%]

============================== 16 passed in 3.87s ==============================
```

---

## Ready for Sprint 3
The Rust core engine and multi-resolution prefix routing tree are verified and ready for **Sprint 3: Continuous Hopfield Dynamics, Resolution Dial & Evaluation Engine** (`HAM-R`).
