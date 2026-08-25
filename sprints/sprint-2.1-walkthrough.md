# Sprint 2.1 Walkthrough: HAM-P Engine Hardening & Continuous Hopfield Dynamics

Sprint 2.1 focused on hardening the **Hierarchical Associative Memory Partitioning (HAM-P)** engine, eliminating edge-boundary recall drops, enforcing strict mathematical normalization invariants, and implementing the **Continuous Hopfield Dynamics & Lyapunov Energy minimization (HAM-R)** layer.

---

## 1. Summary of Changes

### Rust Core (`smriti-core`)
- [`matrix.rs`](file:///Users/anchormath/Code/self/smriti/smriti-core/src/matrix.rs):
  - Added `normalized_prefix_slice(v, dim)` guaranteeing in-flight unit-norm prefix subvectors.
  - Added `logsumexp(logits, beta)` with maximum subtraction for numerical stability across extreme $\beta \in [0.001, 100.0]$.
  - Added `softmax_stable(logits, beta, out_weights)` with zero-division fallback to uniform distributions.
- [`hopfield.rs`](file:///Users/anchormath/Code/self/smriti/smriti-core/src/hopfield.rs):
  - Implemented `continuous_hopfield_retrieve(memory_matrix, item_ids, query, beta, max_steps, tol)`.
  - Implemented exact continuous Lyapunov energy:
    $$E(x) = -\frac{1}{\beta}\text{LSE}(\beta \Xi^\top x) + \frac{1}{2}\|x\|_2^2$$
  - Implemented `resolution_to_beta(R, beta_min, beta_max)` geometric temperature interpolation:
    $$\beta(R) = \beta_{\min} \cdot \left(\frac{\beta_{\max}}{\beta_{\min}}\right)^R$$
  - Implemented `adaptive_density_beta(base_beta, dispersion, alpha)` to scale inverse temperature by local cluster density.
- [`router.rs`](file:///Users/anchormath/Code/self/smriti/smriti-core/src/router.rs):
  - Enforced normalized query prefix slicing across all routing tiers.
  - Implemented `route_multi_probe(root, query, beam_width, score_delta)` exploring all candidate branches where $S_i \ge S_{\max} - \delta$ up to `beam_width` branches.
- [`tree.rs`](file:///Users/anchormath/Code/self/smriti/smriti-core/src/tree.rs):
  - Updated streaming `insert()` to strictly use normalized prefix slices for centroid tracking.
  - Added `recenter()` for bottom-up lazy prototype recalculation from leaf vectors.
  - Added `query_hopfield()` and `query_hopfield_batch()` with Rayon parallelism.
- [`lib.rs`](file:///Users/anchormath/Code/self/smriti/smriti-core/src/lib.rs):
  - Exposed `PyHopfieldResult`, `query_hopfield`, `query_hopfield_batch`, `compute_lyapunov_energy`, and `recenter`.

### Python Wrapper & Baselines (`src/smriti/tree/`)
- [`hamp_tree.py`](file:///Users/anchormath/Code/self/smriti/src/smriti/tree/hamp_tree.py):
  - Added `query_hopfield()`, `sweep_resolution_dial()`, `compute_lyapunov_energy()`, `recenter()`, and `query_hopfield_batch()`.
  - Added exact continuous Hopfield retrieval and energy calculation to `FlatMemorySearch` for ground-truth baseline comparison.

---

## 2. Verification & Benchmark Results

### Benchmark Scorecard (`scripts/benchmark_hardening.py`)
Evaluated on **3,500 enterprise project nodes** against **500 test queries**:

#### Multi-Probe Routing Performance
| Routing Configuration | Top-1 Recall | Top-5 Recall | Mean Latency | p95 Latency |
| :--- | :--- | :--- | :--- | :--- |
| Greedy ($B=1, \delta=0.0$) | 73.60% | 70.28% | $18.0\,\mu\text{s}$ | $26.2\,\mu\text{s}$ |
| Standard Beam ($B=2, \delta=0.0$) | 89.20% | 90.20% | $22.4\,\mu\text{s}$ | $27.1\,\mu\text{s}$ |
| Standard Beam ($B=3, \delta=0.0$) | 92.20% | 93.84% | $26.8\,\mu\text{s}$ | $31.5\,\mu\text{s}$ |
| Multi-Probe ($B=3, \delta=0.05$) | 92.20% | 93.84% | $26.2\,\mu\text{s}$ | $31.5\,\mu\text{s}$ |
| **Multi-Probe ($B=4, \delta=0.08$)** | **93.40%** | **95.60%** | **$31.2\,\mu\text{s}$** | **$36.5\,\mu\text{s}$** |

#### Continuous Hopfield & Lyapunov Physics
* **Lyapunov Energy Monotonic Decay Compliance:** **100.0% (50/50 test queries strictly verified)**
* **Average Energy Minimization per Query:** **$-0.1745$**
* **Hopfield Iterative Convergence Latency:** **$215.1\,\mu\text{s}$** (single-core CPU)

#### Resolution Dial Dynamics
| Resolution ($R$) | State Regime | Final Energy $E(x)$ | Attention Entropy $H(a)$ | Max Attention Weight |
| :---: | :--- | :---: | :---: | :---: |
| **0.0** | Broad Cluster Blend | $-5.3633$ | $5.2781$ | $0.52\%$ |
| **0.2** | Broad Cluster Blend | $-2.5971$ | $5.2778$ | $0.54\%$ |
| **0.4** | Metastable Consensus | $-1.3776$ | $5.2760$ | $0.59\%$ |
| **0.6** | Metastable Consensus | $-0.8410$ | $5.1982$ | $0.85\%$ |
| **0.8** | Exact Precedent Attractor | $-0.6186$ | $4.7056$ | $1.53\%$ |
| **1.0** | Exact Precedent Attractor | $-0.5280$ | $3.5695$ | $5.56\%$ |

#### Centroid Drift & Re-Centering
* **Deviation before re-centering:** $4.76 \times 10^{-7}$
* **Deviation after re-centering:** $1.85 \times 10^{-7}$

---

## 3. Automated Test Suite

Ran full test suite via `.venv/bin/pytest tests/ -v`:
```
tests/test_dataset_pipeline.py::test_dataset_generation_and_export PASSED
tests/test_generators.py::test_taxonomy_coverage PASSED
tests/test_generators.py::test_seed_generator_rice_sampling PASSED
tests/test_generators.py::test_dag_relations_synthesis PASSED
tests/test_generators.py::test_llm_enricher_deterministic_fallback PASSED
tests/test_hamp_hardening.py::test_in_flight_mrl_normalization PASSED
tests/test_hamp_hardening.py::test_multi_probe_delta_routing PASSED
tests/test_hamp_hardening.py::test_lyapunov_energy_monotonic_decay PASSED
tests/test_hamp_hardening.py::test_resolution_dial_entropy_sweep PASSED
tests/test_hamp_hardening.py::test_lazy_recenter PASSED
tests/test_hamp_hardening.py::test_parallel_hopfield_batch PASSED
tests/test_mrl_pipeline.py::test_normalize_l2 PASSED
tests/test_mrl_pipeline.py::test_extract_mrl_slices PASSED
tests/test_mrl_pipeline.py::test_mrl_pipeline_batch_encoding PASSED
tests/test_project_schema.py::test_rice_factors_calculation_and_validation PASSED
tests/test_project_schema.py::test_mrl_embeddings_unit_normalization PASSED
tests/test_project_schema.py::test_enterprise_project_node_serialization PASSED
tests/test_rust_routing.py::test_smriti_core_module_import PASSED
tests/test_rust_routing.py::test_hamp_tree_basic_insertion_and_query PASSED
tests/test_rust_routing.py::test_running_centroid_consistency PASSED
tests/test_rust_routing.py::test_split_stability_and_leaf_balance PASSED
tests/test_rust_routing.py::test_prefix_routing_recall_vs_flat_search PASSED

============================== 22 passed in 2.88s ==============================
```
