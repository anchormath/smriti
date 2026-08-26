# `smriti` — Python Data Engine & Orchestration Layer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat-square&logo=python)](../pyproject.toml)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg?style=flat-square)](https://docs.pydantic.dev/)
[![PyArrow](https://img.shields.io/badge/PyArrow-Parquet-orange.svg?style=flat-square)](https://arrow.apache.org/docs/python/)
[![PyTest](https://img.shields.io/badge/PyTest-22%2F22%20Passing-success.svg?style=flat-square)](../tests/)

`src/smriti` is the high-level Python layer of **Smriti (स्मृति)**. It is responsible for data modeling, synthetic enterprise knowledge graph (EKG) generation, Matryoshka Representation Learning (MRL) embeddings, dataset packaging, and orchestrating native Rust inference.

---

## 1. Architecture & Package Structure

```
src/smriti/
├── models/         # Pydantic v2 schemas (EnterpriseProjectNode, RICEFactors, DAGDependency)
├── generators/     # Synthetic EKG seed distributions, taxonomy trees, LLM narrative enrichers
├── embeddings/     # Nomic MRL embedding pipeline (d0=64, d1=128, d2=768 unit-normalized slicing)
├── data/           # DatasetBuilder for streaming synthesis and Parquet / SQLite packaging
├── tree/           # HAMPTreeWrapper (PyO3 bridge) and FlatMemorySearch (exact brute-force baseline)
└── utils/          # Rich telemetry, console logging, and performance utilities
```

### Module Responsibilities

* **`smriti.models`:**
  * [`project_schema.py`](smriti/models/project_schema.py): Defines `EnterpriseProjectNode`, `RICEFactors` ($R \times I \times C / E$), `DAGDependency`, and multi-resolution vector representations (`mrl_d0_64`, `mrl_d1_128`, `mrl_d2_768`).

* **`smriti.generators`:**
  * [`taxonomy.py`](smriti/generators/taxonomy.py): Hierarchical enterprise taxonomy across 4 domains (Core Infrastructure, Security & Compliance, Customer Platform, Data & AI) and 32 specialized sub-domains.
  * [`seed_generator.py`](smriti/generators/seed_generator.py): Samples statistical distributions (log-normal Reach, right-skewed Effort, calibrated Confidence) and synthesizes DAG dependency graphs.
  * [`llm_enricher.py`](smriti/generators/llm_enricher.py): Generates technical project problem statements, architectural narratives, and outcomes via Gemini / Ollama or robust deterministic fallback templates.

* **`smriti.embeddings`:**
  * [`mrl_pipeline.py`](smriti/embeddings/mrl_pipeline.py): Batched inference with `nomic-ai/nomic-embed-text-v1.5`, extracting nested MRL slices ($d_0=64, d_1=128, d_2=768$) and enforcing $L_2$ unit normalization on each slice.

* **`smriti.data`:**
  * [`dataset_builder.py`](smriti/data/dataset_builder.py): Coordinates end-to-end dataset generation, exporting 3,500 project nodes and 500 test queries to columnar Parquet and SQLite formats.

* **`smriti.tree`:**
  * [`hamp_tree.py`](smriti/tree/hamp_tree.py):
    * `HAMPTreeWrapper`: High-level Python interface interacting with the native `smriti_core` Rust module. Supports online streaming insertions, multi-probe beam queries, continuous Hopfield associative retrieval, Resolution Dial sweeps, and lazy re-centering.
    * `FlatMemorySearch`: Exact $O(M)$ NumPy brute-force baseline for computing ground-truth retrieval recall and Hopfield trajectories.

* **`smriti.utils`:**
  * [`logging.py`](smriti/utils/logging.py): Rich formatting and console output helpers.

---

## 2. Usage & Development

### Import and High-Level Usage
```python
from smriti.tree import HAMPTreeWrapper, FlatMemorySearch
import numpy as np

# Load pre-built 3,500 node EKG dataset directly into native Rust tree
tree = HAMPTreeWrapper.from_parquet("data/ekg_projects_v1.parquet")
print(f"Loaded {len(tree)} projects into HAM-P tree.")

# Multi-probe routing query
query_vec = np.random.randn(768).astype(np.float32)
query_vec /= np.linalg.norm(query_vec)

results = tree.query_multi_probe(query_vec, top_k=5, beam_width=3, score_delta=0.05)
for r in results:
    print(f"ID: {r.item_id:<20} Cosine Score: {r.score:.4f}")

# Continuous Hopfield associative retrieval (Resolution Dial R=0.7)
hop_res = tree.query_hopfield(query_vec, resolution=0.7, beam_width=3, max_steps=5)
print(f"Final Lyapunov Energy: {hop_res.final_energy:.4f}, Iterations: {len(hop_res.energy_trajectory)-1}")
```

### Running Test Suite
```bash
# Run full pytest suite across all modules
pytest tests/ -v
```
