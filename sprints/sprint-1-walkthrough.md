# Walkthrough: Sprint 1 — Data Engine & MRL Embeddings Spike

## Summary of Accomplishments

Sprint 1 objectives have been completed, establishing the testbed data engine, Pydantic project schemas, statistical RICE generators, DAG link synthesizers, and Matryoshka multi-resolution embedding pipelines ($d_0=64, d_1=128, d_2=768$).

---

## Key Deliverables & Code Structure

```
smriti/
├── pyproject.toml
├── src/
│   └── smriti/
│       ├── models/
│       │   └── project_schema.py       # Pydantic EnterpriseProjectNode & RICE Factor models
│       ├── generators/
│       │   ├── taxonomy.py             # 8 Domains, 32 Sub-domains, Epics & Archetypes
│       │   ├── seed_generator.py       # Log-normal Reach, right-skewed Effort & DAG Synthesizer
│       │   └── llm_enricher.py         # LLM Narrative Enricher (Gemini, Ollama, & Deterministic)
│       ├── embeddings/
│       │   └── mrl_pipeline.py         # Nomic MRL embedding engine with prefix slicing (d0, d1, d2)
│       ├── data/
│       │   └── dataset_builder.py      # EKG Dataset builder & Parquet/SQLite exporters
│       └── utils/
│           └── logging.py              # Rich CLI telemetry & tables
├── scripts/
│   ├── generate_dataset.py             # CLI generation command
│   └── validate_dataset.py             # CLI validation suite
├── tests/                              # 11 Unit & Integration Tests (100% passing)
│   ├── test_project_schema.py
│   ├── test_generators.py
│   ├── test_mrl_pipeline.py
│   └── test_dataset_pipeline.py
└── data/                               # Packaged dataset artifacts
    ├── ekg_projects_v1.parquet         # 3,500 project nodes with MRL vector columns (17.1 MB)
    ├── ekg_projects_v1.db              # Relational SQLite DB with indexes and embeddings (79.0 MB)
    ├── ekg_queries_test.parquet        # 500 held-out evaluation queries (2.7 MB)
    └── ekg_queries_test.json           # JSON query companion (13.0 MB)
```

---

## Acceptance Criteria Verification

All four Acceptance Criteria from the Sprint 1 specification are verified:

| Acceptance Criterion | Specification | Result | Status |
| :--- | :--- | :--- | :--- |
| **1. Dataset Size & Diversity** | $\ge 3,000$ validated project nodes across 8 domains and $\ge 24$ sub-domains | **3,500 nodes** across **8 domains** and **32 sub-domains** | **PASS** |
| **2. Schema Validity** | 100% of records pass Pydantic schema validation without missing fields | **0 failures** out of 3,500 records ($100\%$ valid) | **PASS** |
| **3. MRL Vector Normalization** | Vector slices at $d_0=64, d_1=128, d_2=768$ strictly unit-normalized ($|\|v\|_2 - 1.0| < 10^{-4}$) | Maximum norm error: **$1.19 \times 10^{-7}$** | **PASS** |
| **4. RICE Mathematical Integrity** | Strict $\frac{R \cdot I \cdot C}{E}$ calculation, log-normal Reach, right-skewed Effort | **0 calculation errors**, realistic enterprise distributions verified | **PASS** |

---

## Test & Validation Execution

### 1. Automated Test Suite (pytest)
```bash
./.venv/bin/python -m pytest -v
```
Output:
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/anchormath/Code/self/smriti
collected 11 items

tests/test_dataset_pipeline.py::test_dataset_generation_and_export PASSED [  9%]
tests/test_generators.py::test_taxonomy_coverage PASSED                  [ 18%]
tests/test_generators.py::test_seed_generator_rice_sampling PASSED       [ 27%]
tests/test_generators.py::test_dag_relations_synthesis PASSED            [ 36%]
tests/test_generators.py::test_llm_enricher_deterministic_fallback PASSED [ 45%]
tests/test_mrl_pipeline.py::test_normalize_l2 PASSED                     [ 54%]
tests/test_mrl_pipeline.py::test_extract_mrl_slices PASSED               [ 63%]
tests/test_mrl_pipeline.py::test_mrl_pipeline_batch_encoding PASSED      [ 72%]
tests/test_project_schema.py::test_rice_factors_calculation_and_validation PASSED [ 81%]
tests/test_project_schema.py::test_mrl_embeddings_unit_normalization PASSED [ 90%]
tests/test_project_schema.py::test_enterprise_project_node_serialization PASSED [100%]

============================== 11 passed in 2.88s ==============================
```

### 2. Dataset Generation
```bash
./.venv/bin/python scripts/generate_dataset.py --num-items 3500 --num-queries 500
```
- Generated 3,500 full project nodes and 500 test queries.
- Multi-resolution MRL embeddings produced using `nomic-ai/nomic-embed-text-v1.5` on Apple Silicon Metal Performance Shaders (`mps`).

### 3. Acceptance Verification
```bash
./.venv/bin/python scripts/validate_dataset.py
```
Output:
```text
─────────────── Validating Dataset: data/ekg_projects_v1.parquet ───────────────
Loaded 3500 records from Parquet.

1. Dataset Size & Diversity Check:
  • Total Records: 3500 (Target: >= 3,000) -> PASS
  • Unique Domains: 8 (Target: >= 8) -> PASS
  • Unique Sub-Domains: 32 (Target: >= 24) -> PASS

2. Pydantic Schema Validity Check:
  • Schema Failures: 0 / 3500 -> PASS

3. MRL Vector Normalization Integrity Check:
  • d0 (64) Normalization Failures: 0
  • d1 (128) Normalization Failures: 0
  • d2 (768) Normalization Failures: 0
  • Max Vector Norm Error: 1.19e-07 (Threshold: < 1e-4) -> PASS

4. RICE Mathematical Consistency & Distributions:
  • Reach Distribution: Min=20.0, Median=42334.1, Max=6891903.9
  • Effort Distribution: Min=1.0, Median=7.5, Max=40.5
  • Impact Scores Present: [0.25, 0.5, 1.0, 2.0, 3.0]
  • Confidence Percentages: [0.5, 0.8, 1.0]
  • RICE Score Calculation Errors: 0 -> PASS
```

---

## Ready for Sprint 2
The data engine and multi-resolution vector artifacts are ready for **Sprint 2: Rust HAM-P Tree Infrastructure & Routing Engine** (`smriti-core`).
