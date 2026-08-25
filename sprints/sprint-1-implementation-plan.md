# Implementation Plan: Sprint 1 — Data Engine & MRL Embeddings Spike

This plan establishes the foundation for the **HAM-P** and **HAM-R** associative memory system. In Sprint 1, we implement the Enterprise Knowledge Graph (EKG) project node schema, build a high-fidelity synthetic data generator with realistic RICE metrics and graph relationships, construct a Matryoshka Representation Learning (MRL) prefix-embedding pipeline, and package $\ge 3,500$ validated project nodes and 500 test queries into Parquet and SQLite formats.

---

## User Review Required

> [!NOTE]
> - **Embedding Model:** We will configure `nomic-ai/nomic-embed-text-v1.5` as the primary MRL encoder ($d_{\text{full}}=768$, with prefix slices $d_0=64$, $d_1=128$, $d_2=768$). We also support offline/deterministic embedding fallbacks and other standard sentence transformer models.
> - **Synthetic Generation Mode:** The generation engine supports both live LLM enrichment (via Gemini API / Ollama when credentials/endpoints are present) and an intelligent high-diversity template generator ensuring $\ge 3,500$ rich, distinct enterprise engineering specs without requiring external API tokens or incurring latency bottlenecks.

---

## Proposed Changes

We will create a clean, modular Python package `smriti/` with CLI entry points and automated tests.

```
smriti/
├── pyproject.toml
├── src/
│   └── smriti/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── project_schema.py       # Pydantic schemas, validation, RICE logic, MRL serializers
│       ├── generators/
│       │   ├── __init__.py
│       │   ├── taxonomy.py             # 8 Domains, >=24 Sub-domains, Tech stacks, Archetypes
│       │   ├── seed_generator.py       # Statistical RICE generators, DAG graph link synthesis
│       │   └── llm_enricher.py         # Structured technical narrative enricher (Gemini/Ollama/Deterministic)
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── mrl_pipeline.py         # MRL embedding encoder, prefix slicing (d0=64, d1=128, d2=768), L2 norm
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset_builder.py      # Pipeline orchestrator, Parquet & SQLite export, query split
│       └── utils/
│           ├── __init__.py
│           └── logging.py              # Rich logging & progress display
├── scripts/
│   ├── generate_dataset.py             # CLI command to generate, embed, and package EKG dataset
│   └── validate_dataset.py             # CLI validation check against all acceptance criteria
├── tests/
│   ├── __init__.py
│   ├── test_project_schema.py          # Unit tests for Pydantic schema and RICE math
│   ├── test_generators.py              # Unit tests for taxonomy, seed distributions, and DAG relations
│   ├── test_mrl_pipeline.py            # Unit tests for MRL embedding generation, slicing, and L2 normalization
│   └── test_dataset_pipeline.py        # End-to-end pipeline & Parquet/SQLite storage tests
└── data/                               # Generated dataset artifacts (.parquet, .db, .json)
```

---

### Component 1: Data Models & Schema (`smriti/models/`)

#### [NEW] [project_schema.py](file:///Users/anchormath/Code/self/smriti/src/smriti/models/project_schema.py)
- Defines Pydantic v2 models:
  - `Taxonomy`: `domain`, `sub_domain`, `epic`
  - `SemanticContext`: `title`, `problem_statement`, `solution_proposal`, `tech_stack`, `target_users`
  - `Reach`: `value` (float), `unit` (str)
  - `Impact`: `score` (enum `[0.25, 0.5, 1.0, 2.0, 3.0]`), `tier` (enum `["Minimal", "Low", "Medium", "High", "Massive"]`), `rationale` (str)
  - `Confidence`: `percentage` (enum `[0.5, 0.8, 1.0]`), `tier` (enum `["Low / Moonshot", "Medium", "High"]`), `evidence` (str)
  - `EffortBreakdown`: `frontend`, `backend`, `infra_devops`, `qa_testing` (floats)
  - `Effort`: `person_weeks` (float), `breakdown` (optional `EffortBreakdown`)
  - `RICEFactors`: `reach`, `impact`, `confidence`, `effort`, `rice_score` (computed/validated as $\frac{R \cdot I \cdot C}{E}$)
  - `GraphRelations`: `upstream_dependencies` (list[str]), `downstream_impacted` (list[str]), `tags` (list[str])
  - `MRLEmbeddings`: `d0_64` (list[float]), `d1_128` (list[float]), `d2_768` (list[float])
  - `EnterpriseProjectNode`: Root model tying together all components with methods:
    - `to_unified_text()`: Serializes node to the standardized search format:
      `search_document: [Domain: ...] [Sub-domain: ...] Title: ... Problem: ... Solution: ... Tech Stack: ... Target Audience: ...`
    - `to_flat_dict()`: Flattens hierarchical structures for columnar Parquet export.

---

### Component 2: Generators & Narrative Enrichment (`smriti/generators/`)

#### [NEW] [taxonomy.py](file:///Users/anchormath/Code/self/smriti/src/smriti/generators/taxonomy.py)
- Defines 8 core domains and 4-6 sub-domains each ($\ge 32$ sub-domains total):
  1. *Developer Productivity & CI/CD* (Build Caching, Test Automation, Dev Environments, Release Engineering)
  2. *Data Infrastructure & Analytics* (Stream Processing, Data Lakehouse, Metadata & Cataloging, Query Engines)
  3. *AI/ML Platform & Inference* (Model Serving, Feature Stores, Distributed Training, Evaluation & Observability)
  4. *Security & Compliance* (Zero-Trust Identity, Secret Management, Vulnerability Scanning, Cloud SIEM)
  5. *Core Backend & Microservices* (Service Mesh & RPC, Event-Driven Messaging, Distributed Transactions, Rate Limiting)
  6. *Cloud Infrastructure & Reliability* (Kubernetes Orchestration, Multi-Region Failover, Auto-scaling, Chaos Engineering)
  7. *Frontend & Design Systems* (Micro-frontends, Component Libraries, Real-time Collaboration, Web Performance)
  8. *Search & Discovery* (Vector Search Indexing, Semantic Rankers, Knowledge Graph Traversal, Query Understanding)
- Maps domain/sub-domains to realistic tech stacks, problem templates, and solution patterns.

#### [NEW] [seed_generator.py](file:///Users/anchormath/Code/self/smriti/src/smriti/generators/seed_generator.py)
- Generates realistic statistical distributions:
  - Reach: Log-normal distribution ($e^{\mu + \sigma Z}$) reflecting enterprise user scale (e.g. median 250 internal devs or 50k end-users, long tail up to 5M).
  - Effort: Right-skewed distribution (Gamma/Log-normal, min 1 week, median 8 weeks, tail up to 52 person-weeks) with consistent breakdown sum.
  - Impact & Confidence: Sampled according to standard enterprise portfolio tier distributions.
- Graph Link Synthesizer:
  - Generates cross-project dependency DAGs (e.g., Infrastructure & Backend projects become upstream dependencies for Frontend & ML applications).
  - Ensures acyclic validity and bidirectional consistency (`upstream` $\leftrightarrow$ `downstream`).

#### [NEW] [llm_enricher.py](file:///Users/anchormath/Code/self/smriti/src/smriti/generators/llm_enricher.py)
- Narrative enrichment engine:
  - Supports Gemini API (`google-genai` / `httpx`), Ollama API, and an offline deterministic generator with $\ge 200$ parameterizable templates.
  - Produces detailed, highly specific engineering problem statements, architectural solutions, and target user descriptions.

---

### Component 3: Multi-Resolution Embedding Pipeline (`smriti/embeddings/`)

#### [NEW] [mrl_pipeline.py](file:///Users/anchormath/Code/self/smriti/src/smriti/embeddings/mrl_pipeline.py)
- Embedding generator utilizing MRL principles:
  - Encodes unified project texts into 768-dimensional vectors.
  - Extracts prefix slices:
    - $v_0 = v[:64]$ with $L_2$ normalization: $v_0 / \|v_0\|_2$
    - $v_1 = v[:128]$ with $L_2$ normalization: $v_1 / \|v_1\|_2$
    - $v_2 = v[:768]$ with $L_2$ normalization: $v_2 / \|v_2\|_2$
  - Batching, device auto-detection (MPS/CUDA/CPU), and verification that $|\|v\|_2 - 1.0| < 10^{-5}$.

---

### Component 4: Dataset Packaging & CLI (`smriti/data/` & `scripts/`)

#### [NEW] [dataset_builder.py](file:///Users/anchormath/Code/self/smriti/src/smriti/data/dataset_builder.py)
- Orchestrates dataset construction:
  - Generates $\ge 3,500$ memory nodes (for indexing) and $500$ held-out query nodes (for testing routing and imputation in Sprints 2 & 3).
  - Exports to Parquet (`data/ekg_projects_v1.parquet`) and SQLite (`data/ekg_projects_v1.db`).
  - Exports test query set (`data/ekg_queries_test.json` & `data/ekg_queries_test.parquet`) with masked fields for associative completion benchmarking.

#### [NEW] [generate_dataset.py](file:///Users/anchormath/Code/self/smriti/scripts/generate_dataset.py)
- CLI script to execute generation, embedding, and storage with progress bars and statistics summary.

#### [NEW] [validate_dataset.py](file:///Users/anchormath/Code/self/smriti/scripts/validate_dataset.py)
- Automated verification script checking all Acceptance Criteria (Dataset Size, Schema Validity, MRL Unit-Norm Integrity, RICE Distributions).

---

## Verification Plan

### Automated Tests
Run pytest test suite:
```bash
./.venv/bin/pytest tests/ -v
```

Run dataset generation and validation CLI:
```bash
./.venv/bin/python scripts/generate_dataset.py --num-items 3500 --num-queries 500
./.venv/bin/python scripts/validate_dataset.py
```

### Acceptance Criteria Checklist
1. **Dataset Size:** $\ge 3,000$ validated project nodes across 8 domains and $\ge 24$ sub-domains.
2. **Schema Validity:** 100% of records pass Pydantic schema validation.
3. **MRL Integrity:** Vector slices $d_0=64, d_1=128, d_2=768$ strictly unit-normalized ($|\|v\|_2 - 1.0| < 10^{-5}$).
4. **Data Diversity:** RICE scores follow realistic enterprise distribution (Log-normal reach, right-skewed effort).
