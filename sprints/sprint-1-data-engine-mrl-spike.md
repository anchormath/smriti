# Sprint 1: Data Engine & MRL Embeddings Spike

**Timeline:** Weeks 1–2  
**Epic:** Testbed Data Engine & Multi-Resolution Vector Pipeline  
**Goal:** Define the Enterprise Project Node schema, generate a rich synthetic dataset of 2,000–5,000 hierarchical project nodes with ground-truth RICE metrics, and generate multi-resolution MRL embeddings.

---

## 1. Objectives & Scope

- [x] **Schema Definition:** Standardize a typed schema capturing continuous textual descriptions, structured RICE factors, and relational graph links.
- [x] **Synthetic Generation Pipeline:** Combine rule-based/Faker generators with cost-effective LLMs (e.g., Gemini Flash or local Ollama) to produce realistic enterprise project data.
- [x] **MRL Embedding Generation:** Generate prefix-sliced embeddings ($d_0=64$, $d_1=128$, $d_2=768$) with proper $L_2$ normalization using an open-source MRL encoder (`nomic-embed-text-v1.5`).
- [x] **Dataset Artifacts:** Store dataset in columnar format (Parquet / SQLite) split into indexable memory items and held-out test queries.

---

## 2. Project Schema Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EnterpriseProjectNode",
  "type": "object",
  "properties": {
    "id": { "type": "string", "example": "proj_infra_2026_042" },
    "taxonomy": {
      "type": "object",
      "properties": {
        "domain": { "type": "string", "example": "Developer Productivity & CI/CD" },
        "sub_domain": { "type": "string", "example": "Build Caching & Artifact Distribution" },
        "epic": { "type": "string", "example": "Remote Execution Engine" }
      },
      "required": ["domain", "sub_domain", "epic"]
    },
    "semantic_context": {
      "type": "object",
      "properties": {
        "title": { "type": "string" },
        "problem_statement": { "type": "string" },
        "solution_proposal": { "type": "string" },
        "tech_stack": { "type": "array", "items": { "type": "string" } },
        "target_users": { "type": "string" }
      },
      "required": ["title", "problem_statement", "solution_proposal", "tech_stack"]
    },
    "rice_factors": {
      "type": "object",
      "properties": {
        "reach": {
          "type": "object",
          "properties": {
            "value": { "type": "number", "description": "Users / systems impacted per quarter" },
            "unit": { "type": "string" }
          },
          "required": ["value"]
        },
        "impact": {
          "type": "object",
          "properties": {
            "score": { "type": "number", "enum": [0.25, 0.5, 1.0, 2.0, 3.0] },
            "tier": { "type": "string", "enum": ["Minimal", "Low", "Medium", "High", "Massive"] },
            "rationale": { "type": "string" }
          },
          "required": ["score", "tier"]
        },
        "confidence": {
          "type": "object",
          "properties": {
            "percentage": { "type": "number", "enum": [0.5, 0.8, 1.0] },
            "tier": { "type": "string", "enum": ["Low / Moonshot", "Medium", "High"] },
            "evidence": { "type": "string" }
          },
          "required": ["percentage", "tier"]
        },
        "effort": {
          "type": "object",
          "properties": {
            "person_weeks": { "type": "number" },
            "breakdown": {
              "type": "object",
              "properties": {
                "frontend": { "type": "number" },
                "backend": { "type": "number" },
                "infra_devops": { "type": "number" },
                "qa_testing": { "type": "number" }
              }
            }
          },
          "required": ["person_weeks"]
        },
        "rice_score": { "type": "number", "description": "(Reach * Impact * Confidence) / Effort" }
      },
      "required": ["reach", "impact", "confidence", "effort", "rice_score"]
    },
    "graph_relations": {
      "type": "object",
      "properties": {
        "upstream_dependencies": { "type": "array", "items": { "type": "string" } },
        "downstream_impacted": { "type": "array", "items": { "type": "string" } },
        "tags": { "type": "array", "items": { "type": "string" } }
      }
    }
  },
  "required": ["id", "taxonomy", "semantic_context", "rice_factors"]
}
```

---

## 3. Synthetic Generation Architecture

```
┌────────────────────────────────────────────────────────┐
│ 1. Taxonomy & Seed Generator (Python + Faker)          │
│    - Defines 8 Top-Level Domains                       │
│    - Defines 3-5 Sub-domains per Domain                │
│    - Samples Reach (Log-normal), Effort, Impact        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. LLM Narrative Enrichment (Gemini Flash / Ollama)    │
│    - System Prompt: Generates realistic tech specs     │
│    - Inputs: Domain, Sub-domain, Tech stack seed       │
│    - Outputs: Structured JSON (Problem & Solution)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Graph Link Synthesizer                              │
│    - Builds cross-project DAG dependencies             │
│    - Adds shared service dependencies                  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. MRL Embedding Generation Engine                     │
│    - Model: `nomic-ai/nomic-embed-text-v1.5`           │
│    - Unified Text: Title + Problem + Solution + Tech   │
│    - Prefix Slices: d0=64, d1=128, d2=768 (L2 norm)    │
└────────────────────────────────────────────────────────┘
```

### Prompt Strategy for LLM Generation
- Use JSON Mode / Structured Output to enforce schema compliance.
- Vary complexity: 70% standard feature/infra initiatives, 20% major architectural migrations, 10% high-risk R&D prototypes.

---

## 4. MRL Embedding Pipeline Details

### Unified Text Serialization Format
To ensure consistent semantic representations across project nodes:
```text
search_document: [Domain: Developer Productivity & CI/CD] [Sub-domain: Build Caching]
Title: Distributed Bazel Remote Build Cache with S3 Acceleration
Problem: Local build times exceed 45 minutes for core monorepo services, degrading deployment cadence.
Solution: Deploy a distributed gRPC caching proxy backed by NVMe caches and S3 tiered storage.
Tech Stack: Rust, gRPC, AWS S3, Bazel, Redis
Target Audience: All backend and platform engineers (~450 devs)
```

### Slicing & Normalization Protocol
For each vector $v \in \mathbb{R}^{768}$:
```python
import numpy as np

def extract_mrl_slices(v_full: np.ndarray):
    # Slice 0: Root / Category Routing
    v_0 = v_full[:64]
    v_0 = v_0 / (np.linalg.norm(v_0) + 1e-12)
    
    # Slice 1: Subcategory / Cluster Routing
    v_1 = v_full[:128]
    v_1 = v_1 / (np.linalg.norm(v_1) + 1e-12)
    
    # Slice 2: Full Hopfield Memory Vector
    v_2 = v_full[:768]
    v_2 = v_2 / (np.linalg.norm(v_2) + 1e-12)
    
    return v_0, v_1, v_2
```

---

## 5. Sprint Tasks & Deliverables

| Task ID | Task Description | Est. Days | Owner / Tooling | Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **S1-T1** | Implement Pydantic models for `EnterpriseProjectNode` | 1 | Python (`pydantic`) | `models/project_schema.py` |
| **S1-T2** | Build rule-based generator for taxonomy, RICE, and graph links | 2 | Python (`faker`, `numpy`) | `generators/seed_generator.py` |
| **S1-T3** | Build LLM batch enrichment pipeline with rate limiting | 3 | Gemini Flash / Ollama API | `generators/llm_enricher.py` |
| **S1-T4** | Build MRL embedding pipeline (`nomic-embed-text-v1.5`) | 2 | PyTorch / HuggingFace | `embeddings/mrl_pipeline.py` |
| **S1-T5** | Package and validate dataset (4,000 index + 500 test queries) | 2 | Parquet / SQLite | `data/ekg_projects_v1.parquet` |

---

## 6. Acceptance Criteria

1. **Dataset Size:** Minimum 3,000 validated project nodes across 8 domains and $\ge 24$ sub-domains.
2. **Schema Validity:** 100% of records pass Pydantic schema validation without missing fields.
3. **MRL Integrity:** Vector slices at $d_0, d_1, d_2$ are verified to be strictly unit-normalized ($|\|v\|_2 - 1.0| < 10^{-5}$).
4. **Data Diversity:** RICE scores follow realistic enterprise distribution (Reach log-normal, Effort right-skewed).


# Sprint 1 Review

I ran a comprehensive inspection, validation, and empirical sanity check on [`data/ekg_projects_v1.parquet`](file:///Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet) and the Sprint 1 test suite.


### **Validation & Tire-Kicking Report**

#### 1. Dataset Integrity & Schema Sanity
* **Volume & Distribution:** **3,500 enterprise project records** evenly distributed across **8 Top-Level Domains** and **32 Sub-domains** (~109 items per sub-domain).
* **Schema Conformance:** 100% of records conform to the Pydantic schema (zero nulls or invalid values in core fields).
* **RICE Metric Integrity:** 
  * Reach values follow realistic log-normal distributions ($20$ to $6.89\text{M}$ users/systems).
  * Person-week efforts range realistically ($1.0$ to $40.5\text{ pw}$) with valid discipline breakdowns (Frontend, Backend, Infra, QA).
  * Formula verification: **Zero discrepancy** between stored `rice_score` and $(R \times I \times C)/E$.
* **Test Suite Status:** All **11 unit tests** in `tests/` pass in `3.12s`.

---

#### 2. MRL Embeddings & Mathematical Properties
* **Embedding Columns:** `mrl_d0_64`, `mrl_d1_128`, `mrl_d2_768` (float64).
* **$L_2$ Unit Normalization:** Verified strictly across all 3,500 records (max deviation $< 1.2 \times 10^{-7}$).
* **Prefix Slicing Alignment:** Prefix slices $x[:64]$ and $x[:128]$ match the full vector $x[:768]$ prefix directions identically.

---

#### 3. Empirical Proof-of-Concept: Hierarchical Routing Performance
To test the core theory of HAM-P before writing the Rust engine, we evaluated prototype routing on these 3,500 vectors:

| Routing Stage | Vector Dimension | Routing Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **Layer 0 (Domain Prototypes)** | **$d_0 = 64$** | **$98.54\%$** | 64 dimensions successfully isolate the correct top-level domain branch. |
| **Hierarchical ($L_0 \to L_1$)** | **$d_0=64 \to d_1=128$** | **$98.54\%$** | Subcategory routing isolates the destination leaf pool with $< 1.5\%$ error. |
| **Leaf Pool Capacity** | **$d_2 = 768$** | **$\approx 109$ items/leaf** | Ideal scale for continuous Hopfield associative recall and testing the $M_{\max} = 100$ 2-means split trigger! |

---

### **Ready for Sprint 2: Core Rust Engine (`smriti-core`)**

With the data engine validated and stored in [`data/ekg_projects_v1.parquet`](file:///Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet), we can now proceed to [Sprint 2](file:///Users/anchormath/Code/self/smriti/sprints/sprint-2-ham-p-tree-routing.md):

1. Initialize the Rust crate `smriti-core` (column-major contiguous memory matrices).
2. Implement SIMD dot products and `rayon` multithreaded tree routing.
3. Implement online streaming insertions and spherical 2-means dynamic leaf node splitting.
4. Expose the Rust engine to Python via PyO3 / Maturin bindings.


------------
## Details of validation
```bash

Ran command: `python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')
print('=== SHAPE & COLUMNS ===')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\n=== DTYPES & NULLS ===')
print(df.dtypes)
print('Null counts:\n', df.isnull().sum())
print('\n=== HEAD (1 row) ===')
print(df.iloc[0].to_dict())
"`
Ran command: `python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')

print('================================================================')
print(f'TOTAL ROWS: {len(df)}')
print('COLUMNS:', df.columns.tolist())
print('================================================================\n')

# Check taxonomy breakdown
print('--- TAXONOMY BREAKDOWN ---')
if 'domain' in df.columns:
    print('Unique Domains:', df['domain'].nunique())
    print(df['domain'].value_counts())
    print('\nUnique Sub-domains:', df['sub_domain'].nunique())
elif 'taxonomy' in df.columns:
    print('Taxonomy sample:', df['taxonomy'].iloc[0])

# Check RICE metrics
print('\n--- RICE METRIC DISTRIBUTIONS ---')
for col in ['reach', 'impact', 'confidence', 'effort', 'rice_score']:
    if col in df.columns:
        print(f'{col}: min={df[col].min():.2f}, median={df[col].median():.2f}, mean={df[col].mean():.2f}, max={df[col].max():.2f}, nulls={df[col].isnull().sum()}')

# Check Embeddings
print('\n--- EMBEDDINGS CHECK ---')
emb_cols = [c for c in df.columns if 'embed' in c.lower() or 'vec' in c.lower()]
print('Embedding columns found:', emb_cols)

for c in emb_cols:
    first_vec = np.array(df[c].iloc[0])
    print(f'Column [{c}]: shape={first_vec.shape}, dtype={first_vec.dtype}')
    
    # Check norms across all rows
    all_vecs = np.vstack(df[c].values)
    norms_full = np.linalg.norm(all_vecs, axis=1)
    print(f'  Full Norm: min={norms_full.min():.6f}, max={norms_full.max():.6f}, mean={norms_full.mean():.6f}')
    
    if first_vec.shape[0] >= 128:
        norms_64 = np.linalg.norm(all_vecs[:, :64], axis=1)
        norms_128 = np.linalg.norm(all_vecs[:, :128], axis=1)
        print(f'  Slice[:64] Norm (raw): min={norms_64.min():.6f}, max={norms_64.max():.6f}, mean={norms_64.mean():.6f}')
        print(f'  Slice[:128] Norm (raw): min={norms_128.min():.6f}, max={norms_128.max():.6f}, mean={norms_128.mean():.6f}')
"`
Ran command: `python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')

print('=== 1. EMBEDDINGS INTEGRITY CHECK ===')
for col in ['mrl_d0_64', 'mrl_d1_128', 'mrl_d2_768']:
    vecs = np.vstack(df[col].values)
    norms = np.linalg.norm(vecs, axis=1)
    print(f'Column [{col}]: shape={vecs.shape}, dtype={vecs.dtype}')
    print(f'  L2 Norms: min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f}')
    print(f'  Any NaN/Inf: {np.isnan(vecs).any() or np.isinf(vecs).any()}')

print('\n=== 2. MRL PREFIX CONSISTENCY CHECK ===')
v0 = np.vstack(df['mrl_d0_64'].values)
v1 = np.vstack(df['mrl_d1_128'].values)
v2 = np.vstack(df['mrl_d2_768'].values)

# Check if mrl_d0_64 is proportional to v2[:, :64]
norm_v2_64 = v2[:, :64] / np.linalg.norm(v2[:, :64], axis=1, keepdims=True)
diff_0 = np.abs(v0 - norm_v2_64).max()
print(f'Max diff between mrl_d0_64 and norm(mrl_d2_768[:64]): {diff_0:.8f}')

norm_v2_128 = v2[:, :128] / np.linalg.norm(v2[:, :128], axis=1, keepdims=True)
diff_1 = np.abs(v1 - norm_v2_128).max()
print(f'Max diff between mrl_d1_128 and norm(mrl_d2_768[:128]): {diff_1:.8f}')

print('\n=== 3. RICE FACTORS SANITY CHECK ===')
rice_cols = ['reach_value', 'impact_score', 'confidence_percentage', 'effort_person_weeks', 'rice_score', 'effort_frontend', 'effort_backend', 'effort_infra', 'effort_qa']
for c in rice_cols:
    s = df[c]
    print(f'{c:22s} | min={s.min():10.2f} | median={s.median():10.2f} | max={s.max():12.2f} | nulls={s.isnull().sum()}')

# Verify RICE formula: (reach * impact * confidence) / effort
expected_rice = (df['reach_value'] * df['impact_score'] * df['confidence_percentage']) / df['effort_person_weeks']
rice_diff = np.abs(df['rice_score'] - expected_rice).max()
print(f'\nMax discrepancy in RICE formula calculation: {rice_diff:.6f}')

print('\n=== 4. CLUSTER SEPARATION AT d0 (64) & d1 (128) ===')
domains = df['domain'].unique()
domain_centroids_d0 = {}
for d in domains:
    idx = df['domain'] == d
    domain_centroids_d0[d] = v0[idx].mean(axis=0)
    domain_centroids_d0[d] /= np.linalg.norm(domain_centroids_d0[d])

# Measure intra-domain vs inter-domain similarity
intra_sims = []
inter_sims = []
for i, d in enumerate(domains):
    idx = (df['domain'] == d).values
    c = domain_centroids_d0[d]
    intra_sims.extend((v0[idx] @ c).tolist())
    other_idx = (~idx)
    inter_sims.extend((v0[other_idx] @ c).tolist())

print(f'd0 (64-dim) Intra-domain similarity to centroid: mean={np.mean(intra_sims):.4f}, std={np.std(intra_sims):.4f}')
print(f'd0 (64-dim) Inter-domain similarity to centroid: mean={np.mean(inter_sims):.4f}, std={np.std(inter_sims):.4f}')
"`
Ran command: `python3 -c "
import pandas as pd

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')

for i in range(2):
    row = df.iloc[i]
    print(f'=== PROJECT {i+1}: {row[\"id\"]} ===')
    print(f'Domain: {row[\"domain\"]} -> {row[\"sub_domain\"]} -> {row[\"epic\"]}')
    print(f'Title: {row[\"title\"]}')
    print(f'Problem: {row[\"problem_statement\"]}')
    print(f'Solution: {row[\"solution_proposal\"]}')
    print(f'Tech Stack: {row[\"tech_stack\"]}')
    print(f'RICE: Reach={row[\"reach_value\"]} ({row[\"reach_unit\"]}), Impact={row[\"impact_score\"]} ({row[\"impact_tier\"]}), Conf={row[\"confidence_percentage\"]}, Effort={row[\"effort_person_weeks\"]} pw -> Score={row[\"rice_score\"]:.2f}')
    print(f'Graph: Upstream={row[\"upstream_dependencies\"]}, Downstream={row[\"downstream_impacted\"]}, Tags={row[\"tags\"]}')
    print(f'Unified Text Snippet:\n{row[\"unified_text\"][:250]}...\n')
"`
Listed directory smriti
Listed directory scripts
Listed directory src
Listed directory smriti
Listed directory tests
Ran command: `python3 scripts/validate_dataset.py`
Ran command: `pytest`
Ran command: `PYTHONPATH=src pytest`
Ran command: `.venv/bin/pytest`
Ran command: `.venv/bin/python -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')

v0 = np.vstack(df['mrl_d0_64'].values)
v1 = np.vstack(df['mrl_d1_128'].values)
v2 = np.vstack(df['mrl_d2_768'].values)

domains = df['domain'].unique()
sub_domains = df['sub_domain'].unique()

print(f'Total items: {len(df)}')
print(f'Domains: {len(domains)}, Sub-domains: {len(sub_domains)}')

# 1. Compute Layer 0 (Domain Prototypes at d0=64)
domain_prototypes = {}
for d in domains:
    idx = (df['domain'] == d).values
    proto = v0[idx].mean(axis=0)
    proto /= np.linalg.norm(proto)
    domain_prototypes[d] = proto

proto_mat_d0 = np.array([domain_prototypes[d] for d in domains]) # (8, 64)

# Test Layer 0 Routing Accuracy
d0_sims = v0 @ proto_mat_d0.T # (3500, 8)
best_domain_idx = np.argmax(d0_sims, axis=1)
predicted_domains = [domains[i] for i in best_domain_idx]
l0_accuracy = np.mean([p == actual for p, actual in zip(predicted_domains, df['domain'])])
print(f'Layer 0 (d0=64) Domain Routing Accuracy: {l0_accuracy*100:.2f}%')

# 2. Compute Layer 1 (Sub-domain Prototypes at d1=128)
sub_domain_prototypes = {}
for sd in sub_domains:
    idx = (df['sub_domain'] == sd).values
    proto = v1[idx].mean(axis=0)
    proto /= np.linalg.norm(proto)
    sub_domain_prototypes[sd] = proto

# Test Hierarchical Routing: L0 -> L1
hierarchical_correct = 0
for i, row in df.iterrows():
    # Step 1: L0 routing
    pred_domain = predicted_domains[i]
    if pred_domain != row['domain']:
        continue
    
    # Step 2: L1 routing within selected domain
    domain_sds = df[df['domain'] == pred_domain]['sub_domain'].unique()
    sd_proto_mat = np.array([sub_domain_prototypes[sd] for sd in domain_sds])
    sd_sims = v1[i] @ sd_proto_mat.T
    pred_sd = domain_sds[np.argmax(sd_sims)]
    
    if pred_sd == row['sub_domain']:
        hierarchical_correct += 1

print(f'Hierarchical L0 (d0=64) -> L1 (d1=128) Sub-domain Routing Accuracy: {hierarchical_correct / len(df) * 100:.2f}%')

# 3. Memory Matrix Leaf Pool Size distribution
print('\nLeaf Pool item distribution if grouped by Sub-domain:')
sd_counts = df['sub_domain'].value_counts()
print(f'Min items/sub-domain: {sd_counts.min()}, Max: {sd_counts.max()}, Mean: {sd_counts.mean():.1f}')
"`
Ran command: `python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/anchormath/Code/self/smriti/data/ekg_projects_v1.parquet')

v0 = np.vstack(df['mrl_d0_64'].values)
v1 = np.vstack(df['mrl_d1_128'].values)
v2 = np.vstack(df['mrl_d2_768'].values)

domains = df['domain'].unique()
sub_domains = df['sub_domain'].unique()

print(f'Total items: {len(df)}')
print(f'Domains: {len(domains)}, Sub-domains: {len(sub_domains)}')

# 1. Compute Layer 0 (Domain Prototypes at d0=64)
domain_prototypes = {}
for d in domains:
    idx = (df['domain'] == d).values
    proto = v0[idx].mean(axis=0)
    proto /= np.linalg.norm(proto)
    domain_prototypes[d] = proto

proto_mat_d0 = np.array([domain_prototypes[d] for d in domains]) # (8, 64)

# Test Layer 0 Routing Accuracy
d0_sims = v0 @ proto_mat_d0.T # (3500, 8)
best_domain_idx = np.argmax(d0_sims, axis=1)
predicted_domains = [domains[i] for i in best_domain_idx]
l0_accuracy = np.mean([p == actual for p, actual in zip(predicted_domains, df['domain'])])
print(f'Layer 0 (d0=64) Domain Routing Accuracy: {l0_accuracy*100:.2f}%')

# 2. Compute Layer 1 (Sub-domain Prototypes at d1=128)
sub_domain_prototypes = {}
for sd in sub_domains:
    idx = (df['sub_domain'] == sd).values
    proto = v1[idx].mean(axis=0)
    proto /= np.linalg.norm(proto)
    sub_domain_prototypes[sd] = proto

# Test Hierarchical Routing: L0 -> L1
hierarchical_correct = 0
for i, row in df.iterrows():
    pred_domain = predicted_domains[i]
    if pred_domain != row['domain']:
        continue
    
    domain_sds = df[df['domain'] == pred_domain]['sub_domain'].unique()
    sd_proto_mat = np.array([sub_domain_prototypes[sd] for sd in domain_sds])
    sd_sims = v1[i] @ sd_proto_mat.T
    pred_sd = domain_sds[np.argmax(sd_sims)]
    
    if pred_sd == row['sub_domain']:
        hierarchical_correct += 1

print(f'Hierarchical L0 (d0=64) -> L1 (d1=128) Sub-domain Routing Accuracy: {hierarchical_correct / len(df) * 100:.2f}%')

print('\nLeaf Pool item distribution if grouped by Sub-domain:')
sd_counts = df['sub_domain'].value_counts()
print(f'Min items/sub-domain: {sd_counts.min()}, Max: {sd_counts.max()}, Mean: {sd_counts.mean():.1f}')
"`
```
