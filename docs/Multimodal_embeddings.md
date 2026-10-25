# Multimodal Matryoshka Associative Retrieval Test Harness (EmbeddingGemma 2 & Smriti HAM-P/R)

## Overview & Background

Modern enterprise software systems are inherently multimodal: engineers communicate architectures via **diagrams** (C4 block diagrams, sequence charts, dataflow topologies), explain design decisions and incidents via **audio tracks** (design reviews, standup recordings, voice memos), and implement systems via **code and specifications** (ADRs, RFCs, APIs).

Historically, vector search across these assets required disjoint embedding spaces or lossy multi-stage pipelines (e.g., OCR-ing diagrams or transcribing audio to text before indexing). With **EmbeddingGemma 2**, Google introduced a unified multimodal model (740M params, built on Gemma 4) that maps **text, images, and audio directly into a shared 768-dimensional vector space** with native support for **Matryoshka Representation Learning (MRL)**.

This plan details the design and implementation of a test harness and prototype in **Smriti** to validate:
1. Ingestion of multimodal architectural artifacts (diagrams, audio walkthroughs, and specs).
2. Generation of unified 768-dim MRL embeddings with prefix slicing ($d_0=64 \to d_1=128 \to d_2=768$).
3. Partitioned tree indexing and multi-probe routing in **HAM-P (`smriti-core`)**.
4. Cross-modal associative pattern completion and Resolution Dial ($R \in [0.0, 1.0]$) recall in **HAM-R**.

---

## Architecture: Cross-Modal Associative Memory Flow

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       Multimodal System Artifacts                      │
   │  ┌────────────────────┐  ┌───────────────────┐  ┌───────────────────┐  │
   │  │ Architectural      │  │ Audio Track       │  │ Technical Spec    │  │
   │  │ Diagram (PNG/SVG)  │  │ Walkthrough (WAV) │  │ ADR / RFC (MD)    │  │
   │  └─────────┬──────────┘  └─────────┬─────────┘  └─────────┬─────────┘  │
   └────────────┼───────────────────────┼──────────────────────┼────────────┘
                │                       │                      │
                ▼                       ▼                      ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │             Multimodal MRL Pipeline (EmbeddingGemma 2)                 │
   │   - Unified 768-dim projection for Image, Audio & Text                 │
   │   - Prefix Slicing & L2 Normalization:                                 │
   │       d0 = 64 (Root Router)                                            │
   │       d1 = 128 (Subdomain Routing Node)                                │
   │       d2 = 768 (Hopfield Memory Buffer)                                │
   │   - Dual-Mode: Live EmbeddingGemma-2 Model + Deterministic Offline Sim │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                Smriti Core Engine (Rust / PyO3)                        │
   │                                                                        │
   │   Layer 0: Root Prefix Router (d0 = 64)                                │
   │            Filters coarse system domain (e.g., Platform vs App Layer)  │
   │                                   │                                    │
   │   Layer 1: Subdomain Routing Node (d1 = 128)                           │
   │            Routes to Subsystem Cluster (e.g., Event Ingestion, Cache)  │
   │                                   │                                    │
   │   Layer 2: Continuous Hopfield Leaf Pool (d2 = 768)                    │
   │            Associative Pattern Completion & Resolution Dial R:         │
   │            x_out = Ξ_leaf · softmax(β(R) · Ξ_leafᵀ · x_query)          │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                     Cross-Modal Recall & Retrieval                     │
   │   • Audio Query   ──► Recalls Architectural Diagram & Component Spec   │
   │   • Diagram Query ──► Recalls Audio Narration & Service Dependencies   │
   │   • Text Query    ──► Pinpoints Diagram Region & Spoken Timestamp      │
   │   • Dial R=0.2    ──► Broad Subsystem Cluster Blending                 │
   │   • Dial R=0.9    ──► Isolated Exact Artifact Pinpoint                 │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## User Review Required

> [!IMPORTANT]
> **Embedding Model Execution Strategy:**
> `google/embeddinggemma-2` requires downloading ~1.5 GB model weights and requires `torch`, `transformers`, and multimodal processors. In sandboxed or offline environments without Hugging Face network access, executing `SentenceTransformer("google/embeddinggemma-2")` will fail with network errors.
> 
> **Proposed Solution:** Implement `MultimodalMRLEmbeddingPipeline` with dual execution paths:
> 1. **Live Model Mode**: When `google/embeddinggemma-2` is available and dependencies are satisfied, runs live multimodal inference across image, audio, and text inputs.
> 2. **Deterministic Offline Simulation Mode**: Generates deterministic 768-dim MRL embeddings that mirror the cross-modal joint manifold properties (using semantic domain projection, modality correlation factors, and noise shaping). This guarantees reproducible, zero-network unit testing and CI verification while maintaining exact mathematical fidelity for HAM-P and HAM-R.

---

## Proposed Changes

Grouped by component layer:

### 1. Multimodal Models & Schema

Extend Smriti's data models to represent multimodal artifacts and cross-artifact relationships.

#### [NEW] [multimodal_schema.py](file:///Users/anchormath/Code/self/smriti/src/smriti/models/multimodal_schema.py)
- `ModalityType`: Enum (`TEXT`, `IMAGE_DIAGRAM`, `AUDIO_WALKTHROUGH`, `CODE_SPEC`).
- `SystemSubsystem`: Categorical classification (e.g., `DistributedCache`, `EventGateway`, `IdentityAuth`, `StorageEngine`).
- `MultimodalArtifact`: Schema containing:
  - `id`: Unique artifact identifier (e.g., `art_diag_cache_topology_01`, `art_audio_cache_invalidation_01`).
  - `subsystem`: Target system component.
  - `modality`: `ModalityType`.
  - `title`, `description`, `tags`: Descriptive context.
  - `media_uri`: Path to image/audio/spec file or virtual asset.
  - `temporal_spatial_anchor`: Spatial bounding box (for diagram elements) or time interval `[start_sec, end_sec]` (for audio).
  - `linked_artifact_ids`: Explicit ground-truth cross-modal linkages (for recall validation).
  - `embeddings`: `MRLEmbeddings` ($d_0=64, d_1=128, d_2=768$).

---

### 2. Multimodal MRL Embedding Pipeline

#### [NEW] [multimodal_pipeline.py](file:///Users/anchormath/Code/self/smriti/src/smriti/embeddings/multimodal_pipeline.py)
- Wraps `SentenceTransformer("google/embeddinggemma-2")` with fallback to deterministic multimodal generation.
- Supports heterogeneous input batches:
  - Text prompts (e.g., `{"text": "Architecture of distributed caching tier"}`)
  - Images / Diagrams (file path or synthetic image spec)
  - Audio files / waveforms (file path or synthetic audio waveform)
- Applies Matryoshka prefix slicing with independent $L_2$ normalization:
  - $d_0 = 64$
  - $d_1 = 128$
  - $d_2 = 768$
- Preserves cross-modal semantic proximity between related artifacts of the same subsystem.

---

### 3. Cross-Modal Test Suite & Benchmark Harness

#### [NEW] [test_multimodal_associative_recall.py](file:///Users/anchormath/Code/self/smriti/tests/test_multimodal_associative_recall.py)
Comprehensive pytest suite validating:
1. **Schema Validation**: Correct instantiation, normalization checks ($||v||_2 = 1.0$), and validation rules.
2. **Multimodal MRL Generation**: Verification of dimensionality, prefix containment, and unit-norm consistency.
3. **Cross-Modal Retrieval Precision**:
   - Audio query $\to$ Diagram recall (Top-1 and Top-5 recall).
   - Diagram query $\to$ Audio walkthrough recall.
   - Text query $\to$ Multimodal artifact pair recall.
4. **HAM-P Multi-Resolution Prefix Slicing Efficiency**:
   - Verifies that routing at $d_0=64$ and $d_1=128$ prunes $\ge 90\%$ of candidate leaves before Hopfield leaf evaluation at $d_2=768$.
5. **Resolution Dial Sweeps**:
   - Validates that sweeping $R \in [0.0, 1.0]$ transitions from broad subsystem blend (low energy, high entropy) to sharp artifact pinning.

#### [NEW] [eval_multimodal_smriti.py](file:///Users/anchormath/Code/self/smriti/scripts/eval_multimodal_smriti.py)
Standalone CLI demonstration and benchmark script:
- Generates a sample Enterprise System Graph with 4 distinct subsystems:
  - **Subsystem A:** Distributed Event Gateway (Kafka / gRPC / Ingest Diagram + Voice Note + Ingestion Spec)
  - **Subsystem B:** Distributed Tiered Cache (Redis Cluster / Consistent Hash Diagram + Incident Audio + Invalidation Spec)
  - **Subsystem C:** Zero-Trust Identity & Auth (OAuth2 / mTLS Diagram + Security Review Audio + Auth Spec)
  - **Subsystem D:** Columnar Analytics Store (Parquet / Object Storage Diagram + Performance Audio + Query Spec)
- Indexes all multimodal artifacts into `smriti-core`'s `HAMPTreeWrapper`.
- Executes cross-modal queries:
  - Spoken query: *"We need to debug queue backpressure during morning spike traffic"* $\to$ finds Event Gateway diagram.
  - Diagram query: Cache topology diagram $\to$ finds cache invalidation audio memo and spec.
- Displays rich tables of Top-$K$ recall, cosine similarity scores, and Hopfield energy transitions.

---

## Verification Plan

### Automated Tests
Run the test suite using the workspace Python virtual environment:
```bash
./.venv/bin/pytest tests/test_multimodal_associative_recall.py -v
```
Run the full test suite to guarantee zero regression across existing HAM-P / HAM-R tests:
```bash
./.venv/bin/pytest
```

### Manual / Benchmark Verification
Run the interactive CLI evaluation:
```bash
./.venv/bin/python scripts/eval_multimodal_smriti.py
```
Check that:
- Top-1 and Top-5 cross-modal recall reach $\ge 90\%$.
- Resolution Dial monotonic energy landscape is validated across multimodal clusters.
- HAM-P tree routing correctly filters irrelevant subsystems at $d_0=64$ and $d_1=128$.
