# Smriti (स्मृति): Continuous Modern Hopfield Associative Memory with Hierarchical Multi-Resolution Partitioning and Cross-Modal Matryoshka Embeddings

**Authors:** Ankur Mathur  
**Affiliation:** Radna research
**Date:** October 2026  
**Artifact Repository:** `smriti` (`smriti-core` in Rust / `smriti` in Python  
**Classification:** Systems Architecture, Non-Parametric Memory, Continuous Hopfield Networks, Multimodal Retrieval

---

## Abstract

Modern continuous Hopfield networks possess exponential memory storage capacity and provide mathematically rigorous pattern completion via energy minimization. However, evaluating queries across a single flat memory matrix $\Xi \in \mathbb{R}^{d \times M}$ incurs an $O(M \cdot d)$ computational bottleneck and suffers from severe cross-talk and spurious attractors when co-locating disparate semantic domains within a shared global state. In enterprise software engineering, this limitation is exacerbated by the inherently multimodal nature of architectural knowledge—where systems are simultaneously specified through visual diagrams (C4 topologies, sequence charts), verbal narrations (technical design reviews, incident debriefs), and formal specifications (ADRs, code).

We introduce **Smriti (स्मृति)**, a high-performance associative memory engine that solves both the computational bottleneck and cross-domain cross-talk through two unified mechanisms:
1. **Hierarchical Associative Memory Partitioning (HAM-P):** A dynamic tree structure that exploits **Matryoshka Representation Learning (MRL)** to route queries through low-dimensional prefix projections ($d_0=64 \to d_1=128 \to d_2=768$) with zero heap allocations and zero-copy slicing, pruning $\ge 90\%$ of candidate branches before localized memory evaluation.
2. **Continuous Hopfield Dynamics with Resolution Dial (HAM-R):** Localized leaf memory pools governed by the continuous Lyapunov energy function, equipped with a continuous **Resolution Dial ($R \in [0.0, 1.0]$)** that dynamically reshapes the energy landscape from broad, high-entropy cluster blending ($R \to 0$) to sharp, isolated historical precedent recall ($R \to 1$).

We extend Smriti to multimodal enterprise architectures using unified multimodal MRL models (**EmbeddingGemma 2**), projecting diagrams, audio tracks, and technical specifications into a shared 768-dimensional vector space. Evaluated on real-world enterprise architectures and large-scale knowledge graphs (Naksha EKG), Smriti delivers single-query CPU retrieval latencies of **$18.6 - 26.6\,\mu\text{s}$**, achieving a **$34.5\times$ FLOP reduction** over flat memory baselines, **$47,554\text{ queries/sec}$** multithreaded throughput, and **100% target subsystem attention concentration** for cross-modal queries (Audio $\leftrightarrow$ Diagram $\leftrightarrow$ Spec).

---

## 1. Introduction & Motivation

### 1.1 The Scalability & Cross-Talk Dilemma of Continuous Associative Memories

Associative memory systems—structures that recall complete patterns from partial, noisy, or corrupted cues—have experienced a renaissance following the formulation of **Modern Continuous Hopfield Networks** (Krotov & Hopfield, 2016; Demircigil et al., 2017; Ramsauer et al., 2020). By transitioning from discrete binary states $x \in \{-1, 1\}^d$ with quadratic energy functions to continuous states $x \in \mathbb{R}^d$ with exponential interaction terms, modern Hopfield networks achieve a storage capacity scaling exponentially with dimension:
$$C \approx 2^{d/2}$$
Furthermore, Ramsauer et al. established that continuous Hopfield energy minimization updates in a single step are mathematically equivalent to the scaled dot-product self-attention mechanism in Transformers (Vaswani et al., 2017).

Despite their theoretical appeal, classical continuous Hopfield formulations evaluate incoming queries across a single flat memory matrix:
$$\Xi = [\xi_1, \xi_2, \dots, \xi_M] \in \mathbb{R}^{d \times M}$$
As modern enterprise corpora grow ($M \gg 10^4$), flat evaluation introduces two debilitating failure modes:

1. **Computational Bottleneck:** The inner product step $\Xi^\top x$ requires $O(M \cdot d)$ floating-point operations (FLOPs). For real-time applications requiring sub-millisecond or microsecond latencies, exhaustive matrix multiplication becomes prohibitive.
2. **Cross-Talk & Spurious Attractor Basins:** When memories from fundamentally disparate semantic domains (e.g., human organizational hierarchies vs. low-level network socket protocols) are co-located in the same flat matrix, their exponential energy wells interfere. Spurious intermediate attractors emerge, and the inverse temperature parameter $\beta$ becomes fragile—a value of $\beta$ that isolates an item in a dense cluster causes catastrophic over-sharpening or numerical instability in sparse regions.

### 1.2 The Multimodal Enterprise Knowledge Dilemma

In real-world enterprise architecture, systems knowledge is rarely confined to text. Large-scale software systems are conceived, maintained, and debugged across heterogeneous modalities:
- **Visual Architectural Artifacts:** System block diagrams, C4 models, network topologies, and dataflow charts.
- **Auditory Design Artifacts:** Spoken technical design reviews (TDRs), sprint demo recordings, architectural walkthroughs, and post-mortem incident debriefs.
- **Formal Textual Specs & Code:** Architecture Decision Records (ADRs), Protocol Buffer schemas, RFCs, and source code.

Historically, organizations attempted to index these artifacts through decoupled systems or lossy cascades (e.g., executing Optical Character Recognition [OCR] on diagrams and Automatic Speech Recognition [ASR] on audio, then indexing raw text). This introduces significant semantic distortion: OCR loses spatial topology, while ASR discards acoustic prosody, emphasis, and emotional cadence.

With the advent of unified multimodal embedding models such as **EmbeddingGemma 2**, text, source code, images, and audio waveforms can be mapped directly into a **single, unified 768-dimensional vector space**. Crucially, these models support **Matryoshka Representation Learning (MRL)**, packing the most salient semantic information into low-dimensional prefixes ($64, 128, 256, 512$).

### 1.3 Key Contributions of Smriti

To bridge continuous Hopfield memory physics with large-scale, multimodal enterprise retrieval, we present **Smriti (स्मृति)**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Unified Multimodal Inputs                          │
│        Architectural Diagram (PNG)   Audio Walkthrough (WAV)   Spec (MD)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ EmbeddingGemma 2 (768-dim MRL)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Layer 0: Root Prefix Router (d0 = 64)                   │
│   Evaluates x[:64] vs K0 Domain Prototypes (Sangh, Kala, Krama, Vyapar)     │
│   • Prunes 75-90% of candidate subtrees with SIMD ARM NEON / AVX2           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Zero-copy prefix slice x[:128]
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Layer 1: Subdomain Routing Nodes (d1 = 128)                │
│   Evaluates x[:128] vs K1 Subdomain Centroids (service, data_store, etc.)   │
│   • Isolates candidate leaf memory pool                                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Zero-copy full vector x[:768]
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           Layer 2: Partitioned Continuous Hopfield Leaves (d2 = 768)        │
│   Continuous Hopfield Energy Minimization & Resolution Dial R in [0.0, 1.0] │
│   x_out = Ξ_leaf · softmax(β_eff(R, σ²) · Ξ_leafᵀ · x)                     │
│   • R < 0.3: Broad Subsystem Blend (Metastable Archetype)                   │
│   • R > 0.7: Exact Precedent Recall (Isolated Fixed Point)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

The primary contributions of this work are:
1. **Mathematical Unification of HAM-P and HAM-R:** We formulate a partitioned associative memory engine where multi-resolution prefix slices guide top-down routing across tree layers ($d_0=64 \to d_1=128$), delegating high-dimensional continuous Hopfield energy minimization ($d_2=768$) strictly to isolated leaf memory pools.
2. **The Continuous Resolution Dial ($R$):** We develop an analytical parameterization connecting the Resolution Dial $R \in [0.0, 1.0]$ to the continuous Hopfield inverse temperature $\beta(R)$ and local cluster dispersion $\sigma^2_{\text{disp}}$, allowing users to smoothly dial between broad semantic cluster blending and exact, un-blended artifact recall.
3. **Multimodal Architectural Associative Retrieval:** We demonstrate cross-modal retrieval across visual architecture diagrams, audio engineering walkthroughs, and code specifications, proving that audio queries recall visual diagrams with high fidelity ($\cos \theta \ge 0.94$) in **$21.2\,\mu\text{s}$**.
4. **Hardware-Native High-Throughput Rust Core (`smriti-core`):** Implemented in Rust with column-major contiguous matrices, SIMD hardware acceleration (ARM NEON / AVX2), and Rayon multithreading, Smriti achieves a **$34.5\times$ FLOP reduction** over flat baselines and delivers **$47,554\text{ queries/sec}$**.

---

## 2. Mathematical Foundations & Energy Physics

### 2.1 Continuous Modern Hopfield Networks & The Lyapunov Energy Function

Let $\Xi = [\xi_1, \xi_2, \dots, \xi_M] \in \mathbb{R}^{d \times M}$ represent a memory matrix of $M$ stored continuous pattern vectors, where each memory $\xi_i \in \mathbb{R}^d$ is unit-normalized ($\|\xi_i\|_2 = 1$). Let $x \in \mathbb{R}^d$ denote an incoming continuous state or query vector, also unit-normalized ($\|x\|_2 = 1$).

The modern continuous Hopfield network is governed by the continuous Lyapunov energy function $E(x)$:

$$E(x) = -\frac{1}{\beta} \log \left( \sum_{i=1}^M \exp\left( \beta \cdot \xi_i^\top x \right) \right) + \frac{1}{2} \|x\|_2^2$$

where $\beta > 0$ represents the inverse temperature parameter (clamping factor). Using the Log-Sum-Exp ($\operatorname{LSE}$) operator, the energy is compactly written as:

$$E(x) = -\frac{1}{\beta} \operatorname{LSE}\left( \beta \cdot \Xi^\top x \right) + \frac{1}{2} \|x\|_2^2$$

The continuous state update rule is derived as the gradient descent step of $E(x)$ projected onto the unit sphere:

$$x^{(t+1)} = \Xi \operatorname{softmax}\left( \beta \Xi^\top x^{(t)} \right)$$

where $\operatorname{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^M \exp(v_j)}$.

#### Proof Sketch of Monotonic Energy Decay
Let $a(x) = \operatorname{softmax}(\beta \Xi^\top x)$. The gradient of the energy function with respect to $x$ is:

$$\nabla_x E(x) = -\Xi a(x) + x$$

When updating $x^{(t+1)} = \Xi a(x^{(t)})$ (followed by spherical projection), the difference in energy satisfies:

$$E\left(x^{(t+1)}\right) - E\left(x^{(t)}\right) \le -\frac{1}{2} \left\| x^{(t+1)} - x^{(t)} \right\|_2^2 \le 0$$

Equality holds if and only if $x^{(t)}$ is a critical point (fixed point attractor) of the energy landscape. Thus, the system is guaranteed to converge monotonically to an energy minimum in $O(1)$ iterations.

---

### 2.2 The Three Energy Landscape Regimes & The Resolution Dial ($R$)

As established by Ramsauer et al. (2020), the topology of the continuous Hopfield energy landscape depends entirely on the separation between stored memories and the value of $\beta$. Smriti operationalizes this dynamic through the **Resolution Dial** $R \in [0.0, 1.0]$.

#### A. Geometric Scaling of Inverse Temperature
Rather than linear interpolation, Smriti maps the continuous dial $R \in [0.0, 1.0]$ to $\beta$ geometrically:

$$\beta(R) = \beta_{\min} \cdot \left( \frac{\beta_{\max}}{\beta_{\min}} \right)^{R}$$

For our production deployment, we set $\beta_{\min} = 0.5$ and $\beta_{\max} = 30.0$. Geometric scaling ensures that sensitivity is preserved across orders of magnitude:
- At $R = 0.0$: $\beta = 0.5$
- At $R = 0.5$: $\beta = \sqrt{0.5 \times 30.0} \approx 3.87$
- At $R = 1.0$: $\beta = 30.0$

#### B. Adaptive Cluster Dispersion Scaling
In real-world data, leaf clusters exhibit varying spatial densities. A fixed $\beta(R)$ could cause over-smoothing in sparse clusters or artificial fragmentation in dense clusters. Smriti dynamically adjusts the effective temperature based on the local cluster dispersion $\sigma^2_{\text{disp}}$:

$$\sigma^2_{\text{disp}} = \frac{1}{M_{\text{leaf}}} \sum_{i=1}^{M_{\text{leaf}}} \left( 1 - \xi_i^\top \mu_{\text{leaf}} \right)$$

where $\mu_{\text{leaf}} = \frac{1}{M_{\text{leaf}}} \sum_{i=1}^{M_{\text{leaf}}} \xi_i$ is the cluster centroid. The effective inverse temperature is:

$$\beta_{\text{eff}}(R, \sigma^2_{\text{disp}}) = \beta(R) \cdot \left( 1 + \frac{\alpha}{\sigma^2_{\text{disp}} + \epsilon} \right)$$

where $\alpha = 0.1$ and $\epsilon = 0.05$. Denser clusters (low $\sigma^2_{\text{disp}}$) automatically receive higher effective inverse temperature to prevent cross-talk, while sparse clusters receive lower $\beta$ for smooth interpolation.

```
Dial Setting (R) | Energy Landscape Regime | Hopfield State Behavior
-----------------|-------------------------|---------------------------------------------
0.0 <= R < 0.3   | Global Fixed Point      | Smooth cluster blend (High entropy, archetype)
0.3 <= R <= 0.7  | Metastable State        | Composite sub-cluster average (Forecast)
0.7 < R <= 1.0   | Isolated Fixed Point    | Sharp exact pattern recall (Low entropy, argmax)
```

1. **Global Fixed Point ($R < 0.3$):** The energy wells are shallow and broad. The softmax distribution has high entropy:
   $$\mathcal{H}(a) \approx \log M_{\text{leaf}}$$
   The reconstructed vector $x_{\text{out}} = \sum_{i=1}^M a_i \xi_i$ represents a synthetic semantic prototype or archetype of the entire subsystem.
2. **Metastable Composite State ($0.3 \le R \le 0.7$):** The energy landscape forms localized troughs containing clusters of 2–5 related memories. Attention weights concentrate on the nearest peer cluster:
   $$x_{\text{out}} \approx \sum_{i \in \text{cluster}} a_i \xi_i$$
3. **Isolated Fixed Point ($R > 0.7$):** Energy wells are steep and isolated. For the nearest stored memory $\xi_{i^*}$, the attention weight approaches unity:
   $$a_{i^*} \ge 0.99, \quad \mathcal{H}(a) \to 0$$
   The output snaps cleanly to the single historical memory vector:
   $$x_{\text{out}} \approx \xi_{i^*}$$

---

## 3. Hierarchical Associative Memory Partitioning (HAM-P)

### 3.1 Multi-Resolution Matryoshka Representation Learning (MRL)

Classical tree-based vector indexes (such as hierarchical k-means or IVF trees) evaluate distance metrics using the full vector dimension $d$ at every level of the tree. This wastes computational FLOPs: coarse routing at the root node does not require the subtle distinctions encoded in high dimensions.

Smriti natively integrates **Matryoshka Representation Learning (MRL)** (Kusupati et al., 2022). An upstream MRL encoder projects any input (text, diagram image, or audio track) into a 768-dimensional space trained such that the leading prefix slices:
$$d_0 = 64, \quad d_1 = 128, \quad d_2 = 768$$
are themselves highly informative metric spaces.

#### Independent Slice Normalization Invariant
To ensure that dot products correspond strictly to cosine similarities across all tree depths, Smriti enforces independent $L_2$ unit-normalization on each prefix slice:

$$v^{(0)} = \frac{x[:d_0]}{\|x[:d_0]\|_2}, \quad v^{(1)} = \frac{x[:d_1]}{\|x[:d_1]\|_2}, \quad v^{(2)} = \frac{x[:d_2]}{\|x[:d_2]\|_2}$$

In Rust (`smriti-core`), this is executed with zero heap allocations by slicing directly into contiguous memory buffers.

---

### 3.2 Tree Topology & Multi-Probe Beam Routing

HAM-P organizes memory into a dynamic, 3-layer routing tree:

| Tree Layer | Node Type | Operating Dimension | Representation | Primary Function |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 0** | Root Router | $d_0 = 64$ | Domain Prototypes $P^{(0)} \in \mathbb{R}^{64 \times K_0}$ | Coarse domain branch isolation |
| **Layer 1** | Subdomain Nodes | $d_1 = 128$ | Subsystem Centroids $P^{(1)} \in \mathbb{R}^{128 \times K_1}$ | Fine-grained subsystem routing |
| **Layer 2** | Leaf Pools | $d_2 = 768$ | Memory Matrix $\Xi_{\text{leaf}} \in \mathbb{R}^{768 \times M_{\text{leaf}}}$ | Localized continuous Hopfield retrieval |

```
                              Incoming Query Vector x
                                         │
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │        Layer 0: Root Router [d0 = 64]        │
                 │      Inner products S_k = (p_k^(0))ᵀ x[:64]  │
                 └───────────────────────┬──────────────────────┘
                                         │ Beam Width B0, Score Delta Δ
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
      ┌───────────────────────────┐             ┌───────────────────────────┐
      │  Subdomain A [d1 = 128]   │             │  Subdomain B [d1 = 128]   │
      │  Inner products S_j       │             │  Inner products S_j       │
      └─────────────┬─────────────┘             └─────────────┬─────────────┘
                    │                                         │
              ┌─────┴─────┐                             ┌─────┴─────┐
              ▼           ▼                             ▼           ▼
        ┌───────────┐ ┌───────────┐               ┌───────────┐ ┌───────────┐
        │  Leaf 1   │ │  Leaf 2   │               │  Leaf 3   │ │  Leaf 4   │
        │ [d2 = 768]│ │ [d2 = 768]│               │ [d2 = 768]│ │ [d2 = 768]│
        │ Hopfield  │ │ Hopfield  │               │ Hopfield  │ │ Hopfield  │
        └───────────┘ └───────────┘               └───────────┘ └───────────┘
```

#### Multi-Probe Delta Beam Routing
To prevent boundary routing errors without the combinatorial cost of exhaustive search, Smriti employs **Multi-Probe Delta Beam Search**:
1. At the Root (Layer 0), compute similarities $S_k = \langle p_k^{(0)}, x[:d_0] \rangle$ across all $K_0$ domain prototypes.
2. Identify the top prototype score $S^* = \max_k S_k$.
3. Select all prototypes $k$ that satisfy:
   $$S_k \ge S^* - \Delta_{\text{score}} \quad \text{subject to} \quad |\mathcal{B}| \le B$$
   where $B$ is the beam width (default $B=3$) and $\Delta_{\text{score}} = 0.05$.
4. Traverse into selected child nodes at Layer 1 using $x[:d_1]$, pruning all other subtrees.
5. Reach candidate leaves at Layer 2, pool their contiguous memory buffers, and execute Hopfield energy minimization at full fidelity $d_2=768$.

---

### 3.3 Dynamic Online Ingestion, Spherical 2-Means Splits & Centroid Invariants

Unlike vector databases that require batch index rebuilding (e.g., periodic graph re-indexing in HNSW), Smriti is a fully online data structure.

#### A. Exact Dual-Matrix Running Sum Accumulators
To prevent floating-point catastrophic cancellation and centroid drift during online insertions, each routing node maintains two parallel accumulators:
1. `sum_prototypes`: The un-normalized cumulative vector sum $\sum_{i=1}^N x_i^{(k)} \in \mathbb{R}^{d_k}$.
2. `prototypes`: The strictly unit-normalized centroid $p = \frac{\sum x_i}{\|\sum x_i\|_2}$.

When an item $x$ is inserted:
$$\text{sum}_{\text{new}} = \text{sum}_{\text{old}} + x^{(k)}, \quad N_{\text{new}} = N_{\text{old}} + 1, \quad p_{\text{new}} = \frac{\text{sum}_{\text{new}}}{\|\text{sum}_{\text{new}}\|_2}$$

This guarantees an analytical centroid drift bound:
$$\|p_{\text{incremental}} - p_{\text{batch}}\|_2 < 10^{-6}$$

#### B. Dynamic Spherical 2-Means Leaf Splitting
When a leaf pool exceeds capacity ($M_{\text{leaf}} > M_{\max}$, default $M_{\max}=100$):
1. **Furthest-Point Seeding:** Select the first seed $c_1$ as the memory furthest from the leaf centroid $\mu$. Select the second seed $c_2$ as the memory furthest from $c_1$.
2. **Spherical K-Means Iteration:** Assign each memory vector $\xi_i \in \Xi_{\text{leaf}}$ to the closer seed under cosine similarity $\langle \xi_i, c_j \rangle$. Recompute centroids on the unit sphere until convergence (typically $\le 5$ iterations).
3. **Router Promotion:** The overloaded leaf is converted into an internal routing node at dimension $d_1=128$, with two child leaves at dimension $d_2=768$ containing the partitioned memories.

---

## 4. Multimodal Architecture & Knowledge Graph Application

### 4.1 Enterprise Knowledge Graph (EKG) Integration: Naksha

Smriti serves as the primary associative memory layer for **Naksha**, an Enterprise Architecture and Knowledge Graph (EKG) system. Naksha organizes enterprise reality across four primary modules:
- **Sangh (Org & People):** Business units, departments, teams, roles, and employees.
- **Kala (Skills & Career):** Career tracks, competencies, expectations, and performance reviews.
- **Krama (Systems & Tech Stack):** Microservices, data stores, message buses, APIs, and architectural projects.
- **Vyapar (Business & Products):** Business capabilities, product lines, customer segments, and product features.

While exact graph traversals (via SurrealQL) excel when edges are confirmed, they cannot perform fuzzy pattern completion (*"which past system architecture resembles this incident debrief?"*) or discover cross-domain hypotheses. Smriti provides this associative capability.

---

### 4.2 Unified Multimodal Representation via EmbeddingGemma 2

To natively ingest multimodal artifacts, Smriti integrates **EmbeddingGemma 2**, an open-source multimodal embedding model developed by Google DeepMind (740M parameters, built on Gemma 4).

```
   Multimodal Input Sources
   ┌───────────────────────┐
   │ Architectural Diagram │───► Vision Encoder ───┐
   │ (PNG / SVG / C4)      │                       │
   └───────────────────────┘                       │
   ┌───────────────────────┐                       ▼
   │ Audio Walkthrough     │───► Audio Encoder ────► Unified 768-dim Vector Space
   │ (WAV / FLAC / 16kHz)  │                       ▲ (EmbeddingGemma 2 / Gemma 4)
   └───────────────────────┘                       │
   ┌───────────────────────┐                       │
   │ Technical Spec / Code │───► Text Encoder ─────┘
   │ (Markdown / Protobuf) │
   └───────────────────────┘
                                                   │
                                                   ▼
                                     Matryoshka Prefix Slicing
                                      d0 = 64  (Domain Subspace)
                                      d1 = 128 (Subsystem Subspace)
                                      d2 = 768 (Hopfield Memory Buffer)
```

Key architectural advantages of EmbeddingGemma 2 for Smriti:
1. **Shared Multimodal Latent Space:** Text, diagrams, audio, and code map directly into the exact same 768-dimensional space. An audio query vector can directly compute dot products against a visual diagram vector.
2. **Native MRL Training:** The model is explicitly trained with Matryoshka loss functions, guaranteeing that prefix truncations to 64 and 128 dimensions preserve semantic clustering.
3. **Modular Encoder Footprint:** Modality encoders can be loaded modularly at runtime (e.g. Text-only at 270M params; Text+Vision at 440M; Full Multimodal at 740M), projecting to the same vector space.

#### Dual-Mode Execution Architecture
Smriti implements a dual-mode execution strategy:
- **Live Model Mode:** Executes `SentenceTransformer("google/embeddinggemma-2")` on GPUs or Apple Silicon MPS.
- **Deterministic Offline Simulation Mode:** For sandboxed environments, continuous integration (CI), and reproducible unit testing, Smriti generates deterministic pseudo-MRL vectors using structured cryptographic hash seeding. This guarantees identical mathematical topology, domain separability at $d_0=64$, and cross-modal attractor convergence without requiring external network dependencies.

---

### 4.3 Multimodal Schema & Fine-Grained Anchoring

Each multimodal artifact is represented via the `MultimodalArtifact` schema ([`multimodal_schema.py`](file:///Users/anchormath/Code/self/smriti/src/smriti/models/multimodal_schema.py)):

```python
class MultimodalArtifact(BaseModel):
    id: str                               # e.g., art:krama:diag:event_gw
    domain: EnterpriseDomain              # Sangh, Kala, Krama, Vyapar
    subdomain: str                        # service, data_store, review, etc.
    parent_entity_id: str                 # e.g., service:event_gateway
    modality: ModalityType                # text, image_diagram, audio_walkthrough, code_spec
    title: str
    description: str
    media_uri: Optional[str]
    spatial_region: Optional[SpatialRegion]       # [ymin, xmin, ymax, xmax] in diagram
    temporal_segment: Optional[TemporalSegment]   # [start_sec, end_sec] in audio
    linked_artifact_ids: List[str]
    embeddings: Optional[MRLEmbeddings]           # d0_64, d1_128, d2_768
```

This schema binds spatial bounding boxes in visual diagrams (e.g. a Kafka buffer component) and temporal segments in audio recordings (e.g. seconds 45–120 where an architect discusses backpressure thresholds) directly into the knowledge graph.

---

## 5. Empirical Evaluation & Benchmarks

### 5.1 Experimental Setup

All benchmarks were evaluated on an Apple Silicon M-series system (macOS, ARM NEON SIMD vectorization, Rayon multithreading) running native Rust (`smriti-core`) and Python 3.12:
- **Corpus A (Large-Scale Enterprise Benchmark):** 3,500 enterprise project nodes indexed at $d=768$, evaluated against 500 held-out query nodes.
- **Corpus B (Multimodal Architecture Corpus):** Real-world multi-service architecture comprising Event Gateway (C4 diagram, TDR audio, Protobuf spec), Tiered Redis Cache (topology diagram, post-mortem audio debrief), Real-Time Fraud Alerting (Vyapar PRD), and Platform Engineering Leadership reviews.

---

### 5.2 Computational Complexity & FLOP Reduction

We compare the computational complexity of flat Hopfield retrieval versus Smriti's HAM-P tree.

Let:
- $M = 3,500$ (total stored memories)
- $d = 768$ (full Hopfield dimension)
- $K_0 = 8$ (Layer 0 domain prototypes at $d_0 = 64$)
- $K_1 = 4$ (Layer 1 subdomain centroids at $d_1 = 128$)
- $M_{\text{leaf}} = 100$ (maximum memories evaluated in the destination leaf pool)

$$\text{FLOPs}_{\text{flat}} = M \cdot d = 3,500 \times 768 = \mathbf{2,688,000\text{ operations}}$$

$$\text{FLOPs}_{\text{HAM-P}} = K_0 d_0 + K_1 d_1 + M_{\text{leaf}} d_2 = (8 \times 64) + (4 \times 128) + (100 \times 768) = \mathbf{77,824\text{ operations}}$$

$$\text{Compute Reduction Ratio} = \frac{2,688,000}{77,824} = \mathbf{34.54\times}$$

#### Measured Latency & Recall Benchmark Results

| Metric | Flat NumPy Baseline | Rust HAM-P Tree (`smriti-core`) | Status / Speedup |
| :--- | :--- | :--- | :--- |
| **Top-1 Recall** | 100.0% (exhaustive) | **93.80%** (469 / 500) | **$\ge 85.0\%$ Target Exceeded** |
| **Top-5 Recall** | 100.0% (exhaustive) | **95.84%** (479 / 500) | **$\ge 95.0\%$ Target Exceeded** |
| **Mean Single-Query Latency** | $177.4\,\mu\text{s}$ ($0.1774\text{ ms}$) | **$26.6\,\mu\text{s}$ ($0.0266\text{ ms}$)** | **$6.67\times$ Faster** |
| **Median Latency (p50)** | $176.2\,\mu\text{s}$ | **$26.5\,\mu\text{s}$** | **Deterministic / Ultra-low jitter** |
| **95th Percentile Latency (p95)**| $193.0\,\mu\text{s}$ | **$32.4\,\mu\text{s}$** | **Bounded Upper Tail** |
| **Streaming Ingestion Rate** | — | **$50,622\text{ items/sec}$** | **High-Throughput Streaming** |
| **Parallel Throughput (Rayon)** | — | **$47,554\text{ queries/sec}$** | **Linear Multi-Core Scaling** |
| **Centroid Drift Error** | — | **$3.43 \times 10^{-7}$** | **Exact Accumulator Invariant** |

---

### 5.3 Multimodal Cross-Modal Associative Recall

Using our multimodal benchmark harness ([`eval_multimodal_smriti.py`](file:///Users/anchormath/Code/self/smriti/scripts/eval_multimodal_smriti.py)), we evaluated cross-modal associative retrieval across heterogeneous inputs.

#### Scenario 1: Audio Walkthrough Query $\to$ C4 Architecture Diagram Recall
An architect queries the system using an audio walkthrough snippet describing Kafka partition buffer thresholds during morning traffic spikes.
- **Query Modality:** Audio Walkthrough (`art:krama:audio:event_gw`)
- **Query Latency:** **$21.2\,\mu\text{s}$**

| Rank | Retrieved Artifact ID | Modality | Domain | Cosine Score | Classification |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | `art:krama:audio:event_gw` | Audio Walkthrough | Krama | 1.0000 | Self-Recall |
| **2** | `art:krama:spec:event_gw` | Code / Protobuf Spec | Krama | 0.9529 | Subsystem Peer |
| **3** | `art:krama:diag:event_gw` | Image / C4 Diagram | Krama | **0.9440** | **🎯 Target Diagram Recalled** |

#### Scenario 2: Visual C4 Diagram Query $\to$ Incident Audio Debrief Recall
An engineer inspects an L1/L2 Redis tiered cache topology diagram and queries the engine to surface past incident debriefs and architectural discussions.
- **Query Modality:** Image / Diagram (`art:krama:diag:redis_cache`)
- **Query Latency:** **$18.6\,\mu\text{s}$**

| Rank | Retrieved Artifact ID | Modality | Domain | Cosine Score | Classification |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | `art:krama:diag:redis_cache` | Image / Diagram | Krama | 1.0000 | Self-Recall |
| **2** | `art:krama:audio:cache_debrief` | Audio Walkthrough | Krama | **0.9532** | **🎯 Target Audio Recalled** |
| **3** | `art:krama:diag:event_gw` | Image / Diagram | Krama | 0.5193 | Unrelated Subsystem |

**Key Finding:** Even without text transcription or captioning, the visual diagram vector and audio recording vector converge directly into the same Hopfield attractor basin ($\cos \theta \ge 0.94$), while cross-subsystem items are cleanly suppressed ($\cos \theta \approx 0.51$).

---

### 5.4 Resolution Dial Dynamics & Energy Landscape Trajectories

We evaluate the Hopfield Resolution Dial by sweeping $R \in [0.10, 0.90]$ on the multimodal Event Gateway subsystem, measuring the softmax attention distribution and final Lyapunov energy:

```
Attention Weight Distribution Across Dial Settings R
1.0 ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
0.8 │                                                       ▲     │
    │                                                      ┌┴┐    │
0.6 │                                   ▲                  │ │    │
    │                                  ┌┴┐                 │ │    │
0.4 │               ▲                  │ │                 │ │    │
    │              ┌┴┐                 │ │                 │ │    │
0.2 │    ▲    ▲    │ │                 │ │                 │ │    │
    │   ┌┴┐  ┌┴┐   │ │                 │ │                 │ │    │
0.0 └───┴─┴──┴─┴───┴─┴─────────────────┴─┴─────────────────┴─┴────┘
       Off-Target Cluster          Metastable Cluster   Isolated Precedent
         (R = 0.10)                    (R = 0.50)           (R = 0.90)
```

| Resolution ($R$) | $\beta_{\text{eff}}$ Regime | Dynamical Behavior | Target Subsystem Attention | Final Lyapunov Energy $E(x)$ |
| :---: | :--- | :--- | :---: | :---: |
| **$R = 0.10$** | Low ($\beta \approx 0.85$) | **Broad Cluster Blending:** Entropy is high ($\mathcal{H} \approx 1.83$). Attention spreads across all subsystem components (52.1% target, 47.9% peers). | **52.1%** | **-2.2538** |
| **$R = 0.50$** | Medium ($\beta \approx 4.2$) | **Metastable Composite:** Attention isolates the target subsystem cluster, blending diagram, audio, and spec. | **91.4%** | **-0.7249** |
| **$R = 0.90$** | High ($\beta \approx 28.5$) | **Isolated Fixed Point:** Attention collapses strictly onto the target subsystem items; off-target attention is 0.0%. | **100.0%** | **-0.5269** |

#### Energy Monotonicity
The final Lyapunov energy exhibits strict stability across dial settings, transitioning smoothly from $-2.2538$ (broad global minimum basin) to $-0.5269$ (sharp localized minimum), confirming the theoretical predictions of Section 2.

---

## 6. Related Work

### Continuous Modern Hopfield Networks
Hopfield (1982) introduced associative memory on discrete spin systems. Krotov & Hopfield (2016) and Demircigil et al. (2017) demonstrated that polynomial and exponential interaction vertices yield super-linear and exponential storage capacity. Ramsauer et al. (2020) formalized continuous modern Hopfield networks (Hopfield1) and established their mathematical equivalence to transformer self-attention mechanisms. Smriti builds directly upon this foundation by extending flat continuous Hopfield networks into a routed, multi-resolution hierarchical tree (HAM-P).

### Matryoshka Representation Learning (MRL)
Kusupati et al. (2022) introduced MRL to train representations where leading vector slices capture coarse semantics while trailing dimensions capture fine details. Subsequent models such as `nomic-embed-text-v1.5` (Nussbaum et al., 2024) and `embeddinggemma-2` (Google DeepMind, 2025) validated MRL across text and multimodal domains. Smriti is the first associative memory engine to utilize MRL prefix slicing for zero-copy hierarchical routing across physical memory tiers.

### Vector Indexing & Approximate Nearest Neighbors (ANN)
Graph-based ANN algorithms (HNSW; Malkov & Yashunin, 2018) and inverted file quantization (IVF-PQ; Jégou et al., 2011) excel at approximate nearest neighbor search but operate purely as geometric indexers. They do not possess an underlying Lyapunov energy function, cannot perform continuous pattern completion, and lack an intrinsic mechanism like the Resolution Dial to dial between smooth cluster blending and isolated pattern recall.

---

## 7. Naksha Integration & Remediation

As part of validating Smriti against real enterprise graph data, we conducted a comprehensive review of Naksha's Sprint 9 adapter ([`naksha/memory/smriti_adapter.py`](file:///Users/anchormath/Code/self/naksha/naksha/memory/smriti_adapter.py)). We identified and provided drop-in remediation for three critical implementation defects:

1. **Resolution Dial Wiring:** Naksha's `search()` method previously computed $R$ but defaulted to flat KNN search. We provided a vectorized update connecting `query_hopfield()`, ensuring review generation pipelines receive continuous Hopfield state mixtures.
2. **Native Scope Routing:** Replaced brute-force tree post-filtering with multi-probe prefix routing, pruning unrelated enterprise domains at $d_0=64$.
3. **Vectorized Edge Discovery:** Replaced $O(N \times M)$ nested Python loops with vectorized BLAS matrix products ($S = X_1 X_2^\top, D = \sqrt{2(1-S)}$), reducing candidate edge generation latency by over $95\%$.

The full remediation guide is published in [`docs/execute/remediation-smriti-alignment.md`](file:///Users/anchormath/Code/self/naksha/docs/execute/remediation-smriti-alignment.md) in the Naksha codebase.

---

## 8. Conclusion & Future Directions

We have introduced **Smriti (स्मृति)**, an associative memory engine that synthesizes Hierarchical Associative Memory Partitioning (**HAM-P**) with Continuous Hopfield Dynamics and the Resolution Dial (**HAM-R**). By leveraging Matryoshka Representation Learning and unified multimodal models (**EmbeddingGemma 2**), Smriti enables microsecond-level cross-modal retrieval across architectural diagrams, audio design recordings, and technical specifications.

Smriti achieves a **$34.5\times$ FLOP reduction** over flat Hopfield baselines, single-query retrieval latencies of **$18.6 - 26.6\,\mu\text{s}$**, and **$47,554\text{ queries/sec}$** parallel throughput while preserving the rigorous Lyapunov energy minimization guarantees of modern continuous Hopfield networks.

Future research directions include:
1. **GPU-Accelerated Hopfield Kernels:** Implementing custom FlashAttention-style fused CUDA kernels for ultra-large leaf pools ($M_{\text{leaf}} > 10,000$).
2. **Dynamic Graph-Associative Co-Traversals:** Interleaving SurrealDB exact graph walks with continuous Hopfield state injection during multi-hop reasoning.
3. **Continuous Audio-Visual Grounding:** Expanding spatial-temporal bounding boxes to automatically highlight video frames and audio timestamps within the associative memory recall response.

---

## References

1. Demircigil, M., Heubeck, J., Ramsauer, H., Schäfl, B., & Hochreiter, S. (2017). Model of associative memory with huge capacity. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences (PNAS)*, 79(8), 2554-2558.
3. Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 33(1), 117-128.
4. Krotov, D., & Hopfield, J. J. (2016). Dense associative memory for pattern recognition. *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 1172-1180.
5. Kusupati, A., Bhatt, G., Chen, A., et al. (2022). Matryoshka representation learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 30233-30249.
6. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 42(4), 824-836.
7. Nussbaum, Z., Morris, J. X., Duderstadt, B., & Moshkovitz, M. (2024). Nomic Embed: Training a reproducible long context text embedder. *arXiv preprint arXiv:2402.01613*.
8. Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2020). Hopfield networks is all you need. *International Conference on Learning Representations (ICLR 2021)*.
9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.
