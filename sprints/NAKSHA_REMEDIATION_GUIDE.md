# Naksha Agent Remediation Guide: Smriti Associative Memory & Multimodal Alignment

**Target Repository:** `/Users/anchormath/Code/self/naksha`  
**Primary Files to Modify:**
1. `naksha/memory/smriti_adapter.py`
2. `naksha/engine/associative.py`
3. `naksha/engine/review.py`
4. `naksha/models/` (or schema representation of artifacts)
5. `tests/test_smriti_adapter.py`

---

## 1. Executive Summary & Root Cause Analysis

Naksha's Sprint 9 (`docs/execute/sprint-9-smriti-associative-memory.md`) integrates Smriti's Hierarchical Associative Memory Partitioning (**HAM-P**) and Continuous Hopfield Dynamics with Resolution Dial (**HAM-R**).

While the initial plumbing spike successfully proved tree insertion and 100% self-recall, a code inspection of `naksha/memory/smriti_adapter.py` reveals **three critical defects** and **two major optimization gaps** that must be remediated:

1. **Defect 1: The Resolution Dial ($R$) is completely disconnected in `SmritiAdapter.search()`**:
   - `resolution` is clipped to `r_val`, but `self.tree.query()` is invoked instead of `self.tree.query_hopfield()`. As a result, calls to `search(..., resolution=0.2)` and `search(..., resolution=0.9)` produce identical results and ignore Hopfield dynamics.
2. **Defect 2: Inefficient Scope Filtering Bypasses HAM-P Hierarchy**:
   - In `search()`, setting `scope="Krama"` triggers `fetch_k = total_items` and `beam_width = 16`, fetching almost the entire tree into Python memory and post-filtering with `if domain.lower() != scope_filter.lower(): continue`. This bypasses $d_0=64$ prefix routing and degrades performance to $O(N)$ Python overhead.
3. **Defect 3: Brute-Force $O(N \times M)$ Pairwise Loop in `discover_candidate_edges()`**:
   - Pairwise distance evaluation runs in an unvectorized nested Python loop computing `np.linalg.norm(s_vec - t_vec)`.
4. **Enhancement 4: Multimodal Ingestion Support (EmbeddingGemma 2)**:
   - `insert()` only accepts text/JSON records and flattens them via `record_to_text()`. It needs to accept multimodal media references (`image`, `audio`, `spec`) so Naksha can link visual C4 diagrams and audio TDR walkthroughs directly to Krama services and projects.

---

## 2. Step-by-Step Remediation Plan for Naksha Agent

### Step 1: Fix the Resolution Dial in `naksha/memory/smriti_adapter.py`

#### Location: `SmritiAdapter.search()` (Lines 212–260)

**The Defect:**
```python
# CURRENT BROKEN CODE:
q_vec = self.encode_query(query)
r_val = float(np.clip(resolution, 0.0, 1.0))
# r_val is never passed to tree.query!
raw_results = self.tree.query(q_vec, top_k=max(fetch_k, 5), beam_width=bw)
```

**The Remediation:**
`SmritiAdapter.search()` should support two operating modes:
1. When `resolution is None`: Standard nearest-neighbor prefix routing (`tree.query()`).
2. When `resolution is float` ($R \in [0.0, 1.0]$): Continuous Hopfield associative retrieval (`tree.query_hopfield()`), returning the Hopfield energy, reconstructed state vector, and softmax candidate weights.

```python
    def search(
        self,
        query: str | np.ndarray | list[float],
        scope: MemoryScope | str | None = None,
        resolution: float | None = 0.5,
        top_k: int = 5,
        beam_width: int = 3,
        score_delta: float = 0.05,
    ) -> list[SearchResult]:
        """Associative search with scope filtering and continuous Hopfield resolution dial."""
        q_vec = self.encode_query(query)
        scope_filter = None
        if scope and scope not in (MemoryScope.ALL, "all", None):
            scope_filter = scope.value if isinstance(scope, MemoryScope) else str(scope)

        # Mode A: Continuous Hopfield Associative Retrieval (HAM-R)
        if resolution is not None:
            r_val = float(np.clip(resolution, 0.0, 1.0))
            hop_res = self.tree.query_hopfield(
                q_vec,
                resolution=r_val,
                beam_width=beam_width,
                score_delta=score_delta,
            )

            results: list[SearchResult] = []
            for item_id, weight in zip(hop_res.candidate_ids, hop_res.weights):
                meta = self._item_metadata.get(item_id)
                if not meta:
                    continue
                domain, subdomain, text = meta
                if scope_filter and domain.lower() != scope_filter.lower():
                    continue

                results.append(
                    SearchResult(
                        item_id=item_id,
                        domain=domain,
                        subdomain=subdomain,
                        score=round(float(weight), 4),
                        text=text,
                        energy=round(float(hop_res.final_energy), 4),
                    )
                )

            # Sort by highest attention weight
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # Mode B: Standard Nearest-Neighbor Prefix Routing (HAM-P)
        raw_results = self.tree.query_multi_probe(
            q_vec,
            top_k=top_k * 3 if scope_filter else top_k,
            beam_width=beam_width,
            score_delta=score_delta,
        )

        results = []
        for r in raw_results:
            meta = self._item_metadata.get(r.item_id)
            if not meta:
                continue
            domain, subdomain, text = meta
            if scope_filter and domain.lower() != scope_filter.lower():
                continue

            results.append(
                SearchResult(
                    item_id=r.item_id,
                    domain=domain,
                    subdomain=subdomain,
                    score=round(float(r.score), 4),
                    text=text,
                    energy=None,
                )
            )
            if len(results) >= top_k:
                break

        return results
```

---

### Step 2: Vectorize Candidate Edge Discovery

#### Location: `SmritiAdapter.discover_candidate_edges()` (Lines 284–325)

**The Defect:**
Nested Python `for` loops computing Euclidean distance pairwise across Python lists.

**The Remediation:**
Vectorize using NumPy matrix multiplication (or batch query). Since all vectors in Smriti are strictly unit-normalized, the cosine similarity matrix is simply:
$$S = X_{\text{source}} \cdot X_{\text{target}}^\top$$
The associative distance is:
$$D_{ij} = \sqrt{2 \cdot (1 - S_{ij})}$$

```python
    def discover_candidate_edges(
        self,
        source_domain: str = "Krama",
        target_domain: str = "Vyapar",
        threshold: float = 0.82,
        max_edges: int = 10,
    ) -> list[CandidateEdge]:
        """Vectorized discovery of cross-domain candidate edge hypotheses."""
        source_items = [
            (iid, dom, sub, self._item_vectors[iid])
            for iid, (dom, sub, _) in self._item_metadata.items()
            if dom.lower() == source_domain.lower() and iid in self._item_vectors
        ]
        target_items = [
            (iid, dom, sub, self._item_vectors[iid])
            for iid, (dom, sub, _) in self._item_metadata.items()
            if dom.lower() == target_domain.lower() and iid in self._item_vectors
        ]

        if not source_items or not target_items:
            return []

        S_mat = np.stack([item[3] for item in source_items])  # (N, 768)
        T_mat = np.stack([item[3] for item in target_items])  # (M, 768)

        # Dot product matrix (N, M)
        sim_matrix = np.clip(np.dot(S_mat, T_mat.T), -1.0, 1.0)
        dist_matrix = np.sqrt(np.maximum(2.0 * (1.0 - sim_matrix), 0.0))

        # Find entries below distance threshold
        candidates: list[CandidateEdge] = []
        rows, cols = np.where(dist_matrix <= threshold)
        for r, c in zip(rows, cols):
            dist = float(dist_matrix[r, c])
            conf = float(1.0 / (1.0 + dist))
            s_id, s_dom, s_sub, _ = source_items[r]
            t_id, t_dom, t_sub, _ = target_items[c]
            candidates.append(
                CandidateEdge(
                    source_id=s_id,
                    source_domain=s_dom,
                    source_subdomain=s_sub,
                    target_id=t_id,
                    target_domain=t_dom,
                    target_subdomain=t_sub,
                    associative_distance=round(dist, 4),
                    confidence_score=round(conf, 4),
                )
            )

        candidates.sort(key=lambda c: c.associative_distance)
        return candidates[:max_edges]
```

---

### Step 3: Upgrade `SmritiAdapter.insert()` for Multimodal Media Payloads

#### Location: `SmritiAdapter.insert()`

**The Remediation:**
Allow attaching media items (`diagram`, `audio`, `spec`) to EKG entities:

```python
    def insert(
        self,
        item_id: str,
        domain: str,
        subdomain: str,
        data: str | dict[str, Any],
        media: dict[str, Any] | None = None,
        vector: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Insert a node or multimodal artifact into associative memory."""
        if isinstance(data, dict):
            text = record_to_text(data, domain, subdomain)
        else:
            text = str(data)

        if vector is None:
            # Check if multimodal pipeline is available
            if media:
                payload = {
                    "text": text,
                    "domain": domain,
                    "subdomain": subdomain,
                    "parent_entity_id": item_id,
                    **media,
                }
                if hasattr(self.pipeline, "encode_multimodal_batch"):
                    emb = self.pipeline.encode_multimodal_batch([payload])[0]
                else:
                    emb = self.pipeline.encode_text_batch([text])[0]
            else:
                emb = self.pipeline.encode_text_batch([text])[0]
            vec = np.array(emb.d2_768, dtype=np.float32)
        else:
            vec = np.array(vector, dtype=np.float32)

        self.tree.insert(item_id, vec)
        self._item_metadata[item_id] = (domain, subdomain, text)
        self._item_vectors[item_id] = vec
        return vec
```

---

### Step 4: Verification in Naksha

Run the following test commands inside `/Users/anchormath/Code/self/naksha`:

```bash
# 1. Run unit tests for Smriti adapter
pytest tests/test_smriti_adapter.py -v

# 2. Run API associative endpoints
pytest tests/test_api_associative.py -v

# 3. Verify Review Engine State Injection
pytest tests/test_review_engine.py -v

# 4. Run entire Naksha test suite
pytest -m "not eval"
```
