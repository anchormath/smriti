# Sprint 4 Implementation Plan: Naksha Associative Memory Integration

**Sprint:** Smriti Sprint 4 / Naksha Sprint 9  
**Goal:** Deliver end-to-end integration of Smriti's associative memory engine (HAM-P/HAM-R) into Naksha's EKG architecture.

---

## 1. Smriti Crate & Eval Baseline

- **Status**: Verified in `smriti/.venv` with `scripts/eval_naksha_seed.py`.
- **Artifacts to track**:
  - `scripts/eval_naksha_seed.py`: Deterministic and Nomic embedding evaluations on Naksha seed graph.
  - `sprints/HANDOFF_naksha_eval.md`: Documented resolution of build/import environment.

---

## 2. Naksha Memory Adapter Subsystem (`naksha/memory/`)

- **`naksha/memory/__init__.py`**:
  - Export `SmritiAdapter`, `MemoryScope`, `SearchResult`, `ResolutionSweepResult`.
- **`naksha/memory/smriti_adapter.py`**:
  - Class `SmritiAdapter`:
    - `load_from_seed_data(seed_dir)` / `ingest_records(records)`
    - `insert(item_id, domain, subdomain, text_or_dict, vector=None)`
    - `search(query, scope=None, resolution=0.5, top_k=5, beam_width=3) -> list[SearchResult]`
    - `sweep_resolution(query, resolutions=[0.0, 0.25, 0.5, 0.75, 1.0]) -> list[ResolutionSweepResult]`
    - `discover_candidate_edges(source_scope="Krama", target_scope="Vyapar", threshold=0.82) -> list[dict]`
    - `get_topology_stats() -> dict`
  - Support both direct `smriti.tree.hamp_tree.HAMPTreeWrapper` and standalone Python fallback if PyO3 module is in a separate venv/path.

---

## 3. Query & Review Engine Integrations

- **`naksha/engine/associative.py`**:
  - Singleton manager `get_smriti_adapter()` for lazy initialisation and seed data ingestion.
- **`naksha/engine/query.py`**:
  - Integrate associative fallback when exact SurrealQL yields empty result or for semantic similarity queries.
- **`naksha/engine/review.py`**:
  - Use `smriti_adapter.search(..., scope="Kala", resolution=0.7)` to find peer precedents and skill analogies to ground performance review narratives.

---

## 4. API Layer (`naksha/api/main.py`)

- Endpoints:
  - `POST /associative/search`: Semantic / associative query with resolution dial.
  - `GET /associative/candidates`: Propose cross-domain candidate edges.
  - `GET /associative/stats`: HAM-P tree depth, leaves, and item count.

---

## 5. Verification & Tests

- **Unit Tests**:
  - `tests/unit/test_smriti_adapter.py`: Ingestion, self-recall (100%), domain filtering, resolution dial energy sweep, candidate edge discovery.
  - `tests/unit/test_review_engine.py`: Verify review generation with associative context.
- **API Integration Tests**:
  - `tests/integration/test_associative_api.py`: Test `/associative/search`, `/associative/candidates`, `/associative/stats`.
- **Full Suite Run**:
  - `pytest -m "not eval"` in `naksha`.
