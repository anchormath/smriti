# Sprint 4: Naksha Associative Memory Integration

**Timeline:** Sprint 4 (Smriti) / Sprint 9 (Naksha)  
**Epic:** Productionization of HAM-P/HAM-R for Naksha Enterprise Knowledge Graph  
**Goal:** Deliver a production-ready Naksha integration for Smriti: wrap the PyO3 HAM-P/HAM-R core into a Naksha memory adapter, evaluate on real EKG seed data, integrate into Naksha's query and review pipelines, support cross-domain candidate edge discovery, and expose REST endpoints.

---

## 1. Context & Objectives

Smriti was initially developed as a standalone Rust/PyO3 associative memory engine (HAM-P partitioning + continuous Hopfield HAM-R dynamics with a resolution dial $R \in [0, 1]$).

Following the successful evaluation spike on Naksha's 83-node seed graph (`scripts/eval_naksha_seed.py`), this sprint integrates Smriti directly as an associative memory provider for Naksha:

- [x] **Real Seed Evaluation & Plumbing Baseline:** Verify tree topology (4 leaves, depth 3), 100% self-recall (10/10), 1.00 domain purity, and Hopfield energy monotonicity across $R \in [0, 1]$ on real Naksha data.
- [ ] **Naksha Memory Subsystem (`naksha/memory/`):** Create high-level adapter (`SmritiAdapter`) with support for MRL embeddings, fallback mode, scope filtering (Sangh/Kala/Krama/Vyapar), resolution sweeps, and candidate edge discovery.
- [ ] **Hot-Path Tooling & Fallback:** Integrate associative search into Naksha's natural language query engine (`naksha/engine/query.py`) as a sibling/fallback tool when exact graph traversals are empty.
- [ ] **State Injection in Performance Reviews:** Enrich review narrative generation (`naksha/engine/review.py`) using the resolution dial ($R \approx 0.7$) to retrieve contextual peer and skill precedents.
- [ ] **Candidate Edge Discovery:** Expose cross-module associative link hypotheses (e.g., Krama services $\leftrightarrow$ Vyapar product features).
- [ ] **API Endpoints:** Surface `/associative/search`, `/associative/candidates`, and `/associative/stats` in Naksha FastAPI.
- [ ] **Automated Test Suite:** Comprehensive unit & integration tests covering adapter lifecycle, API endpoints, and review engine enrichment.

---

## 2. Architecture & Data Flow

```
   ┌─────────────────────────────────────────────────────────────┐
   │                       Naksha API Layer                      │
   │      POST /ask     POST /review/{id}/draft    POST /associative  │
   └───────────────┬─────────────────┬─────────────────────┬─────┘
                   │                 │                     │
                   ▼                 ▼                     ▼
         ┌──────────────────┐ ┌───────────────┐  ┌───────────────────┐
         │ DSPy Query Engine│ │ Review Drafter│  │ Associative API   │
         └─────────┬────────┘ └───────┬───────┘  └─────────┬─────────┘
                   │                  │                    │
                   │ (Exact Traversal)│ (State Injection)  │ (Direct / Dial)
                   ▼                  ▼                    ▼
         ┌──────────────────┐ ┌──────────────────────────────────────┐
         │   SurrealDB EKG  │ │      Naksha Smriti Adapter           │
         │ (Deterministic)  │ │ (MRL Embeddings + Scope Router)      │
         └──────────────────┘ └──────────────────┬───────────────────┘
                                                 │
                                                 ▼
                                ┌────────────────────────────────────┐
                                │     smriti-core (Rust / PyO3)      │
                                │   HAM-P Tree + Hopfield (HAM-R)    │
                                └────────────────────────────────────┘
```

---

## 3. Stories & Tasks

- **S4.1 Seed Evaluation & Handoff Tracking (`smriti/`):**
  - Track `scripts/eval_naksha_seed.py` and `sprints/HANDOFF_naksha_eval.md`.
  - Ensure clean builds and imports for `smriti_core` and `smriti`.

- **S4.2 Naksha Memory Subsystem (`naksha/memory/`):**
  - `naksha/memory/smriti_adapter.py`: Ingestion, tree construction, query with resolution dial, scope routing, and edge discovery.
  - Support automatic node text formatting for all 4 Naksha domains (Sangh, Kala, Krama, Vyapar).

- **S4.3 Query Engine Integration (`naksha/engine/query.py`):**
  - Connect associative search when exact SurrealQL query matches 0 rows or for similarity queries.

- **S4.4 Review Engine State Injection (`naksha/engine/review.py`):**
  - Inject associative peer evidence and similar project impact clusters into review drafts.

- **S4.5 API Endpoints (`naksha/api/main.py`):**
  - Add `/associative/search`, `/associative/candidates`, and `/associative/stats`.

- **S4.6 Testing & Validation (`tests/`):**
  - Unit tests for adapter, review engine integration, and FastAPI integration tests.

---

## 4. Exit Criteria

1. `eval_naksha_seed.py` committed and clean test suite passing in both `smriti` and `naksha`.
2. `SmritiAdapter` seamlessly handles indexing, querying, resolution sweep, and edge discovery.
3. Review narrative generation incorporates associative peer context.
4. FastAPI exposes all associative memory routes with valid responses.
