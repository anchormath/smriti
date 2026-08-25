# Sprint 3: Rust Hopfield Dynamics, Resolution Dial & Evaluation

**Timeline:** Weeks 5–6  
**Epic:** Unified HAM-P + HAM-R Memory Dynamics & Pattern Completion (`smriti-core`)  
**Goal:** Embed the **Rust** HAM-R Resolution Dial mechanics inside HAM-P leaf pools, validate Hopfield energy landscape regimes across dial settings ($R \in [0.0, 1.0]$), and evaluate associative pattern completion for predicting RICE metrics on incomplete project drafts via PyO3.

---

## 1. Objectives & Scope

- [x] **Rust Leaf-Level HAM-R Integration:** Embed the continuous Hopfield Resolution Dial ($\beta(R)$) inside partitioned Rust leaf buffers $\Xi_{\text{leaf}}^{(2)}$ with SIMD-accelerated dot products and numerically stable softmax.
- [x] **Energy Attractor Validation:** Measure state entropy and pattern convergence across the 3 regimes:
  - $R < 0.3$: Broad Leaf Cluster Archetype
  - $0.3 \le R \le 0.7$: Metastable Composite Forecast
  - $R > 0.7$: Exact Historical Precedent Recall
- [x] **RICE Pattern Completion (Imputation Experiment):** Test the engine's ability to predict missing project metrics (Effort, Impact, Tech Stack) when queried with only high-level problem descriptions and Reach numbers.
- [x] **End-to-End Evaluation Report:** Compile benchmark comparisons covering query latency, FLOPs, memory footprint, and metric imputation accuracy.

---

## 2. Hopfield Mathematical Mechanics in Rust

```
Incoming Query x (Routed to Leaf Pool j in Rust Engine)
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Rust Leaf Buffer: Ξ_leaf in Column-Major Vec<f32>           │
│ Resolution Dial: R in [0.0, 1.0]                            │
│                                                             │
│ 1. Inverse Temperature:                                     │
│    β(R) = β_min + R * (β_max - β_min)                       │
│    (e.g., β_min = 2.0, β_max = 50.0)                        │
│                                                             │
│ 2. SIMD Leaf Similarities:                                  │
│    S = (Ξ_leaf)^T * x[:768] in R^(M_leaf)                   │
│                                                             │
│ 3. Numerically Stable Softmax:                              │
│    max_s = S.iter().fold(NEG_INFINITY, max)                 │
│    a_i = exp(β(R) * (S_i - max_s)) / sum(exp(...))          │
│                                                             │
│ 4. Output Reconstruction (Zero Heap Allocation):            │
│    x_out = Ξ_leaf * a                                       │
│                                                             │
│ 5. Metric Association (RICE Projection):                    │
│    Estimated_Effort = sum(a_i * Stored_Effort_i)            │
│    Estimated_Impact = sum(a_i * Stored_Impact_i)            │
└─────────────────────────────────────────────────────────────┘
```

```rust
// smriti-core/src/hopfield.rs
impl LeafNode {
    pub fn retrieve(&self, query: &[f32], r: f32, beta_min: f32, beta_max: f32) -> (Vec<f32>, Vec<f32>) {
        let beta = beta_min + r.clamp(0.0, 1.0) * (beta_max - beta_min);
        let similarities = self.compute_similarities(query);
        let weights = softmax_stable(&similarities, beta);
        let reconstructed = self.blend_vectors(&weights);
        (reconstructed, weights)
    }
}
```

---

## 3. The Resolution Dial ($R$) Experiment Matrix

| Dial Setting ($R$) | $\beta(R)$ Regime | Expected Hopfield State | RICE Application Use-Case |
| :--- | :--- | :--- | :--- |
| **$R = 0.15$** | Low ($\beta \approx 8$) | **Global/Leaf Fixed Point:** Smooth blend across all $\sim 50$ leaf items. High entropy. | **Executive Baseline:** "What does a generic caching project typically cost in effort/impact across the company?" |
| **$R = 0.50$** | Medium ($\beta \approx 25$) | **Metastable State:** Weighted average over top 3–5 nearest sub-cluster items. | **Consensus Estimate:** "Based on the 3 most similar distributed caching projects, what is our expected effort?" |
| **$R = 0.95$** | High ($\beta \approx 48$) | **Isolated Fixed Point:** Snaps to single top-1 item with near-zero error. Low entropy. | **Exact Historical Precedent:** "Find the exact project spec and actual post-mortem effort for the Bazel S3 project." |

---

## 4. Associative Pattern Completion Experiment (RICE Imputation)

### Experimental Protocol
1. **Holdout Test Set:** 500 synthetic project nodes withheld from indexation.
2. **Partial Query Masking:**
   - **Input Provided:** Title, Problem Statement, Target Audience, Reach Value.
   - **Masked Targets:** Effort (Person-weeks), Impact Score, Confidence Tier, Tech Stack Tags.
3. **Execution:**
   - Embed partial query with `nomic-embed-text-v1.5` in Python.
   - Pass vector to `smriti_core.HAMPTree.retrieve(query, r)` via PyO3.
   - Retrieve blended leaf memory representations and association weights $a$.
   - Compute weighted estimate for Effort and Impact:
     $$\hat{E} = \sum_{i=1}^{M_{\text{leaf}}} a_i \cdot E_i, \quad \hat{I} = \sum_{i=1}^{M_{\text{leaf}}} a_i \cdot I_i$$
4. **Evaluation Metrics:**
   - **Effort Estimation:** Mean Absolute Percentage Error (MAPE) & $R^2$ score.
   - **Impact Classification:** Accuracy / Macro-F1 across 5 tiers.
   - **Tech Stack Recall:** Jaccard similarity between predicted associative tags and ground truth.

---

## 5. Sprint Tasks & Deliverables

| Task ID | Task Description | Est. Days | Tooling | Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **S3-T1** | Implement `LeafHopfieldEngine` with SIMD dot products and softmax in Rust | 2 | Rust (`std::simd`) | `smriti-core/src/hopfield.rs` |
| **S3-T2** | Integrate Hopfield retrieval & Resolution Dial into `HAMPTree` | 2 | Rust | `smriti-core/src/engine.rs` |
| **S3-T3** | Expose retrieval and association weights via PyO3 | 1 | PyO3 / Maturin | `smriti-core/src/lib.rs` |
| **S3-T4** | Build Resolution Dial ($R$) entropy & attractor verification test | 2 | Python / Matplotlib / Scipy | `benchmarks/test_resolution_dial.py` |
| **S3-T5** | Build RICE pattern completion / metric imputation evaluation pipeline | 3 | Python / Pandas / Scikit-learn | `eval/rice_imputation_eval.py` |
| **S3-T6** | Produce end-to-end benchmarking report & architectural findings | 2 | Markdown / Plots | `docs/ham_p_r_evaluation_report.md` |

---

## 6. Acceptance Criteria

1. **Attractor Verification:**
   - At $R < 0.2$, normalized state entropy $> 0.70$ (multi-item blend).
   - At $R > 0.8$, top-1 weight $a_{\max} > 0.95$ (isolated attractor snap).
2. **RICE Prediction Accuracy:**
   - Effort imputation achieves $\text{MAPE} < 20\%$ against holdout test cases at $R = 0.5$.
   - Impact tier classification achieves $\ge 80\%$ top-1 accuracy.
3. **End-to-End Latency:** Total tree routing + Hopfield reconstruction takes $< 0.25\text{ ms}$ on CPU for $M=4,000$ items in Rust.
4. **Comprehensive Documentation:** Full evaluation report published with charts detailing error rates, latency vs. flat search, and resolution dial transition curves.

