# Multi-Resolution Hierarchical Associative Memory Engine (HAM-R)

## 1. Theoretical Foundations

### 1.1 From Classical to Modern Continuous Hopfield Networks

Classical Hopfield networks operate on discrete binary states $x \in \{-1, 1\}^N$ governed by quadratic energy functions. Their storage capacity is constrained to approximately $0.14 N$ patterns due to spurious attractors and inter-pattern crosstalk.

Modern Hopfield networks transition to continuous state spaces $x \in \mathbb{R}^d$ and employ non-linear, exponential interaction functions. This modification expands the storage capacity exponentially relative to the feature dimension $d$, enabling the network to store and retrieve high-dimensional continuous representations in a single update step.

### 1.2 Non-Parametric Memory vs. Representation Learning

A critical theoretical distinction lies between **memory physics** and **representation learning**:

- **Non-Parametric Memory Engine:** The core continuous Hopfield dynamics require no stochastic gradient descent (SGD) or backpropagation. Stored patterns are placed directly as continuous column vectors into a memory matrix $\Xi = [\xi_1, \xi_2, \dots, \xi_M] \in \mathbb{R}^{d \times M}$.
- **Latent Space Alignment:** SGD is external to the retrieval mechanism; it is used only when training upstream feature extractors (e.g., Graph Neural Networks or Matryoshka encoders) to project raw domain data into a latent space where vector inner products reflect semantic similarity.

### 1.3 Equivalence to Attention Mechanics and Energy Minima

The continuous update rule for a modern Hopfield network given a query state $x$ is:

$$
x^{\text{new}} = \Xi \operatorname{softmax}(\beta \Xi^\top x)
$$

where $\beta$ represents the inverse temperature (clamping parameter). This update rule is mathematically equivalent to the scaled dot-product attention mechanism found in transformer architectures.

As proven by Ramsauer et al., continuous Hopfield update dynamics exhibit three distinct types of energy minima (attractor fixed points) based on the interaction between pattern similarity and the parameter $\beta$:

1. **Global Fixed Points:** The state converges to a smooth, global average over the entire corpus of stored patterns.
2. **Metastable States:** The state converges to a local, weighted average over a semantically coherent subset (cluster) of stored patterns.
3. **Isolated Fixed Points:** The state converges precisely to an exact, un-blended stored pattern $\xi_i$ with exponentially small retrieval error.

## 2. Architecture & The "Resolution Dial" Engine

### 2.1 Multi-Tiered Memory Layout

To enable multi-resolution access without redundant computation, stored items are split into two parallel, index-aligned matrices:

| **Memory Layer**    | **Matrix Symbol** | **Dimensions** | **Purpose**                                                  |
| ------------------- | ----------------- | -------------- | ------------------------------------------------------------ |
| **Tier 1 (Coarse)** | $\Xi^{(1)}$ (`Tier1_Coarse`) | $d_1 \times M$ | Low-dimensional features capturing structural macro-categories, archetypes, and high-level primitives. |
| **Tier 2 (Fine)**   | $\Xi^{(2)}$ (`Tier2_Fine`)   | $d_2 \times M$ | High-dimensional features capturing granular metrics, exact parameters, and full execution details. |

### 2.2 The Resolution Dial ($R \in [0.0, 1.0]$)

The Resolution Dial $R$ controls the retrieval dynamics by dynamically mapping to the inverse temperature parameter $\beta$:

$$
\beta(R) = \beta_{\min} + R \cdot (\beta_{\max} - \beta_{\min})
$$

Changing $R$ dynamically reshapes the energy landscape, steering retrieval across the three fundamental energy minima regimes:

```
Dial Setting (R) | Energy Landscape Regime | Operational Output
-----------------|-------------------------|---------------------------------------------
0.0 <= R < 0.3   | Global Fixed Point      | Broad Category Archetype (Tier 1 blend)
0.3 <= R <= 0.7  | Metastable State        | Cluster Composite Blend (Tier 2 blend)
0.7 < R <= 1.0   | Isolated Fixed Point    | Exact Single Item Retrieval (Tier 2 vector)
```

### 2.3 Retrieval Execution Pipeline

When presented with a query $x^{(1)} \in \mathbb{R}^{d_1}$ and dial setting $R$:

1. **Calculate Active Temperature:** Compute $\beta(R) = \beta_{\min} + R \cdot (\beta_{\max} - \beta_{\min})$.
2. **Compute Tier 1 Similarities:** Evaluate dot products against all $M$ coarse patterns: $S = (\Xi^{(1)})^\top x^{(1)} \in \mathbb{R}^M$.
3. **Compute Continuous Weights:** Apply softmax with clamping parameter $\beta(R)$: $a = \operatorname{softmax}(\beta(R) \cdot S)$.
4. **Multi-Resolution Output Generation:**
   - **If $R < 0.3$ (Low Resolution):** Compute coarse prototype $x_{\text{out}} = \Xi^{(1)} a$.
   - **If $0.3 \le R \le 0.7$ (Medium Resolution):** Compute composite fine estimate $x_{\text{out}} = \Xi^{(2)} a$.
   - **If $R > 0.7$ (High Resolution):** Extract top memory index $i^* = \operatorname{argmax}(a)$ and return isolated item $x_{\text{out}} = \Xi_{:, i^*}^{(2)}$.

## 3. Rust Implementation Specification

### 3.1 Design Principles

- **Zero Allocation on Retrieval:** Memory matrices are allocated as contiguous arrays (`Vec<f32>`) during initialization to maximize CPU L1/L2 cache locality.
- **Parallel Execution:** Matrix-vector multiplications across $M$ memory patterns are parallelized across threads using `rayon`.
- **Numerical Stability:** Softmax calculations subtract $\max(S)$ prior to exponentiation to prevent floating-point overflow at high $\beta$ values.

### 3.2 Core Rust Module

```rust
use rayon::prelude::*;

pub struct MemoryTier {
    pub dim: usize,
    /// Memory matrix stored in column-major order (dim x num_memories)
    pub data: Vec<f32>,
}

impl MemoryTier {
    pub fn new(dim: usize, num_memories: usize) -> Self {
        Self {
            dim,
            data: vec![0.0; dim * num_memories],
        }
    }

    #[inline]
    pub fn get_column(&self, col_idx: usize) -> &[f32] {
        let start = col_idx * self.dim;
        &self.data[start..start + self.dim]
    }
}

pub struct HierarchicalAssociativeMemory {
    pub num_memories: usize,
    pub tier1_coarse: MemoryTier,
    pub tier2_fine: MemoryTier,
    pub beta_min: f32,
    pub beta_max: f32,
}

impl HierarchicalAssociativeMemory {
    pub fn new(
        num_memories: usize,
        d1: usize,
        d2: usize,
        beta_min: f32,
        beta_max: f32,
    ) -> Self {
        Self {
            num_memories,
            tier1_coarse: MemoryTier::new(d1, num_memories),
            tier2_fine: MemoryTier::new(d2, num_memories),
            beta_min,
            beta_max,
        }
    }

    /// Executes multi-resolution memory retrieval
    pub fn retrieve(&self, query_coarse: &[f32], r: f32) -> Vec<f32> {
        let r_clamped = r.clamp(0.0, 1.0);
        let beta = self.beta_min + r_clamped * (self.beta_max - self.beta_min);

        // Step 1: Compute inner products against Tier 1 memories in parallel
        let d1 = self.tier1_coarse.dim;
        let mut similarities = vec![0.0f32; self.num_memories];

        similarities
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, sim)| {
                let col = &self.tier1_coarse.data[i * d1..(i + 1) * d1];
                *sim = dot_product(query_coarse, col);
            });

        // Step 2: Numerically stable softmax with temperature
        let weights = softmax_stable(&similarities, beta);

        // Step 3: Multi-resolution reconstruction
        if r_clamped < 0.3 {
            // Mode A: Coarse Category Archetype
            blend_memories(&self.tier1_coarse, &weights)
        } else if r_clamped <= 0.7 {
            // Mode B: Composite Cluster Forecast
            blend_memories(&self.tier2_fine, &weights)
        } else {
            // Mode C: Exact Single Pattern Retrieval
            let top_idx = argmax(&weights);
            self.tier2_fine.get_column(top_idx).to_vec()
        }
    }
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn softmax_stable(logits: &[f32], beta: f32) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let exps: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) * beta).exp())
        .collect();

    let sum_exps: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum_exps).collect()
}

fn blend_memories(tier: &MemoryTier, weights: &[f32]) -> Vec<f32> {
    let dim = tier.dim;
    let mut output = vec![0.0f32; dim];

    for (i, &w) in weights.iter().enumerate() {
        if w > 1e-6 {
            let col = tier.get_column(i);
            for d in 0..dim {
                output[d] += w * col[d];
            }
        }
    }
    output
}

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
```

### 3.3 Dependencies & Integration Blueprint

- **`rayon`:** Used for parallel dot-product calculations across memory columns.
- **`serde` & `bincode`:** Used to serialize/deserialize memory structs for fast loading from disk.
- **FFI / WASM Integration:** The core Rust struct can be exposed to Python via `pyo3` or compiled to WebAssembly (`wasm-bindgen`) for browser/edge environments.



## Notes

 **Hierarchical Associative Memory (HAM)** is a pure, standalone memory engine, equipped with a **Resolution Dial**. By removing deep learning training loops and backpropagation, we can focus entirely on the memory physics and energy dynamics.

### The Core Concept: The "Resolution Dial" ($R \in [0, 1]$)

In standard associative memory, a query vector drops into an energy landscape and snaps to a single nearest memory.

With a **Resolution Dial ($R$)**, you control the shape of the energy landscape itself:

- **At Low Resolution ($R \approx 0.1$):** The energy wells are broad, shallow, and overlapping. The network converges to a **global or metastable state**—a continuous, smooth blend (average archetype) across a category of similar memories.
- **At High Resolution ($R \approx 0.9$):** The energy wells become deep, sharp, and isolated. The network snaps precisely into an **isolated fixed point**—recalling exact, un-blended details of a single specific memory.

### Component 1: Memory Representation Architecture

Instead of storing each memory item as a single flat vector, each memory item $i$ (out of $M$ total stored items) is stored across **two connected abstraction tiers**:

| **Tier**                          | **Matrix Representation**                 | **Role & Features**                                          |
| --------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| **Tier 1 (Coarse / Abstraction)** | $\Xi^{(1)} \in \mathbb{R}^{d_1 \times M}$ | Low-dimensional features ($d_1$). Stores macro-level structures, broad categories, and core prototypes. |
| **Tier 2 (Fine / Detail)**        | $\Xi^{(2)} \in \mathbb{R}^{d_2 \times M}$ | High-dimensional features ($d_2$). Stores granular attributes, exact metrics, and specific details. |

Both tiers share the same column indices ($i = 1, \dots, M$), establishing a direct structural link between a coarse prototype and its fine-grained details.

### Component 2: The Physics of the Dial Levers

The Resolution Dial $R$ dynamically controls two parameters during retrieval:

#### 1. The Inverse Temperature Lever ($\beta$)

The inverse temperature $\beta$ governs how "sharp" or "selective" the association is:

$$
\beta(R) = \beta_{\min} + R \cdot (\beta_{\max} - \beta_{\min})
$$

- **Small $\beta$ ($R \to 0$):** High entropy. The network weighs many stored memories nearly equally, yielding a broad category summary.
- **Large $\beta$ ($R \to 1$):** Low entropy. The network acts as a sharp argmax filter, picking out the single most similar memory.

#### 2. The Abstraction Depth Lever ($d$)

The dial determines whether information stays in Tier 1 or propagates down into Tier 2:

- **Low $R$:** Search dynamics operate exclusively in Tier 1 space ($\Xi^{(1)}$) to save compute and keep the output high-level.
- **High $R$:** Tier 1 dynamics settle first, generating a top-down prior that constrains Tier 2 dynamics ($\Xi^{(2)}$) to retrieve exact details.

### Component 3: Step-by-Step Retrieval Algorithm

When you present an input query vector $x$ to the memory engine alongside a Resolution Dial setting $R \in [0, 1]$, the engine executes the following non-parametric update:

#### Step 1: Inverse Temperature Calculation

Compute the active inverse temperature $\beta(R)$ based on the dial setting.

#### Step 2: Coarse Association (Tier 1 Dynamics)

Extract the coarse features of the query $x^{(1)} \in \mathbb{R}^{d_1}$ and evaluate inner products against all stored coarse memories:

$$
S^{(1)} = (\Xi^{(1)})^\top x^{(1)} \in \mathbb{R}^M
$$

Compute the continuous association weights $a \in \mathbb{R}^M$ across the memory pool:

$$
a = \operatorname{softmax}(\beta(R) \cdot S^{(1)})
$$

#### Step 3: Multi-Resolution Output Generation

- **Mode A: Low Resolution ($R < 0.3$):** The network uses the continuous weights $a$ to construct a coarse prototype $x_{\text{coarse}}$:

  $$
  x_{\text{coarse}} = \Xi^{(1)} \cdot a
  $$

  *Output:* A high-level continuous baseline or category archetype.

- **Mode B: Medium Resolution ($0.3 \le R \le 0.7$):** The network identifies a **metastable cluster**. It uses $a$ to blend the detailed metrics from Tier 2 ($\Xi^{(2)}$) across a small group of highly relevant memories:

  $$
  x_{\text{intermediate}} = \Xi^{(2)} \cdot a
  $$

  *Output:* A balanced composite forecast based on a cluster of similar past items.

- **Mode C: High Resolution ($R > 0.7$):** Because $\beta(R)$ is high, $a$ approaches a one-hot vector. Top-down feedback isolates the single winning memory index $i^* = \operatorname{argmax}(a)$. The network projects down to Tier 2 to perform full, un-blended pattern completion:

  $$
  x_{\text{fine}} = \Xi_{:, i^*}^{(2)}
  $$

  *Output:* The exact full-fidelity recall of a single historical memory.

### Operational Behavior of the Dial

```
Dial Setting (R)    Memory Behavior                     Energy Landscape Shape
----------------------------------------------------------------------------------
0.0 - 0.3           Broad Prototype (Category Level)   Wide, smooth, single global basin
0.3 - 0.7           Cluster Blend (Sub-type Level)     Metastable wells with multiple local minima
0.7 - 1.0           Exact Recall (Item Level)          Sharp, deep, highly isolated wells
```

### Why This Engine is Practical

1. **Zero Training / No Backpropagation:** The memory matrices $\Xi^{(1)}$ and $\Xi^{(2)}$ are populated directly by stacking feature vectors.
2. **Instant Execution:** Memory retrieval consists of basic matrix multiplications and a softmax, executing in milliseconds.
3. **Controllable Behavior:** You can programmatically turn the "dial" $R$ up or down depending on whether your application needs a broad, conservative baseline estimate or a hyper-specific past precedent.

### Why Rust Fits This Blueprint Perfectly

1. **Cache Locality & Memory Control:** In matrix-vector operations like computing inner products $S = \Xi^\top x$, CPU L1/L2 cache misses are the main latency bottleneck. In Rust, you can lay out the Tier 1 ($\Xi^{(1)}$) and Tier 2 ($\Xi^{(2)}$) memory matrices as contiguous, column-major contiguous memory blocks (`Vec<f32>`), guaranteeing optimal memory bandwidth.  

2. **Safe Concurrency (`rayon`):** Evaluating inner products across thousands of stored memory vectors is an embarrassingly parallel operation. Rust's `rayon` crate lets you parallelize dot product computations across CPU threads with zero data-race risk.

3. **Hardware Vectorization (SIMD):** With crates like `ndarray` or `faer` (or Rust's `std::simd`), vector dot products automatically compile down to native AVX2 / AVX-512 CPU instruction sets. This gives you microsecond-level retrieval speeds on standard CPU hardware without needing a GPU cluster.  

   

   