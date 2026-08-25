//! PyO3 bindings for the smriti-core HAM-P Rust Engine.

pub mod hopfield;
pub mod kmeans;
pub mod matrix;
pub mod node;
pub mod router;
pub mod split;
pub mod tree;

use hopfield::HopfieldResult;
use pyo3::prelude::*;
use tree::HAMPTree;

/// Python query retrieval result item.
#[pyclass(name = "QueryResult")]
#[derive(Clone)]
pub struct PyQueryResult {
    #[pyo3(get)]
    pub item_id: String,
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl PyQueryResult {
    fn __repr__(&self) -> String {
        format!("QueryResult(item_id='{}', score={:.4})", self.item_id, self.score)
    }
}

/// Python continuous Hopfield associative retrieval result.
#[pyclass(name = "HopfieldResult")]
#[derive(Clone)]
pub struct PyHopfieldResult {
    #[pyo3(get)]
    pub retrieved_vector: Vec<f32>,
    #[pyo3(get)]
    pub final_energy: f32,
    #[pyo3(get)]
    pub energy_trajectory: Vec<f32>,
    #[pyo3(get)]
    pub weights: Vec<f32>,
    #[pyo3(get)]
    pub candidate_ids: Vec<String>,
}

#[pymethods]
impl PyHopfieldResult {
    fn __repr__(&self) -> String {
        format!(
            "HopfieldResult(dim={}, final_energy={:.4}, steps={}, candidates={})",
            self.retrieved_vector.len(),
            self.final_energy,
            self.energy_trajectory.len() - 1,
            self.candidate_ids.len()
        )
    }
}

/// Python tree statistics summary.
#[pyclass(name = "TreeStats")]
#[derive(Clone)]
pub struct PyTreeStats {
    #[pyo3(get)]
    pub num_nodes: usize,
    #[pyo3(get)]
    pub num_leaves: usize,
    #[pyo3(get)]
    pub num_items: usize,
    #[pyo3(get)]
    pub depth: usize,
    #[pyo3(get)]
    pub avg_leaf_size: f32,
    #[pyo3(get)]
    pub min_leaf_size: usize,
    #[pyo3(get)]
    pub max_leaf_size: usize,
}

#[pymethods]
impl PyTreeStats {
    fn __repr__(&self) -> String {
        format!(
            "TreeStats(items={}, leaves={}, depth={}, avg_leaf_size={:.1}, min_leaf={}, max_leaf={})",
            self.num_items,
            self.num_leaves,
            self.depth,
            self.avg_leaf_size,
            self.min_leaf_size,
            self.max_leaf_size
        )
    }
}

/// Python-exposed Hierarchical Partitioned Associative Memory Tree.
#[pyclass(name = "HAMPTree")]
pub struct PyHAMPTree {
    inner: HAMPTree,
}

#[pymethods]
impl PyHAMPTree {
    /// Instantiate a new HAM-P Tree.
    #[new]
    #[pyo3(signature = (d0=64, d1=128, d2=768, max_leaf_capacity=100))]
    fn new(d0: usize, d1: usize, d2: usize, max_leaf_capacity: usize) -> Self {
        Self {
            inner: HAMPTree::new(d0, d1, d2, max_leaf_capacity),
        }
    }

    /// Insert a project vector online.
    fn insert(&mut self, item_id: &str, vector: Vec<f32>) {
        self.inner.insert(item_id, &vector);
    }

    /// Insert a batch of items sequentially.
    fn insert_batch(&mut self, items: Vec<(String, Vec<f32>)>) {
        for (id, vec) in items {
            self.inner.insert(&id, &vec);
        }
    }

    /// Query Top-K nearest neighbors for a single query vector using prefix routing.
    #[pyo3(signature = (query_vector, top_k=5, beam_width=1))]
    fn query(&self, query_vector: Vec<f32>, top_k: usize, beam_width: usize) -> Vec<PyQueryResult> {
        self.inner
            .query(&query_vector, top_k, beam_width)
            .into_iter()
            .map(|r| PyQueryResult {
                item_id: r.item_id,
                score: r.score,
            })
            .collect()
    }

    /// Query Top-K nearest neighbors using multi-probe beam search with delta margin.
    #[pyo3(signature = (query_vector, top_k=5, beam_width=3, score_delta=0.05))]
    fn query_multi_probe(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        beam_width: usize,
        score_delta: f32,
    ) -> Vec<PyQueryResult> {
        self.inner
            .query_multi_probe(&query_vector, top_k, beam_width, score_delta)
            .into_iter()
            .map(|r| PyQueryResult {
                item_id: r.item_id,
                score: r.score,
            })
            .collect()
    }

    /// Query continuous Hopfield associative retrieval using the Resolution Dial R in [0.0, 1.0].
    #[pyo3(signature = (query_vector, resolution=0.5, beam_width=3, score_delta=0.05, max_steps=5))]
    fn query_hopfield(
        &self,
        query_vector: Vec<f32>,
        resolution: f32,
        beam_width: usize,
        score_delta: f32,
        max_steps: usize,
    ) -> PyHopfieldResult {
        let res = self.inner.query_hopfield(
            &query_vector,
            resolution,
            beam_width,
            score_delta,
            max_steps,
        );
        PyHopfieldResult {
            retrieved_vector: res.retrieved_vector,
            final_energy: res.final_energy,
            energy_trajectory: res.energy_trajectory,
            weights: res.weights,
            candidate_ids: res.candidate_ids,
        }
    }

    /// Batch parallel query evaluation across CPU cores using Rayon.
    #[pyo3(signature = (queries, top_k=5, beam_width=1))]
    fn query_batch(
        &self,
        queries: Vec<Vec<f32>>,
        top_k: usize,
        beam_width: usize,
    ) -> Vec<Vec<PyQueryResult>> {
        self.inner
            .query_batch(&queries, top_k, beam_width)
            .into_iter()
            .map(|list| {
                list.into_iter()
                    .map(|r| PyQueryResult {
                        item_id: r.item_id,
                        score: r.score,
                    })
                    .collect()
            })
            .collect()
    }

    /// Batch parallel Hopfield queries across CPU cores using Rayon.
    #[pyo3(signature = (queries, resolution=0.5, beam_width=3, score_delta=0.05, max_steps=5))]
    fn query_hopfield_batch(
        &self,
        queries: Vec<Vec<f32>>,
        resolution: f32,
        beam_width: usize,
        score_delta: f32,
        max_steps: usize,
    ) -> Vec<PyHopfieldResult> {
        self.inner
            .query_hopfield_batch(
                &queries,
                resolution,
                beam_width,
                score_delta,
                max_steps,
            )
            .into_iter()
            .map(|res: HopfieldResult| PyHopfieldResult {
                retrieved_vector: res.retrieved_vector,
                final_energy: res.final_energy,
                energy_trajectory: res.energy_trajectory,
                weights: res.weights,
                candidate_ids: res.candidate_ids,
            })
            .collect()
    }

    /// Compute the continuous Lyapunov energy of a state vector relative to the indexed memories.
    #[pyo3(signature = (state_vector, beta=1.0))]
    fn compute_lyapunov_energy(&self, state_vector: Vec<f32>, beta: f32) -> f32 {
        self.inner.compute_lyapunov_energy(&state_vector, beta)
    }

    /// Lazy re-centering task to recompute exact centroids directly from leaves.
    fn recenter(&mut self) {
        self.inner.recenter();
    }

    /// Retrieve tree statistics.
    fn stats(&self) -> PyTreeStats {
        let s = self.inner.stats();
        PyTreeStats {
            num_nodes: s.num_nodes,
            num_leaves: s.num_leaves,
            num_items: s.num_items,
            depth: s.depth,
            avg_leaf_size: s.avg_leaf_size,
            min_leaf_size: s.min_leaf_size,
            max_leaf_size: s.max_leaf_size,
        }
    }

    /// List all individual leaf node sizes.
    fn get_leaf_sizes(&self) -> Vec<usize> {
        self.inner.get_leaf_sizes()
    }

    /// Validate running-mean centroids against exact batch centroids.
    fn validate_running_centroids(&self) -> f32 {
        self.inner.validate_running_centroids()
    }

    /// Total items stored in the tree.
    fn count_items(&self) -> usize {
        self.inner.num_items
    }
}

/// Fast SIMD-accelerated dot product exposed for benchmarking.
#[pyfunction]
fn dot_product_py(a: Vec<f32>, b: Vec<f32>) -> f32 {
    matrix::dot_product(&a, &b)
}

/// Smriti Core PyO3 native module.
#[pymodule]
fn smriti_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHAMPTree>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyHopfieldResult>()?;
    m.add_class::<PyTreeStats>()?;
    m.add_function(wrap_pyfunction!(dot_product_py, m)?)?;
    Ok(())
}
