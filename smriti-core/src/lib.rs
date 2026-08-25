//! PyO3 bindings for the smriti-core HAM-P Rust Engine.

pub mod kmeans;
pub mod matrix;
pub mod node;
pub mod router;
pub mod split;
pub mod tree;

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
    ///
    /// Parameters:
    /// - `d0`: Layer 0 root routing dimension (e.g. 64)
    /// - `d1`: Layer 1 subcategory routing dimension (e.g. 128)
    /// - `d2`: Layer 2 leaf memory dimension (e.g. 768)
    /// - `max_leaf_capacity`: Maximum memory count before triggering 2-means split (e.g. 100)
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

    /// Query batch in parallel across CPU cores using Rayon.
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
    m.add_class::<PyTreeStats>()?;
    m.add_function(wrap_pyfunction!(dot_product_py, m)?)?;
    Ok(())
}
