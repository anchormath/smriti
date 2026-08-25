"""Python interface and high-level utilities for HAM-P Multi-Resolution Tree."""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pyarrow.parquet as pq

try:
    import smriti_core
    from smriti_core import HAMPTree as NativeHAMPTree, QueryResult, TreeStats
except ImportError:
    smriti_core = None
    NativeHAMPTree = None
    QueryResult = None
    TreeStats = None


class HAMPTreeWrapper:
    """High-level Python wrapper for the native Rust HAM-P Tree engine.

    Provides multi-resolution prefix routing, online insertion, and batch query support.
    """

    def __init__(
        self,
        d0: int = 64,
        d1: int = 128,
        d2: int = 768,
        max_leaf_capacity: int = 100,
    ) -> None:
        if NativeHAMPTree is None:
            raise ImportError(
                "smriti_core native extension is not available. Please build using `maturin develop`."
            )
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.max_leaf_capacity = max_leaf_capacity
        self._tree = NativeHAMPTree(
            d0=d0,
            d1=d1,
            d2=d2,
            max_leaf_capacity=max_leaf_capacity,
        )

    def insert(self, item_id: str, vector: Union[List[float], np.ndarray]) -> None:
        """Insert a single high-dimensional vector into the HAM-P tree online."""
        if isinstance(vector, np.ndarray):
            vec_list = vector.astype(np.float32).tolist()
        else:
            vec_list = [float(x) for x in vector]
        self._tree.insert(item_id, vec_list)

    def insert_batch(
        self,
        items: List[Tuple[str, Union[List[float], np.ndarray]]],
    ) -> None:
        """Insert a batch of item (id, vector) tuples."""
        native_items = []
        for item_id, vec in items:
            if isinstance(vec, np.ndarray):
                vec_list = vec.astype(np.float32).tolist()
            else:
                vec_list = [float(x) for x in vec]
            native_items.append((item_id, vec_list))
        self._tree.insert_batch(native_items)

    def query(
        self,
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 5,
        beam_width: int = 1,
    ) -> List[QueryResult]:
        """Query Top-K nearest neighbors using top-down prefix routing."""
        if isinstance(query_vector, np.ndarray):
            q_list = query_vector.astype(np.float32).tolist()
        else:
            q_list = [float(x) for x in query_vector]
        return self._tree.query(q_list, top_k, beam_width)

    def query_batch(
        self,
        query_vectors: Union[List[List[float]], np.ndarray],
        top_k: int = 5,
        beam_width: int = 1,
    ) -> List[List[QueryResult]]:
        """Batch parallel query evaluation using native Rayon multithreading."""
        if isinstance(query_vectors, np.ndarray):
            q_list = query_vectors.astype(np.float32).tolist()
        else:
            q_list = [[float(x) for x in q] for q in query_vectors]
        return self._tree.query_batch(q_list, top_k, beam_width)

    def stats(self) -> TreeStats:
        """Get summary statistics for the tree."""
        return self._tree.stats()

    def get_leaf_sizes(self) -> List[int]:
        """Get list of individual leaf node sizes."""
        return self._tree.get_leaf_sizes()

    def validate_running_centroids(self) -> float:
        """Validate running-mean centroids against exact batch centroids.

        Returns maximum L2 deviation across all routing prototypes.
        """
        return self._tree.validate_running_centroids()

    def __len__(self) -> int:
        return self._tree.count_items()

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        vector_column: str = "mrl_d2_768",
        id_column: str = "id",
        max_leaf_capacity: int = 100,
    ) -> HAMPTreeWrapper:
        """Construct and populate a HAM-P tree directly from a Parquet dataset."""
        table = pq.read_table(parquet_path, columns=[id_column, vector_column])
        ids = table[id_column].to_pylist()
        vectors = table[vector_column].to_pylist()

        dim = len(vectors[0])
        tree = cls(d0=64, d1=128, d2=dim, max_leaf_capacity=max_leaf_capacity)

        batch = list(zip(ids, vectors))
        tree.insert_batch(batch)
        return tree


class FlatMemorySearch:
    """Brute-force flat O(M) vector nearest neighbor search for evaluation baseline."""

    def __init__(self, item_ids: List[str], vectors: np.ndarray) -> None:
        self.item_ids = list(item_ids)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # Verify L2 normalization
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-12)

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Compute exhaustive dot products against all M memories."""
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = q / max(np.linalg.norm(q), 1e-12)
        scores = np.dot(self.vectors, q_norm)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.item_ids[i], float(scores[i])) for i in top_indices]

    def query_batch(
        self, query_vectors: np.ndarray, top_k: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """Exhaustive batch matrix multiplication."""
        q = np.asarray(query_vectors, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.maximum(norms, 1e-12)
        scores = np.dot(q, self.vectors.T)  # Shape (N_queries, M)
        results = []
        for i in range(len(q)):
            top_indices = np.argsort(-scores[i])[:top_k]
            results.append(
                [(self.item_ids[idx], float(scores[i, idx])) for idx in top_indices]
            )
        return results

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        vector_column: str = "mrl_d2_768",
        id_column: str = "id",
    ) -> FlatMemorySearch:
        table = pq.read_table(parquet_path, columns=[id_column, vector_column])
        ids = table[id_column].to_pylist()
        vec_list = table[vector_column].to_pylist()
        vectors = np.array(vec_list, dtype=np.float32)
        return cls(item_ids=ids, vectors=vectors)
