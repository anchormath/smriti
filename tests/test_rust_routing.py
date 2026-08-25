"""Unit and integration tests for Rust HAM-P Tree Engine (smriti-core)."""

import numpy as np
import pytest
from smriti.tree.hamp_tree import FlatMemorySearch, HAMPTreeWrapper
import smriti_core


def generate_cluster_data(n_clusters=8, points_per_cluster=50, dim=768, noise=0.05):
    """Generate synthetic clustered data on unit hypersphere."""
    np.random.seed(42)
    vectors = []
    ids = []
    # Generate random cluster centers
    centers = np.random.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    for c_idx in range(n_clusters):
        for p_idx in range(points_per_cluster):
            v = centers[c_idx] + np.random.randn(dim).astype(np.float32) * noise
            v /= np.linalg.norm(v)
            item_id = f"proj_c{c_idx}_p{p_idx}"
            vectors.append(v)
            ids.append(item_id)

    return ids, np.array(vectors, dtype=np.float32)


def test_smriti_core_module_import():
    """Verify PyO3 module and classes exist."""
    assert hasattr(smriti_core, "HAMPTree")
    assert hasattr(smriti_core, "QueryResult")
    assert hasattr(smriti_core, "TreeStats")
    assert hasattr(smriti_core, "dot_product_py")


def test_hamp_tree_basic_insertion_and_query():
    """Test basic streaming insertion and nearest neighbor retrieval."""
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=20)
    ids, vectors = generate_cluster_data(n_clusters=4, points_per_cluster=15, dim=768)

    for item_id, vec in zip(ids, vectors):
        tree.insert(item_id, vec)

    assert len(tree) == 60
    stats = tree.stats()
    assert stats.num_items == 60
    assert stats.num_leaves >= 3

    # Query with exact vector of first item
    results = tree.query(vectors[0], top_k=5, beam_width=2)
    assert len(results) >= 1
    assert results[0].item_id == ids[0]
    assert results[0].score > 0.99


def test_running_centroid_consistency():
    """Verify running-mean centroids deviate by < 1e-4 from exact batch centroids."""
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=30)
    ids, vectors = generate_cluster_data(n_clusters=6, points_per_cluster=40, dim=768)

    tree.insert_batch(list(zip(ids, vectors)))

    max_deviation = tree.validate_running_centroids()
    print(f"Max running centroid deviation: {max_deviation:.6e}")
    assert max_deviation < 1e-4, f"Centroid deviation {max_deviation} exceeds threshold 1e-4"


def test_split_stability_and_leaf_balance():
    """Streaming insertion of 2,000+ nodes triggers dynamic splits with balanced leaves."""
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=100)
    ids, vectors = generate_cluster_data(n_clusters=16, points_per_cluster=150, dim=768)  # 2,400 items

    tree.insert_batch(list(zip(ids, vectors)))

    assert len(tree) == 2400
    stats = tree.stats()
    leaf_sizes = tree.get_leaf_sizes()

    assert stats.num_leaves >= 24
    assert stats.max_leaf_size <= 100
    # Average leaf size should be well-distributed
    assert 30 <= stats.avg_leaf_size <= 100
    # No empty leaves
    assert all(s > 0 for s in leaf_sizes)


def test_prefix_routing_recall_vs_flat_search():
    """Verify prefix routing achieves >= 95% Top-5 recall and >= 85% Top-1 recall."""
    ids, vectors = generate_cluster_data(n_clusters=10, points_per_cluster=100, dim=768)  # 1,000 items
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=100)
    tree.insert_batch(list(zip(ids, vectors)))

    flat_search = FlatMemorySearch(ids, vectors)

    # Sample 100 test queries
    np.random.seed(123)
    query_indices = np.random.choice(len(ids), size=100, replace=False)

    top1_matches = 0
    top5_matches = 0
    top_k = 5

    for idx in query_indices:
        q = vectors[idx] + np.random.randn(768).astype(np.float32) * 0.02
        q /= np.linalg.norm(q)

        # Ground truth flat search
        flat_res = flat_search.query(q, top_k=top_k)
        flat_top1 = flat_res[0][0]
        flat_top5_set = {r[0] for r in flat_res}

        # HAM-P Tree query (beam_width=2)
        tree_res = tree.query(q, top_k=top_k, beam_width=2)
        tree_top1 = tree_res[0].item_id if tree_res else None
        tree_top5_set = {r.item_id for r in tree_res}

        if tree_top1 == flat_top1:
            top1_matches += 1

        # Recall@5: fraction of ground truth top-5 retrieved
        overlap = len(flat_top5_set.intersection(tree_top5_set))
        top5_matches += overlap / top_k

    top1_recall = top1_matches / 100.0
    top5_recall = top5_matches / 100.0

    print(f"Top-1 Recall: {top1_recall:.2%}, Top-5 Recall: {top5_recall:.2%}")
    assert top1_recall >= 0.85, f"Top-1 Recall {top1_recall:.2%} below 85%"
    assert top5_recall >= 0.95, f"Top-5 Recall {top5_recall:.2%} below 95%"
