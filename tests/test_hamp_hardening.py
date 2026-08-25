"""Tests for Sprint 2.1: HAM-P Engine Hardening & Continuous Hopfield Dynamics."""

import numpy as np
import pytest
from smriti.tree import HAMPTreeWrapper, FlatMemorySearch


@pytest.fixture
def sample_tree():
    """Create a populated HAM-P tree with 200 synthetic vectors."""
    np.random.seed(42)
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=30)
    items = []
    for i in range(200):
        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)
        items.append((f"proj_{i:04d}", vec))
    tree.insert_batch(items)
    return tree


def test_in_flight_mrl_normalization(sample_tree):
    """Verify that in-flight prefix normalization maintains centroid precision."""
    max_dev = sample_tree.validate_running_centroids()
    assert max_dev < 1e-4, f"Running centroid deviation {max_dev} exceeds tolerance 1e-4"


def test_multi_probe_delta_routing(sample_tree):
    """Verify multi-probe beam search with delta-margin thresholding."""
    np.random.seed(123)
    query = np.random.randn(768).astype(np.float32)
    query /= np.linalg.norm(query)

    # Standard beam search
    res_standard = sample_tree.query(query, top_k=5, beam_width=1)
    assert len(res_standard) == 5

    # Multi-probe with delta margin
    res_multi = sample_tree.query_multi_probe(query, top_k=5, beam_width=3, score_delta=0.1)
    assert len(res_multi) == 5
    # The top-1 score in multi-probe should be >= standard greedy beam
    assert res_multi[0].score >= res_standard[0].score - 1e-6


def test_lyapunov_energy_monotonic_decay(sample_tree):
    """Verify that continuous Hopfield associative retrieval satisfies monotonic energy decay."""
    np.random.seed(999)
    for _ in range(5):
        query = np.random.randn(768).astype(np.float32)
        query /= np.linalg.norm(query)

        hop_res = sample_tree.query_hopfield(
            query,
            resolution=0.6,
            beam_width=3,
            score_delta=0.05,
            max_steps=8,
        )

        assert len(hop_res.retrieved_vector) == 768
        assert len(hop_res.energy_trajectory) >= 1

        # Check strictly non-increasing Lyapunov energy
        for step in range(1, len(hop_res.energy_trajectory)):
            prev_e = hop_res.energy_trajectory[step - 1]
            curr_e = hop_res.energy_trajectory[step]
            assert curr_e <= prev_e + 1e-5, (
                f"Energy increased at step {step}: prev={prev_e}, curr={curr_e}"
            )


def test_resolution_dial_entropy_sweep(sample_tree):
    """Verify that increasing Resolution Dial R decreases softmax attention entropy."""
    np.random.seed(42)
    query = np.random.randn(768).astype(np.float32)
    query /= np.linalg.norm(query)

    # Sweep from low R (broad blend) to high R (sharp recall)
    sweep = sample_tree.sweep_resolution_dial(
        query,
        resolutions=[0.0, 0.25, 0.5, 0.75, 1.0],
        beam_width=3,
        score_delta=0.05,
    )

    entropies = []
    for entry in sweep:
        weights = entry["weights"]
        # Shannon entropy: -sum(p * log(p))
        nz = weights[weights > 1e-12]
        h = -float(np.sum(nz * np.log(nz)))
        entropies.append(h)

    # Entropy should generally decrease as resolution dial R increases
    assert entropies[0] > entropies[-1], (
        f"Low dial entropy ({entropies[0]}) should be greater than high dial entropy ({entropies[-1]})"
    )


def test_lazy_recenter(sample_tree):
    """Verify that calling recenter() cleans up any cumulative drift."""
    dev_before = sample_tree.validate_running_centroids()
    sample_tree.recenter()
    dev_after = sample_tree.validate_running_centroids()

    assert dev_before < 1e-4
    assert dev_after < 1e-6, f"Recentered deviation {dev_after} exceeds 1e-6"


def test_parallel_hopfield_batch(sample_tree):
    """Verify multithreaded Rayon batch Hopfield evaluation."""
    np.random.seed(77)
    queries = np.random.randn(20, 768).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    batch_res = sample_tree.query_hopfield_batch(
        queries,
        resolution=0.7,
        beam_width=2,
        score_delta=0.05,
        max_steps=5,
    )

    assert len(batch_res) == 20
    for res in batch_res:
        assert len(res.retrieved_vector) == 768
        assert not np.isnan(res.final_energy)
