"""
Unit tests for MRL embedding pipeline, prefix slicing, and strict L2 normalization.
"""

import numpy as np
from smriti.embeddings.mrl_pipeline import (
    MRLEmbeddingPipeline,
    extract_mrl_slices,
    normalize_l2,
)


def test_normalize_l2():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = normalize_l2(vec)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-6
    assert np.allclose(normed, np.array([0.6, 0.8], dtype=np.float32))


def test_extract_mrl_slices():
    full_vec = np.random.randn(768).astype(np.float32)
    v0, v1, v2 = extract_mrl_slices(full_vec, d0=64, d1=128, d2=768)

    assert v0.shape == (64,)
    assert v1.shape == (128,)
    assert v2.shape == (768,)

    assert abs(np.linalg.norm(v0) - 1.0) < 1e-5
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5
    assert abs(np.linalg.norm(v2) - 1.0) < 1e-5


def test_mrl_pipeline_batch_encoding():
    pipeline = MRLEmbeddingPipeline(use_deterministic_fallback=True)
    texts = [
        "search_document: [Domain: Developer Productivity & CI/CD] Title: Bazel Build Cache",
        "search_document: [Domain: Data Infrastructure & Analytics] Title: Real-time Kafka Stream",
    ]

    embeddings = pipeline.encode_text_batch(texts, batch_size=2)
    assert len(embeddings) == 2

    for emb in embeddings:
        assert len(emb.d0_64) == 64
        assert len(emb.d1_128) == 128
        assert len(emb.d2_768) == 768

        # Verify unit norm on returned Pydantic object
        assert abs(np.linalg.norm(emb.d0_64) - 1.0) < 1e-5
        assert abs(np.linalg.norm(emb.d1_128) - 1.0) < 1e-5
        assert abs(np.linalg.norm(emb.d2_768) - 1.0) < 1e-5
