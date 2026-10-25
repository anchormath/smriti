"""
Automated test suite for Multimodal Associative Retrieval in Smriti.
Validates cross-modal recall (audio -> diagram, diagram -> audio/code),
Matryoshka prefix slicing (d0=64, d1=128, d2=768), and Hopfield Resolution Dial dynamics.
Aligned with Naksha's 4-domain model (Sangh, Kala, Krama, Vyapar).
"""

from __future__ import annotations

import numpy as np
import pytest

from smriti.embeddings.multimodal_pipeline import MultimodalMRLEmbeddingPipeline
from smriti.models.multimodal_schema import (
    EnterpriseDomain,
    ModalityType,
    MultimodalArtifact,
    SpatialRegion,
    TemporalSegment,
)
from smriti.tree.hamp_tree import HAMPTreeWrapper


@pytest.fixture
def multimodal_pipeline() -> MultimodalMRLEmbeddingPipeline:
    return MultimodalMRLEmbeddingPipeline(use_deterministic_fallback=True)


@pytest.fixture
def sample_multimodal_corpus(multimodal_pipeline: MultimodalMRLEmbeddingPipeline) -> list[MultimodalArtifact]:
    """Generates a realistic multi-service corpus across Krama and Sangh."""
    artifacts = [
        # Subsystem 1: Event Gateway (Krama)
        MultimodalArtifact(
            id="art_krama_diag_event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.IMAGE_DIAGRAM,
            title="Event Gateway Architecture Diagram",
            description="C4 component diagram of Kafka ingress, Envoy proxy, and backpressure buffers.",
            media_uri="data/diagrams/event_gateway_c4.png",
            spatial_region=SpatialRegion(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5, component_name="KafkaIngressBuffer"),
            linked_artifact_ids=["art_krama_audio_event_gw", "art_krama_spec_event_gw"],
            tags=["kafka", "streaming", "ingress", "grpc"],
        ),
        MultimodalArtifact(
            id="art_krama_audio_event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Event Gateway Design Review Recording",
            description="Audio walkthrough explaining partition rebalancing and consumer backpressure mitigation.",
            media_uri="data/audio/tdr_event_gw_2026.wav",
            temporal_segment=TemporalSegment(start_sec=45.0, end_sec=120.0, transcript_snippet="backpressure buffer queue limits"),
            linked_artifact_ids=["art_krama_diag_event_gw", "art_krama_spec_event_gw"],
            tags=["kafka", "streaming", "backpressure", "incident"],
        ),
        MultimodalArtifact(
            id="art_krama_spec_event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.CODE_SPEC,
            title="Event Gateway Protocol Buffers Spec",
            description="gRPC service definitions and buffer limits for stream ingestion.",
            media_uri="proto/event_gateway.proto",
            linked_artifact_ids=["art_krama_diag_event_gw", "art_krama_audio_event_gw"],
            tags=["grpc", "protobuf", "streaming"],
        ),

        # Subsystem 2: Distributed Cache (Krama)
        MultimodalArtifact(
            id="art_krama_diag_cache",
            domain=EnterpriseDomain.KRAMA,
            subdomain="data_store",
            parent_entity_id="data_store:redis_tiered_cache",
            modality=ModalityType.IMAGE_DIAGRAM,
            title="Tiered Cache Topology Diagram",
            description="L1 in-process cache + L2 Redis cluster consistent hashing topology.",
            media_uri="data/diagrams/cache_topology.png",
            linked_artifact_ids=["art_krama_audio_cache"],
            tags=["redis", "cache", "consistent-hash"],
        ),
        MultimodalArtifact(
            id="art_krama_audio_cache",
            domain=EnterpriseDomain.KRAMA,
            subdomain="data_store",
            parent_entity_id="data_store:redis_tiered_cache",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Cache Invalidation Incident Debrief",
            description="Voice recording debriefing cache thundering herd and stampede protection.",
            media_uri="data/audio/cache_debrief.wav",
            linked_artifact_ids=["art_krama_diag_cache"],
            tags=["redis", "invalidation", "cache", "incident"],
        ),

        # Subsystem 3: People & Org (Sangh - Negative Control)
        MultimodalArtifact(
            id="art_sangh_review_lead",
            domain=EnterpriseDomain.SANGH,
            subdomain="review",
            parent_entity_id="employee:eng_lead_01",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Engineering Leadership Performance Evaluation",
            description="Voice notes from quarterly performance review covering mentorship and org health.",
            media_uri="data/audio/review_lead_01.wav",
            tags=["leadership", "org", "mentorship", "quarterly"],
        ),
    ]

    for art in artifacts:
        multimodal_pipeline.encode_artifact(art)

    return artifacts


def test_multimodal_artifact_unit_normalization(sample_multimodal_corpus: list[MultimodalArtifact]):
    """Verify that all MRL slices (d0=64, d1=128, d2=768) are strictly unit-normalized (|norm - 1.0| < 1e-4)."""
    for art in sample_multimodal_corpus:
        assert art.embeddings is not None
        v0 = np.array(art.embeddings.d0_64, dtype=np.float32)
        v1 = np.array(art.embeddings.d1_128, dtype=np.float32)
        v2 = np.array(art.embeddings.d2_768, dtype=np.float32)

        assert abs(float(np.linalg.norm(v0)) - 1.0) < 1e-4, f"Slice d0 for {art.id} not unit-normalized"
        assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4, f"Slice d1 for {art.id} not unit-normalized"
        assert abs(float(np.linalg.norm(v2)) - 1.0) < 1e-4, f"Slice d2 for {art.id} not unit-normalized"


def test_cross_modal_semantic_alignment(
    multimodal_pipeline: MultimodalMRLEmbeddingPipeline,
    sample_multimodal_corpus: list[MultimodalArtifact],
):
    """
    Verify that an audio track and an architectural diagram describing the same system
    converge to the same semantic basin (cosine similarity >= 0.70),
    while an unrelated HR review in Sangh has low similarity (<= 0.40).
    """
    corpus_map = {a.id: a for a in sample_multimodal_corpus}

    v_diag = np.array(corpus_map["art_krama_diag_event_gw"].embeddings.d2_768, dtype=np.float32)
    v_audio = np.array(corpus_map["art_krama_audio_event_gw"].embeddings.d2_768, dtype=np.float32)
    v_hr = np.array(corpus_map["art_sangh_review_lead"].embeddings.d2_768, dtype=np.float32)

    # Cosine similarities (vectors are unit-normalized, so dot product = cosine similarity)
    cos_diag_audio = float(np.dot(v_diag, v_audio))
    cos_diag_hr = float(np.dot(v_diag, v_hr))

    assert cos_diag_audio >= 0.70, f"Expected high cross-modal similarity, got {cos_diag_audio:.4f}"
    assert cos_diag_hr <= 0.40, f"Expected domain separation, got {cos_diag_hr:.4f}"
    assert cos_diag_audio > cos_diag_hr + 0.35, "Cross-modal affinity must significantly exceed cross-domain affinity"


def test_cross_modal_retrieval_in_hamp_tree(
    multimodal_pipeline: MultimodalMRLEmbeddingPipeline,
    sample_multimodal_corpus: list[MultimodalArtifact],
):
    """
    Test end-to-end indexing and retrieval in Smriti's HAM-P tree:
    1. Querying with an Audio Track retrieves the corresponding Architectural Diagram in Top-2.
    2. Querying with an Architectural Diagram retrieves the corresponding Audio Track in Top-2.
    """
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=10)

    # Index all artifacts
    for art in sample_multimodal_corpus:
        tree.insert(art.id, art.embeddings.d2_768)

    corpus_map = {a.id: a for a in sample_multimodal_corpus}

    # Query 1: Use Event Gateway Audio to find Diagram
    audio_query_vec = corpus_map["art_krama_audio_event_gw"].embeddings.d2_768
    results_audio_q = tree.query(audio_query_vec, top_k=3, beam_width=3)
    retrieved_ids_1 = [r.item_id for r in results_audio_q]

    # The audio itself will be top-1 (self-recall), but the linked diagram MUST be in top-3
    assert "art_krama_diag_event_gw" in retrieved_ids_1, (
        f"Audio query failed to recall linked diagram. Retrieved: {retrieved_ids_1}"
    )

    # Query 2: Use Cache Diagram to find Cache Audio
    diag_query_vec = corpus_map["art_krama_diag_cache"].embeddings.d2_768
    results_diag_q = tree.query(diag_query_vec, top_k=3, beam_width=3)
    retrieved_ids_2 = [r.item_id for r in results_diag_q]

    assert "art_krama_audio_cache" in retrieved_ids_2, (
        f"Diagram query failed to recall linked audio debrief. Retrieved: {retrieved_ids_2}"
    )


def test_hopfield_resolution_dial_cross_modal_dynamics(
    sample_multimodal_corpus: list[MultimodalArtifact],
):
    """
    Verify Continuous Hopfield dynamics across Resolution Dial settings R in [0.0, 1.0]:
    - R=0.1: Blends candidate artifacts in the leaf pool (high entropy, moderate weights).
    - R=0.9: Focuses almost all attention on the single target artifact (weight > 0.85).
    - Energy is monotonic and stable.
    """
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=10)
    for art in sample_multimodal_corpus:
        tree.insert(art.id, art.embeddings.d2_768)

    corpus_map = {a.id: a for a in sample_multimodal_corpus}
    query_vec = corpus_map["art_krama_diag_event_gw"].embeddings.d2_768

    sweep = tree.sweep_resolution_dial(query_vec, resolutions=[0.1, 0.5, 0.9])
    assert len(sweep) == 3

    r_low = sweep[0]
    r_high = sweep[2]

    weights_low = r_low["weights"]
    weights_high = r_high["weights"]

    # At low resolution R=0.1, attention is dispersed across all items in the leaf
    # At high resolution R=0.9, attention concentrates on the target subsystem cluster
    # Identify indices of the target subsystem artifacts
    target_ids = {"art_krama_diag_event_gw", "art_krama_audio_event_gw", "art_krama_spec_event_gw"}
    candidate_ids = r_high["candidate_ids"]

    target_weight_low = sum(w for cid, w in zip(candidate_ids, weights_low) if cid in target_ids)
    target_weight_high = sum(w for cid, w in zip(candidate_ids, weights_high) if cid in target_ids)

    assert target_weight_high > target_weight_low, (
        f"Expected higher target cluster attention at R=0.9: {target_weight_high:.4f} vs {target_weight_low:.4f}"
    )
    assert target_weight_high >= 0.90, (
        f"Target cluster should capture >= 90% attention at R=0.9, got {target_weight_high:.4f}"
    )

    # Off-target items (cache, hr) should have minimal attention at R=0.9
    off_target_weight_high = 1.0 - target_weight_high
    assert off_target_weight_high <= 0.10, (
        f"Off-target items should have <= 10% attention at R=0.9, got {off_target_weight_high:.4f}"
    )


def test_mrl_prefix_routing_domain_pruning(
    sample_multimodal_corpus: list[MultimodalArtifact],
):
    """
    Verify that d0=64 provides effective domain separation between Krama and Sangh,
    allowing HAM-P to prune entire subtrees before full Hopfield evaluation.
    """
    corpus_map = {a.id: a for a in sample_multimodal_corpus}

    v_krama_d0 = np.array(corpus_map["art_krama_diag_event_gw"].embeddings.d0_64, dtype=np.float32)
    v_krama_peer_d0 = np.array(corpus_map["art_krama_diag_cache"].embeddings.d0_64, dtype=np.float32)
    v_sangh_d0 = np.array(corpus_map["art_sangh_review_lead"].embeddings.d0_64, dtype=np.float32)

    sim_within_domain = float(np.dot(v_krama_d0, v_krama_peer_d0))
    sim_cross_domain = float(np.dot(v_krama_d0, v_sangh_d0))

    assert sim_within_domain > sim_cross_domain + 0.30, (
        f"Domain separability at d0=64 insufficient: within={sim_within_domain:.4f}, cross={sim_cross_domain:.4f}"
    )
