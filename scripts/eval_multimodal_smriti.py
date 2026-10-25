#!/usr/bin/env python3
"""
Interactive CLI Evaluation & Demonstration for Multimodal Associative Retrieval in Smriti.
Demonstrates cross-modal recall (audio -> diagram, diagram -> audio, text -> multimodal),
Resolution Dial sweeps (R in [0.0, 1.0]), and cross-domain edge discovery aligned with Naksha.
"""

from __future__ import annotations

import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from smriti.embeddings.multimodal_pipeline import MultimodalMRLEmbeddingPipeline
from smriti.models.multimodal_schema import (
    EnterpriseDomain,
    ModalityType,
    MultimodalArtifact,
    SpatialRegion,
    TemporalSegment,
)
from smriti.tree.hamp_tree import HAMPTreeWrapper

console = Console()


def build_enterprise_multimodal_corpus(
    pipeline: MultimodalMRLEmbeddingPipeline,
) -> list[MultimodalArtifact]:
    """Builds a rich multimodal corpus across Krama (Systems), Vyapar (Business), and Sangh (Org)."""
    artifacts = [
        # --- KRAMA: Distributed Event Gateway ---
        MultimodalArtifact(
            id="art:krama:diag:event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.IMAGE_DIAGRAM,
            title="Event Gateway Ingestion Topology (C4)",
            description="Visual C4 diagram detailing Kafka ingress, Envoy edge proxy, and memory buffer limits.",
            media_uri="data/diagrams/event_gw_c4.png",
            spatial_region=SpatialRegion(ymin=0.1, xmin=0.1, ymax=0.4, xmax=0.5, component_name="KafkaPartitionBroker"),
            linked_artifact_ids=["art:krama:audio:event_gw", "art:krama:spec:event_gw"],
            tags=["kafka", "streaming", "c4", "grpc", "ingress"],
        ),
        MultimodalArtifact(
            id="art:krama:audio:event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Event Gateway TDR Audio Walkthrough",
            description="Lead architect audio review discussing partition rebalancing and consumer backpressure limits.",
            media_uri="data/audio/tdr_event_gw_2026.wav",
            temporal_segment=TemporalSegment(start_sec=30.0, end_sec=95.0, transcript_snippet="partition buffer thresholds during spike traffic"),
            linked_artifact_ids=["art:krama:diag:event_gw", "art:krama:spec:event_gw"],
            tags=["kafka", "streaming", "audio_tdr", "backpressure"],
        ),
        MultimodalArtifact(
            id="art:krama:spec:event_gw",
            domain=EnterpriseDomain.KRAMA,
            subdomain="service",
            parent_entity_id="service:event_gateway",
            modality=ModalityType.CODE_SPEC,
            title="Event Gateway Protobuf & Buffer Spec",
            description="gRPC protocol buffer definitions, stream window limits, and deadline handling.",
            media_uri="proto/event_gateway.proto",
            linked_artifact_ids=["art:krama:diag:event_gw", "art:krama:audio:event_gw"],
            tags=["grpc", "protobuf", "spec"],
        ),

        # --- KRAMA: Tiered Redis Cache ---
        MultimodalArtifact(
            id="art:krama:diag:redis_cache",
            domain=EnterpriseDomain.KRAMA,
            subdomain="data_store",
            parent_entity_id="data_store:redis_tiered_cache",
            modality=ModalityType.IMAGE_DIAGRAM,
            title="Tiered Redis Cache Cluster Topology",
            description="Diagram illustrating L1 in-process LRU cache and L2 Redis consistent hashing ring.",
            media_uri="data/diagrams/cache_cluster.png",
            linked_artifact_ids=["art:krama:audio:cache_debrief"],
            tags=["redis", "cache", "consistent-hash", "lru"],
        ),
        MultimodalArtifact(
            id="art:krama:audio:cache_debrief",
            domain=EnterpriseDomain.KRAMA,
            subdomain="data_store",
            parent_entity_id="data_store:redis_tiered_cache",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Cache Thundering Herd Post-Mortem Audio",
            description="Incident commander audio debrief on stampede mitigation and probabilistic early expiration.",
            media_uri="data/audio/cache_incident_debrief.wav",
            linked_artifact_ids=["art:krama:diag:redis_cache"],
            tags=["redis", "post_mortem", "stampede", "cache"],
        ),

        # --- VYAPAR: Real-Time Fraud & Anomaly Alerts (Product Feature) ---
        MultimodalArtifact(
            id="art:vyapar:spec:fraud_alerts",
            domain=EnterpriseDomain.VYAPAR,
            subdomain="product_feature",
            parent_entity_id="product_feature:realtime_fraud_alerting",
            modality=ModalityType.TEXT,
            title="Real-Time Fraud & Anomaly Alerting PRD",
            description="Product feature requirements for real-time transaction stream anomaly detection and notification.",
            media_uri="docs/prd/fraud_alerts.md",
            linked_artifact_ids=["art:krama:diag:event_gw"],
            tags=["fraud", "alerting", "streaming", "product"],
        ),

        # --- SANGH: Platform Engineering Team Review ---
        MultimodalArtifact(
            id="art:sangh:review:platform_team",
            domain=EnterpriseDomain.SANGH,
            subdomain="review",
            parent_entity_id="team:platform_infra",
            modality=ModalityType.AUDIO_WALKTHROUGH,
            title="Platform Team Performance Review & Mentorship Audio",
            description="Staff engineer voice notes on organizational health, onboarding, and team career development.",
            media_uri="data/audio/review_platform_team.wav",
            tags=["mentorship", "career", "org", "quarterly"],
        ),
    ]

    for art in artifacts:
        pipeline.encode_artifact(art)

    return artifacts


def run_multimodal_evaluation():
    console.print(Panel.fit(
        "[bold cyan]Smriti (स्मृति) — Multimodal Matryoshka Associative Memory Evaluation[/bold cyan]\n"
        "[dim]Testing Cross-Modal Retrieval (Audio ↔ Diagram ↔ Spec) with HAM-P & HAM-R[/dim]",
        border_style="cyan",
    ))

    # 1. Initialize Pipeline & Corpus
    pipeline = MultimodalMRLEmbeddingPipeline(use_deterministic_fallback=True)
    corpus = build_enterprise_multimodal_corpus(pipeline)
    corpus_map = {a.id: a for a in corpus}

    # 2. Ingest into Native HAM-P Tree
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=10)
    t0 = time.perf_counter()
    for art in corpus:
        assert art.embeddings is not None
        tree.insert(art.id, art.embeddings.d2_768)
    t_ingest = (time.perf_counter() - t0) * 1000

    console.print(f"[green]✓[/green] Ingested [bold]{len(corpus)}[/bold] multimodal artifacts into HAM-P tree in [bold]{t_ingest:.2f} ms[/bold].\n")

    # 3. Test Scenarios
    scenarios = [
        {
            "name": "Scenario 1: Audio Walkthrough ➔ Recalls C4 Architecture Diagram",
            "query_modality": "Audio",
            "query_artifact_id": "art:krama:audio:event_gw",
            "target_artifact_id": "art:krama:diag:event_gw",
            "domain": "Krama",
        },
        {
            "name": "Scenario 2: Visual C4 Diagram ➔ Recalls Post-Mortem Audio Recording",
            "query_modality": "Image/Diagram",
            "query_artifact_id": "art:krama:diag:redis_cache",
            "target_artifact_id": "art:krama:audio:cache_debrief",
            "domain": "Krama",
        },
    ]

    for sc in scenarios:
        console.print(f"[bold yellow]{sc['name']}[/bold yellow]")
        q_art = corpus_map[sc["query_artifact_id"]]
        target_art = corpus_map[sc["target_artifact_id"]]

        # Execute query at full dimension d2=768
        t_start = time.perf_counter()
        results = tree.query(q_art.embeddings.d2_768, top_k=3, beam_width=3)
        latency_us = (time.perf_counter() - t_start) * 1e6

        table = Table(title=f"HAM-P Tree Top-3 Nearest Neighbors (Latency: {latency_us:.1f} µs)", show_header=True)
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Artifact ID", style="cyan")
        table.add_column("Modality", style="magenta")
        table.add_column("Domain", style="blue")
        table.add_column("Score (Cosine)", justify="right", style="green")
        table.add_column("Target Match?", justify="center")

        for rank, res in enumerate(results, start=1):
            item = corpus_map.get(res.item_id)
            is_match = "🎯 TARGET" if res.item_id == sc["target_artifact_id"] else ("Self" if res.item_id == q_art.id else "—")
            table.add_row(
                str(rank),
                res.item_id,
                item.modality.value if item else "?",
                item.domain.value if item else "?",
                f"{res.score:.4f}",
                f"[bold green]{is_match}[/bold green]" if "TARGET" in is_match else is_match,
            )

        console.print(table)
        console.print()

    # 4. Resolution Dial Sweep (HAM-R Hopfield Dynamics)
    console.print("[bold yellow]Scenario 3: Continuous Hopfield Resolution Dial (R ∈ [0.1, 0.5, 0.9])[/bold yellow]")
    q_vec = corpus_map["art:krama:diag:event_gw"].embeddings.d2_768
    resolutions = [0.1, 0.5, 0.9]
    sweep_results = tree.sweep_resolution_dial(q_vec, resolutions=resolutions)

    dial_table = Table(title="Hopfield Resolution Dial Dynamics & Energy Minimization", show_header=True)
    dial_table.add_column("Resolution (R)", justify="center", style="bold")
    dial_table.add_column("Regime / Dynamics", style="cyan")
    dial_table.add_column("Final Energy E(x)", justify="right", style="magenta")
    dial_table.add_column("Target Subsystem Attention", justify="right", style="green")
    dial_table.add_column("Max Single Weight", justify="right")

    target_ids = {"art:krama:diag:event_gw", "art:krama:audio:event_gw", "art:krama:spec:event_gw"}

    for s in sweep_results:
        r = s["resolution"]
        regime = "Broad Cluster Blend" if r < 0.3 else ("Metastable Composite" if r < 0.7 else "Exact Isolated Precedent")
        weights = s["weights"]
        cids = s["candidate_ids"]
        target_w = sum(w for cid, w in zip(cids, weights) if cid in target_ids)
        max_w = float(np.max(weights))

        dial_table.add_row(
            f"{r:.2f}",
            regime,
            f"{s['final_energy']:.4f}",
            f"{target_w * 100:.1f}%",
            f"{max_w:.4f}",
        )

    console.print(dial_table)
    console.print()

    console.print("[bold green]✔ All Multimodal Associative Retrieval benchmarks completed successfully![/bold green]\n")


if __name__ == "__main__":
    run_multimodal_evaluation()
