"""
Unit tests for Pydantic project schema, RICE factors, and MRL serializers.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from smriti.models.project_schema import (
    Confidence,
    ConfidencePercentage,
    ConfidenceTier,
    Effort,
    EffortBreakdown,
    EnterpriseProjectNode,
    GraphRelations,
    Impact,
    ImpactScore,
    ImpactTier,
    MRLEmbeddings,
    RICEFactors,
    Reach,
    SemanticContext,
    Taxonomy,
)


def test_rice_factors_calculation_and_validation():
    reach = Reach(value=500.0, unit="engineers")
    impact = Impact(score=ImpactScore.HIGH, tier=ImpactTier.HIGH, rationale="Major platform speedup")
    confidence = Confidence(percentage=ConfidencePercentage.MEDIUM, tier=ConfidenceTier.MEDIUM, evidence="Benchmarked in staging")
    effort = Effort(person_weeks=10.0, breakdown=EffortBreakdown(frontend=2.0, backend=5.0, infra_devops=3.0, qa_testing=0.0))

    # Auto-calculates: (500 * 2.0 * 0.8) / 10.0 = 80.0
    factors = RICEFactors(
        reach=reach,
        impact=impact,
        confidence=confidence,
        effort=effort,
        rice_score=80.0,
    )
    assert factors.rice_score == 80.0
    assert factors.reach.value == 500.0
    assert factors.impact.score == 2.0
    assert factors.confidence.percentage == 0.8


def test_mrl_embeddings_unit_normalization():
    # Valid unit vector
    vec_64 = np.ones(64, dtype=np.float32) / np.sqrt(64)
    vec_128 = np.ones(128, dtype=np.float32) / np.sqrt(128)
    vec_768 = np.ones(768, dtype=np.float32) / np.sqrt(768)

    emb = MRLEmbeddings(
        d0_64=vec_64.tolist(),
        d1_128=vec_128.tolist(),
        d2_768=vec_768.tolist(),
    )
    assert len(emb.d0_64) == 64
    assert len(emb.d1_128) == 128
    assert len(emb.d2_768) == 768

    # Non-normalized vector should raise ValidationError
    bad_vec = [10.0] * 64
    with pytest.raises(ValidationError):
        MRLEmbeddings(d0_64=bad_vec)


def test_enterprise_project_node_serialization():
    node = EnterpriseProjectNode(
        id="proj_devprod_2026_0001",
        taxonomy=Taxonomy(
            domain="Developer Productivity & CI/CD",
            sub_domain="Build Caching & Artifact Distribution",
            epic="Distributed Bazel Cache",
        ),
        semantic_context=SemanticContext(
            title="Distributed Bazel Remote Build Cache with S3 Acceleration",
            problem_statement="Local build times exceed 45 minutes for core monorepo services.",
            solution_proposal="Deploy a distributed gRPC caching proxy backed by NVMe caches and S3 tiered storage.",
            tech_stack=["Rust", "gRPC", "AWS S3", "Bazel", "Redis"],
            target_users="All backend and platform engineers (~450 devs)",
        ),
        rice_factors=RICEFactors(
            reach=Reach(value=450.0, unit="engineers"),
            impact=Impact(score=ImpactScore.HIGH, tier=ImpactTier.HIGH),
            confidence=Confidence(percentage=ConfidencePercentage.HIGH, tier=ConfidenceTier.HIGH),
            effort=Effort(person_weeks=6.0),
            rice_score=150.0,
        ),
        graph_relations=GraphRelations(
            upstream_dependencies=[],
            downstream_impacted=["proj_frontend_2026_0010"],
            tags=["infra", "bazel", "caching"],
        ),
    )

    unified_text = node.to_unified_text()
    assert "search_document:" in unified_text
    assert "[Domain: Developer Productivity & CI/CD]" in unified_text
    assert "Distributed Bazel Remote Build Cache" in unified_text
    assert "Rust, gRPC, AWS S3, Bazel, Redis" in unified_text

    flat_dict = node.to_flat_dict()
    assert flat_dict["id"] == "proj_devprod_2026_0001"
    assert flat_dict["domain"] == "Developer Productivity & CI/CD"
    assert flat_dict["impact_score"] == 2.0
    assert flat_dict["rice_score"] == 150.0
