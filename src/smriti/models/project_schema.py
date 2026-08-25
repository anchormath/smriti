"""
Pydantic schemas and serialization methods for Enterprise Project Nodes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import math
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ImpactScore(float, Enum):
    MINIMAL = 0.25
    LOW = 0.5
    MEDIUM = 1.0
    HIGH = 2.0
    MASSIVE = 3.0


class ImpactTier(str, Enum):
    MINIMAL = "Minimal"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    MASSIVE = "Massive"


class ConfidencePercentage(float, Enum):
    LOW = 0.5
    MEDIUM = 0.8
    HIGH = 1.0


class ConfidenceTier(str, Enum):
    LOW = "Low / Moonshot"
    MEDIUM = "Medium"
    HIGH = "High"


class Taxonomy(BaseModel):
    domain: str = Field(..., description="Top-level engineering or business domain")
    sub_domain: str = Field(..., description="Specialized sub-domain or platform area")
    epic: str = Field(..., description="High-level initiative or theme")


class SemanticContext(BaseModel):
    title: str = Field(..., description="Initiative title")
    problem_statement: str = Field(..., description="Detailed description of problem and bottlenecks")
    solution_proposal: str = Field(..., description="Technical architecture and implementation proposal")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies, languages, and tools involved")
    target_users: Optional[str] = Field(default=None, description="Audience or internal teams impacted")


class Reach(BaseModel):
    value: float = Field(..., gt=0, description="Users or systems impacted per quarter")
    unit: str = Field(default="users/quarter", description="Unit of measurement")


class Impact(BaseModel):
    score: ImpactScore = Field(..., description="Numerical impact multiplier [0.25, 0.5, 1.0, 2.0, 3.0]")
    tier: ImpactTier = Field(..., description="Qualitative impact category")
    rationale: Optional[str] = Field(default=None, description="Business and architectural reasoning")


class Confidence(BaseModel):
    percentage: ConfidencePercentage = Field(..., description="Confidence level multiplier [0.5, 0.8, 1.0]")
    tier: ConfidenceTier = Field(..., description="Qualitative confidence assessment")
    evidence: Optional[str] = Field(default=None, description="Supporting data, prototypes, or user research")


class EffortBreakdown(BaseModel):
    frontend: float = Field(default=0.0, ge=0.0, description="Frontend engineering person-weeks")
    backend: float = Field(default=0.0, ge=0.0, description="Backend engineering person-weeks")
    infra_devops: float = Field(default=0.0, ge=0.0, description="Infrastructure / DevOps person-weeks")
    qa_testing: float = Field(default=0.0, ge=0.0, description="QA and testing person-weeks")


class Effort(BaseModel):
    person_weeks: float = Field(..., gt=0, description="Total effort required in person-weeks")
    breakdown: Optional[EffortBreakdown] = Field(default=None, description="Breakdown across disciplines")


class RICEFactors(BaseModel):
    reach: Reach
    impact: Impact
    confidence: Confidence
    effort: Effort
    rice_score: float = Field(..., description="Computed RICE score = (Reach * Impact * Confidence) / Effort")

    @model_validator(mode="before")
    @classmethod
    def calculate_or_validate_rice_score(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reach_val = data.get("reach", {})
            if isinstance(reach_val, dict):
                reach_num = reach_val.get("value")
            elif hasattr(reach_val, "value"):
                reach_num = reach_val.value
            else:
                reach_num = None

            impact_val = data.get("impact", {})
            if isinstance(impact_val, dict):
                impact_num = impact_val.get("score")
            elif hasattr(impact_val, "score"):
                impact_num = impact_val.score
            else:
                impact_num = None

            conf_val = data.get("confidence", {})
            if isinstance(conf_val, dict):
                conf_num = conf_val.get("percentage")
            elif hasattr(conf_val, "percentage"):
                conf_num = conf_val.percentage
            else:
                conf_num = None

            effort_val = data.get("effort", {})
            if isinstance(effort_val, dict):
                effort_num = effort_val.get("person_weeks")
            elif hasattr(effort_val, "person_weeks"):
                effort_num = effort_val.person_weeks
            else:
                effort_num = None

            if (
                reach_num is not None
                and impact_num is not None
                and conf_num is not None
                and effort_num is not None
                and effort_num > 0
            ):
                impact_f = float(impact_num)
                conf_f = float(conf_num)
                reach_f = float(reach_num)
                effort_f = float(effort_num)
                expected_score = round((reach_f * impact_f * conf_f) / effort_f, 4)
                if "rice_score" not in data or data["rice_score"] is None:
                    data["rice_score"] = expected_score
        return data


class GraphRelations(BaseModel):
    upstream_dependencies: List[str] = Field(default_factory=list, description="IDs of upstream dependency projects")
    downstream_impacted: List[str] = Field(default_factory=list, description="IDs of downstream projects impacted")
    tags: List[str] = Field(default_factory=list, description="Categorical and architectural tags")


class MRLEmbeddings(BaseModel):
    d0_64: Optional[List[float]] = Field(default=None, description="Prefix slice dim 64 (L2 normalized)")
    d1_128: Optional[List[float]] = Field(default=None, description="Prefix slice dim 128 (L2 normalized)")
    d2_768: Optional[List[float]] = Field(default=None, description="Full Hopfield vector dim 768 (L2 normalized)")

    @field_validator("d0_64", "d1_128", "d2_768")
    @classmethod
    def validate_unit_norm(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        arr = np.array(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"Vector slice must be unit-normalized (|norm - 1.0| < 1e-4), got {norm}")
        return v


class EnterpriseProjectNode(BaseModel):
    id: str = Field(..., description="Unique project identifier, e.g., proj_infra_2026_042")
    taxonomy: Taxonomy
    semantic_context: SemanticContext
    rice_factors: RICEFactors
    graph_relations: GraphRelations = Field(default_factory=GraphRelations)
    embeddings: Optional[MRLEmbeddings] = Field(default=None)

    def to_unified_text(self) -> str:
        """
        Serializes the project node to the standardized MRL search document format.
        """
        tech_str = ", ".join(self.semantic_context.tech_stack) if self.semantic_context.tech_stack else "None"
        target_str = self.semantic_context.target_users or "General Engineering"
        return (
            f"search_document: [Domain: {self.taxonomy.domain}] [Sub-domain: {self.taxonomy.sub_domain}]\n"
            f"Title: {self.semantic_context.title}\n"
            f"Problem: {self.semantic_context.problem_statement}\n"
            f"Solution: {self.semantic_context.solution_proposal}\n"
            f"Tech Stack: {tech_str}\n"
            f"Target Audience: {target_str}"
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flattens the project node for columnar export (Parquet / Pandas).
        """
        flat = {
            "id": self.id,
            "domain": self.taxonomy.domain,
            "sub_domain": self.taxonomy.sub_domain,
            "epic": self.taxonomy.epic,
            "title": self.semantic_context.title,
            "problem_statement": self.semantic_context.problem_statement,
            "solution_proposal": self.semantic_context.solution_proposal,
            "tech_stack": self.semantic_context.tech_stack,
            "target_users": self.semantic_context.target_users,
            "reach_value": self.rice_factors.reach.value,
            "reach_unit": self.rice_factors.reach.unit,
            "impact_score": float(self.rice_factors.impact.score),
            "impact_tier": self.rice_factors.impact.tier.value if hasattr(self.rice_factors.impact.tier, "value") else self.rice_factors.impact.tier,
            "impact_rationale": self.rice_factors.impact.rationale,
            "confidence_percentage": float(self.rice_factors.confidence.percentage),
            "confidence_tier": self.rice_factors.confidence.tier.value if hasattr(self.rice_factors.confidence.tier, "value") else self.rice_factors.confidence.tier,
            "confidence_evidence": self.rice_factors.confidence.evidence,
            "effort_person_weeks": self.rice_factors.effort.person_weeks,
            "rice_score": self.rice_factors.rice_score,
            "upstream_dependencies": self.graph_relations.upstream_dependencies,
            "downstream_impacted": self.graph_relations.downstream_impacted,
            "tags": self.graph_relations.tags,
            "unified_text": self.to_unified_text(),
        }

        if self.rice_factors.effort.breakdown:
            flat["effort_frontend"] = self.rice_factors.effort.breakdown.frontend
            flat["effort_backend"] = self.rice_factors.effort.breakdown.backend
            flat["effort_infra"] = self.rice_factors.effort.breakdown.infra_devops
            flat["effort_qa"] = self.rice_factors.effort.breakdown.qa_testing
        else:
            flat["effort_frontend"] = 0.0
            flat["effort_backend"] = 0.0
            flat["effort_infra"] = 0.0
            flat["effort_qa"] = 0.0

        if self.embeddings:
            flat["mrl_d0_64"] = self.embeddings.d0_64
            flat["mrl_d1_128"] = self.embeddings.d1_128
            flat["mrl_d2_768"] = self.embeddings.d2_768
        else:
            flat["mrl_d0_64"] = None
            flat["mrl_d1_128"] = None
            flat["mrl_d2_768"] = None

        return flat
