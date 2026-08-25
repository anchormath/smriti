"""
Seed generator for statistical RICE factors, taxonomy assignment, and graph DAG synthesis.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from faker import Faker

from smriti.generators.taxonomy import DOMAIN_TAXONOMY, SubDomainMeta
from smriti.models.project_schema import (
    Confidence,
    ConfidencePercentage,
    ConfidenceTier,
    Effort,
    EffortBreakdown,
    GraphRelations,
    Impact,
    ImpactScore,
    ImpactTier,
    RICEFactors,
    Reach,
    Taxonomy,
)

IMPACT_DISTRIBUTION: List[Tuple[ImpactScore, ImpactTier, float]] = [
    (ImpactScore.MINIMAL, ImpactTier.MINIMAL, 0.05),
    (ImpactScore.LOW, ImpactTier.LOW, 0.20),
    (ImpactScore.MEDIUM, ImpactTier.MEDIUM, 0.40),
    (ImpactScore.HIGH, ImpactTier.HIGH, 0.25),
    (ImpactScore.MASSIVE, ImpactTier.MASSIVE, 0.10),
]

CONFIDENCE_DISTRIBUTION: List[Tuple[ConfidencePercentage, ConfidenceTier, float]] = [
    (ConfidencePercentage.LOW, ConfidenceTier.LOW, 0.15),
    (ConfidencePercentage.MEDIUM, ConfidenceTier.MEDIUM, 0.40),
    (ConfidencePercentage.HIGH, ConfidenceTier.HIGH, 0.45),
]

DOMAIN_PREFIXES = {
    "Developer Productivity & CI/CD": "devprod",
    "Data Infrastructure & Analytics": "datainfra",
    "AI/ML Platform & Inference": "aiml",
    "Security & Zero-Trust Architecture": "secops",
    "Core Backend & Microservices": "backend",
    "Cloud Infrastructure & Reliability": "cloudinfra",
    "Frontend & Design Systems": "frontend",
    "Search & Discovery": "search",
}


class SeedGenerator:
    """
    Generates structured structural properties, RICE metrics, and relational links.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

    def sample_taxonomy(self, domain: Optional[str] = None) -> Tuple[Taxonomy, SubDomainMeta]:
        """
        Samples a domain, sub-domain, and epic from the curated enterprise taxonomy.
        """
        domains = list(DOMAIN_TAXONOMY.keys())
        selected_domain = domain if domain in DOMAIN_TAXONOMY else str(self.rng.choice(domains))
        sub_domains: List[SubDomainMeta] = DOMAIN_TAXONOMY[selected_domain]
        sub_domain_meta = random.choice(sub_domains)
        epic = random.choice(sub_domain_meta.epics)
        taxonomy = Taxonomy(
            domain=selected_domain,
            sub_domain=sub_domain_meta.name,
            epic=epic,
        )
        return taxonomy, sub_domain_meta

    def sample_reach(self, domain: str) -> Reach:
        """
        Samples Reach value from a log-normal distribution matching enterprise systems.
        """
        # Internal developer tools vs user-facing systems have different scale baselines
        if domain in ["Developer Productivity & CI/CD", "Security & Zero-Trust Architecture"]:
            # Internal scale: 50 to 5,000 developers/services
            log_mean = float(np.log(350))
            log_std = 0.85
            raw = float(self.rng.lognormal(mean=log_mean, sigma=log_std))
            value = max(20.0, min(raw, 25000.0))
            unit = "engineers / services"
        else:
            # End-user / cloud scale: 1,000 to 5,000,000 users/queries
            log_mean = float(np.log(75000))
            log_std = 1.35
            raw = float(self.rng.lognormal(mean=log_mean, sigma=log_std))
            value = max(100.0, min(raw, 10000000.0))
            unit = "active users / reqs / qtr"

        return Reach(value=round(value, 1), unit=unit)

    def sample_impact(self) -> Impact:
        """
        Samples Impact factor according to standard enterprise distribution.
        """
        choices = [item for item in IMPACT_DISTRIBUTION]
        probs = [item[2] for item in IMPACT_DISTRIBUTION]
        idx = int(self.rng.choice(len(choices), p=probs))
        score, tier, _ = choices[idx]

        rationales = {
            ImpactTier.MINIMAL: "Minor operational cleanup or low-frequency telemetry adjustment.",
            ImpactTier.LOW: "Incremental efficiency optimization with modest engineering time savings.",
            ImpactTier.MEDIUM: "Standard architectural feature delivering measurable reliability and developer velocity gains.",
            ImpactTier.HIGH: "Significant infrastructure breakthrough removing major production bottlenecks and improving SLAs.",
            ImpactTier.MASSIVE: "Strategic platform initiative unlocking 10x scale, substantial cost reduction, or critical compliance.",
        }

        return Impact(
            score=score,
            tier=tier,
            rationale=rationales.get(tier, "Evaluated against quarterly strategic initiatives."),
        )

    def sample_confidence(self) -> Confidence:
        """
        Samples Confidence factor according to enterprise evidence criteria.
        """
        choices = [item for item in CONFIDENCE_DISTRIBUTION]
        probs = [item[2] for item in CONFIDENCE_DISTRIBUTION]
        idx = int(self.rng.choice(len(choices), p=probs))
        pct, tier, _ = choices[idx]

        evidences = {
            ConfidenceTier.LOW: "Exploratory moonshot / prototype with unverified production benchmarks.",
            ConfidenceTier.MEDIUM: "Proven in staging POC with synthetic load testing and clear technical feasibility.",
            ConfidenceTier.HIGH: "Validated with production shadow traffic, clear vendor SLA, and historical team velocity.",
        }

        return Confidence(
            percentage=pct,
            tier=tier,
            evidence=evidences.get(tier, "Based on team historical velocity and prototype results."),
        )

    def sample_effort(self, domain: str) -> Effort:
        """
        Samples Effort from a right-skewed distribution (Gamma/Log-normal) with domain-specific breakdowns.
        """
        # Person-weeks: typical median ~8 weeks, range 1 to 52 weeks
        shape = 2.5
        scale = 3.5
        total_weeks = float(self.rng.gamma(shape=shape, scale=scale))
        total_weeks = max(1.0, min(total_weeks, 52.0))
        total_weeks = round(total_weeks, 1)

        # Generate discipline breakdown matching domain characteristics
        if domain == "Frontend & Design Systems":
            weights = [0.55, 0.20, 0.10, 0.15]  # frontend, backend, infra, qa
        elif domain in ["Cloud Infrastructure & Reliability", "Security & Zero-Trust Architecture"]:
            weights = [0.05, 0.30, 0.55, 0.10]
        elif domain in ["Data Infrastructure & Analytics", "AI/ML Platform & Inference"]:
            weights = [0.05, 0.50, 0.35, 0.10]
        else:
            weights = [0.20, 0.45, 0.20, 0.15]

        # Add Dirichlet noise for variability
        noisy_weights = self.rng.dirichlet(np.array(weights) * 10)
        fe = round(float(noisy_weights[0] * total_weeks), 1)
        be = round(float(noisy_weights[1] * total_weeks), 1)
        infra = round(float(noisy_weights[2] * total_weeks), 1)
        qa = max(0.0, round(total_weeks - (fe + be + infra), 1))

        breakdown = EffortBreakdown(
            frontend=fe,
            backend=be,
            infra_devops=infra,
            qa_testing=qa,
        )

        return Effort(person_weeks=total_weeks, breakdown=breakdown)

    def generate_rice_factors(self, domain: str) -> RICEFactors:
        """
        Generates full RICE factors with calculated score.
        """
        reach = self.sample_reach(domain)
        impact = self.sample_impact()
        confidence = self.sample_confidence()
        effort = self.sample_effort(domain)

        rice_score = round(
            (reach.value * float(impact.score) * float(confidence.percentage)) / effort.person_weeks,
            4,
        )

        return RICEFactors(
            reach=reach,
            impact=impact,
            confidence=confidence,
            effort=effort,
            rice_score=rice_score,
        )

    def generate_project_id(self, domain: str, index: int) -> str:
        """
        Generates standard project identifier: proj_<domain>_2026_<idx:04d>
        """
        prefix = DOMAIN_PREFIXES.get(domain, "core")
        return f"proj_{prefix}_2026_{index:04d}"

    @classmethod
    def synthesize_dag_relations(
        cls,
        nodes: List[Dict[str, Any]],
        edge_probability: float = 0.03,
        max_upstream_per_node: int = 4,
    ) -> None:
        """
        Synthesizes realistic cross-project DAG dependency links.
        Orders projects topologically so edges only flow from earlier (foundational) to later projects.
        """
        n = len(nodes)
        if n <= 1:
            return

        # Map domain layer priorities: Core Infra & Security come first, Applications last
        domain_layer_map = {
            "Cloud Infrastructure & Reliability": 0,
            "Security & Zero-Trust Architecture": 1,
            "Data Infrastructure & Analytics": 2,
            "Developer Productivity & CI/CD": 2,
            "Core Backend & Microservices": 3,
            "AI/ML Platform & Inference": 4,
            "Search & Discovery": 4,
            "Frontend & Design Systems": 5,
        }

        # Sort indices primarily by domain layer to encourage realistic upstream -> downstream flow
        sorted_indices = sorted(
            range(n),
            key=lambda i: (
                domain_layer_map.get(nodes[i]["taxonomy"]["domain"], 3),
                i,
            ),
        )

        # Adjacency tracking
        upstream_map: Dict[str, Set[str]] = {nodes[i]["id"]: set() for i in range(n)}
        downstream_map: Dict[str, Set[str]] = {nodes[i]["id"]: set() for i in range(n)}

        # Build edges: earlier sorted nodes can be upstream to later sorted nodes
        for pos_j, idx_j in enumerate(sorted_indices):
            node_j = nodes[idx_j]
            id_j = node_j["id"]
            domain_j = node_j["taxonomy"]["domain"]

            # Candidate upstreams are earlier in topological order
            if pos_j == 0:
                continue

            candidates = sorted_indices[:pos_j]
            # Bias toward same-domain or foundational infra
            num_edges = random.choices(
                [0, 1, 2, 3],
                weights=[0.45, 0.35, 0.15, 0.05],
                k=1,
            )[0]
            num_edges = min(num_edges, max_upstream_per_node)

            if num_edges > 0 and candidates:
                # Sample candidates with probability boosted for same domain or foundational domains
                weights = []
                for idx_i in candidates:
                    cand_domain = nodes[idx_i]["taxonomy"]["domain"]
                    w = 3.0 if cand_domain == domain_j else 1.0
                    weights.append(w)
                prob_norm = np.array(weights) / sum(weights)
                chosen_cand_indices = np.random.choice(
                    candidates,
                    size=min(num_edges, len(candidates)),
                    replace=False,
                    p=prob_norm,
                )

                for idx_i in chosen_cand_indices:
                    id_i = nodes[idx_i]["id"]
                    upstream_map[id_j].add(id_i)
                    downstream_map[id_i].add(id_j)

        # Assign back to nodes
        for node in nodes:
            node_id = node["id"]
            node["graph_relations"]["upstream_dependencies"] = sorted(list(upstream_map[node_id]))
            node["graph_relations"]["downstream_impacted"] = sorted(list(downstream_map[node_id]))
