"""
Unit tests for taxonomy definitions, seed distributions, and DAG relation synthesizer.
"""

from smriti.generators.llm_enricher import LLMEnricher
from smriti.generators.seed_generator import SeedGenerator
from smriti.generators.taxonomy import DOMAIN_TAXONOMY


def test_taxonomy_coverage():
    domains = list(DOMAIN_TAXONOMY.keys())
    assert len(domains) == 8, f"Expected 8 domains, got {len(domains)}"

    total_subdomains = 0
    for domain, sub_domains in DOMAIN_TAXONOMY.items():
        assert len(sub_domains) >= 4, f"Domain '{domain}' has {len(sub_domains)} sub-domains (< 4)"
        total_subdomains += len(sub_domains)

    assert total_subdomains >= 24, f"Expected >= 24 sub-domains, got {total_subdomains}"


def test_seed_generator_rice_sampling():
    gen = SeedGenerator(seed=123)
    domain = "Developer Productivity & CI/CD"

    rice = gen.generate_rice_factors(domain)
    assert rice.reach.value > 0
    assert rice.effort.person_weeks > 0
    assert float(rice.impact.score) in [0.25, 0.5, 1.0, 2.0, 3.0]
    assert float(rice.confidence.percentage) in [0.5, 0.8, 1.0]

    expected_score = round(
        (rice.reach.value * float(rice.impact.score) * float(rice.confidence.percentage))
        / rice.effort.person_weeks,
        4,
    )
    assert abs(rice.rice_score - expected_score) < 1e-3


def test_dag_relations_synthesis():
    gen = SeedGenerator(seed=42)
    nodes = []
    domains = list(DOMAIN_TAXONOMY.keys())

    for i in range(50):
        domain = domains[i % len(domains)]
        taxonomy, sub_meta = gen.sample_taxonomy(domain=domain)
        nodes.append({
            "id": gen.generate_project_id(domain, i + 1),
            "taxonomy": taxonomy.model_dump(),
            "graph_relations": {"upstream_dependencies": [], "downstream_impacted": [], "tags": []},
        })

    SeedGenerator.synthesize_dag_relations(nodes)

    # Verify bidirectional link consistency
    node_map = {n["id"]: n for n in nodes}
    for n in nodes:
        for up_id in n["graph_relations"]["upstream_dependencies"]:
            assert up_id in node_map
            assert n["id"] in node_map[up_id]["graph_relations"]["downstream_impacted"]


def test_llm_enricher_deterministic_fallback():
    enricher = LLMEnricher(prefer_offline=True)
    gen = SeedGenerator(seed=99)
    taxonomy, sub_meta = gen.sample_taxonomy()

    context = enricher.enrich(
        taxonomy=taxonomy,
        sub_domain_meta=sub_meta,
        rice_reach_val=350.0,
        index=1,
    )

    assert context.title
    assert context.problem_statement
    assert context.solution_proposal
    assert len(context.tech_stack) >= 2
    assert context.target_users
