#!/usr/bin/env python3
"""
Validation script for Sprint 1 Acceptance Criteria.
Checks dataset size, schema validity, MRL vector unit-norm integrity, and RICE factor distributions.
"""

import argparse
import os
import sys
import numpy as np
import pyarrow.parquet as pq

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from smriti.models.project_schema import (
    Confidence,
    Effort,
    EnterpriseProjectNode,
    Impact,
    Reach,
    RICEFactors,
    Taxonomy,
    SemanticContext,
    GraphRelations,
    MRLEmbeddings,
)
from smriti.utils.logging import console, print_header, print_metric_table


def validate_dataset(parquet_path: str) -> bool:
    print_header(f"Validating Dataset: {parquet_path}")

    if not os.path.exists(parquet_path):
        console.print(f"[red]Error: Parquet file '{parquet_path}' does not exist![/red]")
        return False

    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    num_records = len(records)

    console.print(f"Loaded [bold cyan]{num_records}[/bold cyan] records from Parquet.")

    # -------------------------------------------------------------
    # Acceptance Criterion 1: Dataset Size & Domain/Sub-Domain Coverage
    # -------------------------------------------------------------
    domains = set()
    sub_domains = set()
    for r in records:
        domains.add(r["domain"])
        sub_domains.add(r["sub_domain"])

    pass_size = num_records >= 3000
    pass_domains = len(domains) >= 8
    pass_subdomains = len(sub_domains) >= 24

    console.print(f"\n[bold]1. Dataset Size & Diversity Check:[/bold]")
    console.print(f"  • Total Records: {num_records} (Target: >= 3,000) -> [{'green' if pass_size else 'red'}]{'PASS' if pass_size else 'FAIL'}[/]")
    console.print(f"  • Unique Domains: {len(domains)} (Target: >= 8) -> [{'green' if pass_domains else 'red'}]{'PASS' if pass_domains else 'FAIL'}[/]")
    console.print(f"  • Unique Sub-Domains: {len(sub_domains)} (Target: >= 24) -> [{'green' if pass_subdomains else 'red'}]{'PASS' if pass_subdomains else 'FAIL'}[/]")

    # -------------------------------------------------------------
    # Acceptance Criterion 2: Schema Validity
    # -------------------------------------------------------------
    schema_failures = 0
    for r in records:
        try:
            # Reconstruct model object to verify 100% schema compliance
            node_dict = {
                "id": r["id"],
                "taxonomy": {
                    "domain": r["domain"],
                    "sub_domain": r["sub_domain"],
                    "epic": r["epic"],
                },
                "semantic_context": {
                    "title": r["title"],
                    "problem_statement": r["problem_statement"],
                    "solution_proposal": r["solution_proposal"],
                    "tech_stack": r["tech_stack"],
                    "target_users": r["target_users"],
                },
                "rice_factors": {
                    "reach": {"value": r["reach_value"], "unit": r["reach_unit"]},
                    "impact": {
                        "score": r["impact_score"],
                        "tier": r["impact_tier"],
                        "rationale": r["impact_rationale"],
                    },
                    "confidence": {
                        "percentage": r["confidence_percentage"],
                        "tier": r["confidence_tier"],
                        "evidence": r["confidence_evidence"],
                    },
                    "effort": {
                        "person_weeks": r["effort_person_weeks"],
                        "breakdown": {
                            "frontend": r.get("effort_frontend", 0.0),
                            "backend": r.get("effort_backend", 0.0),
                            "infra_devops": r.get("effort_infra", 0.0),
                            "qa_testing": r.get("effort_qa", 0.0),
                        },
                    },
                    "rice_score": r["rice_score"],
                },
                "graph_relations": {
                    "upstream_dependencies": r["upstream_dependencies"],
                    "downstream_impacted": r["downstream_impacted"],
                    "tags": r["tags"],
                },
                "embeddings": {
                    "d0_64": r["mrl_d0_64"],
                    "d1_128": r["mrl_d1_128"],
                    "d2_768": r["mrl_d2_768"],
                } if r.get("mrl_d0_64") is not None else None,
            }
            EnterpriseProjectNode.model_validate(node_dict)
        except Exception as e:
            schema_failures += 1
            if schema_failures <= 3:
                console.print(f"[red]Validation failure on {r['id']}: {e}[/red]")

    pass_schema = schema_failures == 0
    console.print(f"\n[bold]2. Pydantic Schema Validity Check:[/bold]")
    console.print(f"  • Schema Failures: {schema_failures} / {num_records} -> [{'green' if pass_schema else 'red'}]{'PASS' if pass_schema else 'FAIL'}[/]")

    # -------------------------------------------------------------
    # Acceptance Criterion 3: MRL Integrity (Unit-Norm Verification)
    # -------------------------------------------------------------
    norm_failures_d0 = 0
    norm_failures_d1 = 0
    norm_failures_d2 = 0
    max_error = 0.0

    for r in records:
        for slice_key, dim, failure_counter_name in [
            ("mrl_d0_64", 64, "d0"),
            ("mrl_d1_128", 128, "d1"),
            ("mrl_d2_768", 768, "d2"),
        ]:
            vec_list = r.get(slice_key)
            if vec_list is None or len(vec_list) != dim:
                if failure_counter_name == "d0":
                    norm_failures_d0 += 1
                elif failure_counter_name == "d1":
                    norm_failures_d1 += 1
                else:
                    norm_failures_d2 += 1
                continue

            arr = np.array(vec_list, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            err = abs(norm - 1.0)
            max_error = max(max_error, err)

            if err > 1e-4:
                if failure_counter_name == "d0":
                    norm_failures_d0 += 1
                elif failure_counter_name == "d1":
                    norm_failures_d1 += 1
                else:
                    norm_failures_d2 += 1

    pass_mrl = (norm_failures_d0 + norm_failures_d1 + norm_failures_d2) == 0
    console.print(f"\n[bold]3. MRL Vector Normalization Integrity Check:[/bold]")
    console.print(f"  • d0 (64) Normalization Failures: {norm_failures_d0}")
    console.print(f"  • d1 (128) Normalization Failures: {norm_failures_d1}")
    console.print(f"  • d2 (768) Normalization Failures: {norm_failures_d2}")
    console.print(f"  • Max Vector Norm Error: {max_error:.2e} (Threshold: < 1e-4) -> [{'green' if pass_mrl else 'red'}]{'PASS' if pass_mrl else 'FAIL'}[/]")

    # -------------------------------------------------------------
    # Acceptance Criterion 4: RICE Factor Distribution & Mathematical Exactness
    # -------------------------------------------------------------
    reaches = np.array([r["reach_value"] for r in records])
    efforts = np.array([r["effort_person_weeks"] for r in records])
    impact_scores = [r["impact_score"] for r in records]
    conf_scores = [r["confidence_percentage"] for r in records]

    # Math consistency check: rice_score == (reach * impact * conf) / effort
    rice_math_errors = 0
    for r in records:
        expected = (r["reach_value"] * r["impact_score"] * r["confidence_percentage"]) / r["effort_person_weeks"]
        if abs(r["rice_score"] - expected) > 1e-2:
            rice_math_errors += 1

    pass_math = rice_math_errors == 0
    console.print(f"\n[bold]4. RICE Mathematical Consistency & Distributions:[/bold]")
    console.print(f"  • Reach Distribution: Min={np.min(reaches):.1f}, Median={np.median(reaches):.1f}, Max={np.max(reaches):.1f}")
    console.print(f"  • Effort Distribution: Min={np.min(efforts):.1f}, Median={np.median(efforts):.1f}, Max={np.max(efforts):.1f}")
    console.print(f"  • Impact Scores Present: {sorted(list(set(impact_scores)))}")
    console.print(f"  • Confidence Percentages: {sorted(list(set(conf_scores)))}")
    console.print(f"  • RICE Score Calculation Errors: {rice_math_errors} -> [{'green' if pass_math else 'red'}]{'PASS' if pass_math else 'FAIL'}[/]")

    all_passed = pass_size and pass_domains and pass_subdomains and pass_schema and pass_mrl and pass_math

    summary_metrics = {
        "Total Validated Nodes": num_records,
        "Domains (>=8)": f"{len(domains)} [{'green' if pass_domains else 'red'}]{'✓' if pass_domains else '✗'}[/]",
        "Sub-Domains (>=24)": f"{len(sub_domains)} [{'green' if pass_subdomains else 'red'}]{'✓' if pass_subdomains else '✗'}[/]",
        "Schema Validation": f"100% [{'green' if pass_schema else 'red'}]{'✓' if pass_schema else '✗'}[/]",
        "MRL Normalization": f"100% [Max Err: {max_error:.1e}] [{'green' if pass_mrl else 'red'}]{'✓' if pass_mrl else '✗'}[/]",
        "RICE Integrity": f"100% [{'green' if pass_math else 'red'}]{'✓' if pass_math else '✗'}[/]",
        "Acceptance Status": f"[{'bold green' if all_passed else 'bold red'}]{'ACCEPTED (PASS)' if all_passed else 'REJECTED (FAIL)'}[/]",
    }

    console.print("\n")
    print_metric_table(summary_metrics, title="Sprint 1 Acceptance Verification")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Sprint 1 EKG Dataset against Acceptance Criteria.")
    parser.add_argument("--parquet-file", type=str, default="data/ekg_projects_v1.parquet", help="Path to Parquet dataset")
    args = parser.parse_args()

    success = validate_dataset(args.parquet_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
