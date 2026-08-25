"""
Dataset builder and packaging engine for EKG Project Nodes and Evaluation Queries.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from smriti.embeddings.mrl_pipeline import MRLEmbeddingPipeline
from smriti.generators.llm_enricher import LLMEnricher
from smriti.generators.seed_generator import SeedGenerator
from smriti.generators.taxonomy import DOMAIN_TAXONOMY
from smriti.models.project_schema import EnterpriseProjectNode


class DatasetBuilder:
    """
    Orchestrates the generation, narrative enrichment, MRL embedding generation,
    and persistent serialization of enterprise project datasets.
    """

    def __init__(
        self,
        seed_generator: Optional[SeedGenerator] = None,
        llm_enricher: Optional[LLMEnricher] = None,
        embedding_pipeline: Optional[MRLEmbeddingPipeline] = None,
    ):
        self.seed_gen = seed_generator or SeedGenerator(seed=42)
        self.enricher = llm_enricher or LLMEnricher(prefer_offline=True)
        self.embedder = embedding_pipeline or MRLEmbeddingPipeline()

    def generate_dataset(
        self,
        num_items: int = 3500,
        num_queries: int = 500,
        show_progress: bool = True,
    ) -> Tuple[List[EnterpriseProjectNode], List[Dict[str, Any]]]:
        """
        Generates indexable enterprise project nodes and held-out evaluation query records.
        """
        domains = list(DOMAIN_TAXONOMY.keys())
        raw_nodes: List[Dict[str, Any]] = []

        # 1. Generate structural data & RICE metrics evenly across all 8 domains
        items_per_domain = (num_items + len(domains) - 1) // len(domains)
        index_counter = 0

        pbar = tqdm(total=num_items, desc="Generating Project Nodes", disable=not show_progress)
        for domain in domains:
            sub_domains = DOMAIN_TAXONOMY[domain]
            for i in range(items_per_domain):
                if index_counter >= num_items:
                    break
                # Cycle through sub-domains to guarantee full sub-domain coverage
                sub_meta = sub_domains[i % len(sub_domains)]
                epic = sub_meta.epics[i % len(sub_meta.epics)]

                taxonomy, _ = self.seed_gen.sample_taxonomy(domain=domain)
                # Override with guaranteed sub_domain and epic
                taxonomy.sub_domain = sub_meta.name
                taxonomy.epic = epic

                project_id = self.seed_gen.generate_project_id(domain, index_counter + 1)
                rice = self.seed_gen.generate_rice_factors(domain)

                context = self.enricher.enrich(
                    taxonomy=taxonomy,
                    sub_domain_meta=sub_meta,
                    rice_reach_val=rice.reach.value,
                    index=index_counter,
                )

                raw_nodes.append({
                    "id": project_id,
                    "taxonomy": taxonomy.model_dump(),
                    "semantic_context": context.model_dump(),
                    "rice_factors": rice.model_dump(),
                    "graph_relations": {
                        "upstream_dependencies": [],
                        "downstream_impacted": [],
                        "tags": list(set(sub_meta.tags + [taxonomy.domain.lower().replace(" ", "-")])),
                    },
                })
                index_counter += 1
                pbar.update(1)
        pbar.close()

        # 2. Synthesize cross-project DAG relations
        self.seed_gen.synthesize_dag_relations(raw_nodes)

        # 3. Generate MRL Embeddings
        unified_texts = []
        for raw in raw_nodes:
            tech_str = ", ".join(raw["semantic_context"]["tech_stack"])
            target_str = raw["semantic_context"].get("target_users") or "General Engineering"
            text = (
                f"search_document: [Domain: {raw['taxonomy']['domain']}] [Sub-domain: {raw['taxonomy']['sub_domain']}]\n"
                f"Title: {raw['semantic_context']['title']}\n"
                f"Problem: {raw['semantic_context']['problem_statement']}\n"
                f"Solution: {raw['semantic_context']['solution_proposal']}\n"
                f"Tech Stack: {tech_str}\n"
                f"Target Audience: {target_str}"
            )
            unified_texts.append(text)

        if show_progress:
            print("Encoding multi-resolution MRL embeddings (d0=64, d1=128, d2=768)...")
        mrl_embeddings_list = self.embedder.encode_text_batch(
            unified_texts,
            batch_size=64,
            show_progress_bar=show_progress,
        )

        # 4. Construct validated Pydantic EnterpriseProjectNode objects
        project_nodes: List[EnterpriseProjectNode] = []
        for raw, emb in zip(raw_nodes, mrl_embeddings_list):
            raw["embeddings"] = emb.model_dump()
            node = EnterpriseProjectNode.model_validate(raw)
            project_nodes.append(node)

        # 5. Generate held-out evaluation query items
        query_items = self._generate_test_queries(project_nodes, num_queries=num_queries)

        return project_nodes, query_items

    def _generate_test_queries(
        self,
        nodes: List[EnterpriseProjectNode],
        num_queries: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Generates structured test queries for evaluating routing recall (HAM-P)
        and associative completion (HAM-R).
        """
        num_queries = min(num_queries, len(nodes))
        sampled_nodes = self.seed_gen.rng.choice(nodes, size=num_queries, replace=False)
        query_records = []

        query_templates = [
            "We are seeking an architectural solution for: {title}. The core challenge is that {problem_snippet}.",
            "How can we implement {epic} to resolve {problem_snippet}?",
            "Initiative query: {title}. Target domain: {domain}, sub-domain: {sub_domain}.",
            "Design proposal request for {solution_snippet} using {tech_stack_snippet}.",
        ]

        query_texts = []
        for i, node in enumerate(sampled_nodes):
            prob_snippet = node.semantic_context.problem_statement.split(".")[0]
            sol_snippet = node.semantic_context.solution_proposal.split(".")[0]
            tech_snippet = ", ".join(node.semantic_context.tech_stack[:3])

            template = query_templates[i % len(query_templates)]
            query_prompt = template.format(
                title=node.semantic_context.title,
                problem_snippet=prob_snippet,
                solution_snippet=sol_snippet,
                epic=node.taxonomy.epic,
                domain=node.taxonomy.domain,
                sub_domain=node.taxonomy.sub_domain,
                tech_stack_snippet=tech_snippet,
            )
            query_text_serialized = f"search_query: {query_prompt}"
            query_texts.append((query_text_serialized, query_prompt, node))

        # Encode query vectors
        encoded_queries = self.embedder.encode_text_batch(
            [q[0] for q in query_texts],
            batch_size=64,
            show_progress_bar=False,
        )

        for i, (q_serialized, q_prompt, target_node) in enumerate(query_texts):
            emb = encoded_queries[i]
            query_records.append({
                "query_id": f"query_eval_2026_{i+1:04d}",
                "query_prompt": q_prompt,
                "query_text": q_serialized,
                "target_project_id": target_node.id,
                "ground_truth_taxonomy": {
                    "domain": target_node.taxonomy.domain,
                    "sub_domain": target_node.taxonomy.sub_domain,
                    "epic": target_node.taxonomy.epic,
                },
                "ground_truth_rice": {
                    "reach": target_node.rice_factors.reach.value,
                    "impact_score": float(target_node.rice_factors.impact.score),
                    "confidence_percentage": float(target_node.rice_factors.confidence.percentage),
                    "effort_person_weeks": target_node.rice_factors.effort.person_weeks,
                    "rice_score": target_node.rice_factors.rice_score,
                },
                "mrl_d0_64": emb.d0_64,
                "mrl_d1_128": emb.d1_128,
                "mrl_d2_768": emb.d2_768,
            })

        return query_records

    @classmethod
    def export_to_parquet(
        cls,
        nodes: List[EnterpriseProjectNode],
        output_path: str,
    ) -> None:
        """
        Exports project nodes to a columnar Parquet file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        flat_records = [node.to_flat_dict() for node in nodes]

        # Convert to Arrow Table
        table = pa.Table.from_pylist(flat_records)
        pq.write_table(table, output_path, compression="zstd")

    @classmethod
    def export_queries_to_parquet(
        cls,
        queries: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Exports queries to Parquet and JSON files.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        table = pa.Table.from_pylist(queries)
        pq.write_table(table, output_path, compression="zstd")

        # Also write JSON companion for easy inspection
        json_path = output_path.replace(".parquet", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(queries, f, indent=2)

    @classmethod
    def export_to_sqlite(
        cls,
        nodes: List[EnterpriseProjectNode],
        db_path: str,
    ) -> None:
        """
        Exports project nodes, graph edges, tech stacks, and embeddings to SQLite database.
        """
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Create schema
        cur.executescript("""
            CREATE TABLE projects (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                sub_domain TEXT NOT NULL,
                epic TEXT NOT NULL,
                title TEXT NOT NULL,
                problem_statement TEXT NOT NULL,
                solution_proposal TEXT NOT NULL,
                target_users TEXT,
                reach_value REAL NOT NULL,
                reach_unit TEXT,
                impact_score REAL NOT NULL,
                impact_tier TEXT NOT NULL,
                confidence_percentage REAL NOT NULL,
                confidence_tier TEXT NOT NULL,
                effort_person_weeks REAL NOT NULL,
                rice_score REAL NOT NULL,
                unified_text TEXT NOT NULL
            );

            CREATE TABLE project_tech_stacks (
                project_id TEXT NOT NULL,
                technology TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );

            CREATE TABLE graph_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, rel_type),
                FOREIGN KEY (source_id) REFERENCES projects(id),
                FOREIGN KEY (target_id) REFERENCES projects(id)
            );

            CREATE TABLE project_tags (
                project_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );

            CREATE TABLE project_embeddings (
                project_id TEXT PRIMARY KEY,
                d0_64_json TEXT NOT NULL,
                d1_128_json TEXT NOT NULL,
                d2_768_json TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );

            CREATE INDEX idx_projects_domain ON projects(domain);
            CREATE INDEX idx_projects_subdomain ON projects(sub_domain);
            CREATE INDEX idx_projects_rice ON projects(rice_score);
            CREATE INDEX idx_graph_source ON graph_edges(source_id);
            CREATE INDEX idx_graph_target ON graph_edges(target_id);
        """)

        # Insert records in batches
        proj_records = []
        tech_records = []
        tag_records = []
        edge_records = []
        emb_records = []

        for node in nodes:
            proj_records.append((
                node.id,
                node.taxonomy.domain,
                node.taxonomy.sub_domain,
                node.taxonomy.epic,
                node.semantic_context.title,
                node.semantic_context.problem_statement,
                node.semantic_context.solution_proposal,
                node.semantic_context.target_users,
                node.rice_factors.reach.value,
                node.rice_factors.reach.unit,
                float(node.rice_factors.impact.score),
                str(node.rice_factors.impact.tier.value if hasattr(node.rice_factors.impact.tier, "value") else node.rice_factors.impact.tier),
                float(node.rice_factors.confidence.percentage),
                str(node.rice_factors.confidence.tier.value if hasattr(node.rice_factors.confidence.tier, "value") else node.rice_factors.confidence.tier),
                node.rice_factors.effort.person_weeks,
                node.rice_factors.rice_score,
                node.to_unified_text(),
            ))

            for tech in node.semantic_context.tech_stack:
                tech_records.append((node.id, tech))

            for tag in node.graph_relations.tags:
                tag_records.append((node.id, tag))

            for up in node.graph_relations.upstream_dependencies:
                edge_records.append((up, node.id, "upstream_of"))

            for down in node.graph_relations.downstream_impacted:
                edge_records.append((node.id, down, "downstream_of"))

            if node.embeddings:
                emb_records.append((
                    node.id,
                    json.dumps(node.embeddings.d0_64),
                    json.dumps(node.embeddings.d1_128),
                    json.dumps(node.embeddings.d2_768),
                ))

        cur.executemany("INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", proj_records)
        cur.executemany("INSERT INTO project_tech_stacks VALUES (?,?)", tech_records)
        cur.executemany("INSERT INTO project_tags VALUES (?,?)", tag_records)
        cur.executemany("INSERT OR IGNORE INTO graph_edges VALUES (?,?,?)", edge_records)
        if emb_records:
            cur.executemany("INSERT INTO project_embeddings VALUES (?,?,?,?)", emb_records)

        conn.commit()
        conn.close()
