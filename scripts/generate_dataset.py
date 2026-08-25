#!/usr/bin/env python3
"""
CLI script to generate, embed, and package the Enterprise Knowledge Graph (EKG) dataset.
"""

import argparse
import os
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from smriti.data.dataset_builder import DatasetBuilder
from smriti.embeddings.mrl_pipeline import MRLEmbeddingPipeline
from smriti.generators.llm_enricher import LLMEnricher
from smriti.generators.seed_generator import SeedGenerator
from smriti.utils.logging import console, print_header, print_metric_table


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EKG dataset with MRL embeddings.")
    parser.add_argument("--num-items", type=int, default=3500, help="Number of project nodes to generate (default: 3500)")
    parser.add_argument("--num-queries", type=int, default=500, help="Number of test queries to generate (default: 500)")
    parser.add_argument("--parquet-out", type=str, default="data/ekg_projects_v1.parquet", help="Parquet output path")
    parser.add_argument("--sqlite-out", type=str, default="data/ekg_projects_v1.db", help="SQLite DB output path")
    parser.add_argument("--queries-out", type=str, default="data/ekg_queries_test.parquet", help="Test queries output path")
    parser.add_argument("--model-name", type=str, default="nomic-ai/nomic-embed-text-v1.5", help="Embedding model name")
    parser.add_argument("--deterministic-embeddings", action="store_true", help="Force deterministic pseudo-MRL embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print_header("Smriti Sprint 1: EKG Dataset & MRL Embedding Engine")
    console.print(f"[cyan]Target items:[/cyan] {args.num_items} | [cyan]Test queries:[/cyan] {args.num_queries}")
    console.print(f"[cyan]Parquet target:[/cyan] {args.parquet_out} | [cyan]SQLite target:[/cyan] {args.sqlite_out}")

    start_time = time.time()

    seed_gen = SeedGenerator(seed=args.seed)
    enricher = LLMEnricher(prefer_offline=True)
    embedder = MRLEmbeddingPipeline(
        model_name=args.model_name,
        use_deterministic_fallback=args.deterministic_embeddings,
    )

    builder = DatasetBuilder(
        seed_generator=seed_gen,
        llm_enricher=enricher,
        embedding_pipeline=embedder,
    )

    console.print("\n[bold green]1. Generating project records and multi-resolution embeddings...[/bold green]")
    nodes, queries = builder.generate_dataset(
        num_items=args.num_items,
        num_queries=args.num_queries,
        show_progress=True,
    )

    console.print("\n[bold green]2. Exporting dataset to Parquet...[/bold green]")
    builder.export_to_parquet(nodes, args.parquet_out)
    console.print(f" Saved Parquet to [bold]{args.parquet_out}[/bold]")

    console.print("\n[bold green]3. Exporting dataset to SQLite...[/bold green]")
    builder.export_to_sqlite(nodes, args.sqlite_out)
    console.print(f" Saved SQLite to [bold]{args.sqlite_out}[/bold]")

    console.print("\n[bold green]4. Exporting test queries...[/bold green]")
    builder.export_queries_to_parquet(queries, args.queries_out)
    console.print(f" Saved queries to [bold]{args.queries_out}[/bold] and JSON companion.")

    elapsed = time.time() - start_time
    parquet_size_mb = os.path.getsize(args.parquet_out) / (1024 * 1024)
    sqlite_size_mb = os.path.getsize(args.sqlite_out) / (1024 * 1024)

    metrics = {
        "Total Projects Generated": len(nodes),
        "Total Test Queries": len(queries),
        "Total Processing Time": f"{elapsed:.2f}s",
        "Parquet Size": f"{parquet_size_mb:.2f} MB",
        "SQLite DB Size": f"{sqlite_size_mb:.2f} MB",
        "Device Used": embedder.device,
        "Model": embedder.model_name if embedder.model is not None else "Deterministic Pseudo-MRL",
    }

    print_metric_table(metrics, title="Generation Summary")
    console.print("[bold green]✓ Sprint 1 Dataset Generation Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    main()
