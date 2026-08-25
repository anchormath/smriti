"""
Integration tests for dataset building, Parquet packaging, and SQLite storage.
"""

import os
import sqlite3
import tempfile
import pyarrow.parquet as pq

from smriti.data.dataset_builder import DatasetBuilder
from smriti.embeddings.mrl_pipeline import MRLEmbeddingPipeline
from smriti.generators.llm_enricher import LLMEnricher
from smriti.generators.seed_generator import SeedGenerator


def test_dataset_generation_and_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_ekg.parquet")
        sqlite_path = os.path.join(tmpdir, "test_ekg.db")
        queries_path = os.path.join(tmpdir, "test_queries.parquet")

        seed_gen = SeedGenerator(seed=42)
        enricher = LLMEnricher(prefer_offline=True)
        embedder = MRLEmbeddingPipeline(use_deterministic_fallback=True)

        builder = DatasetBuilder(
            seed_generator=seed_gen,
            llm_enricher=enricher,
            embedding_pipeline=embedder,
        )

        nodes, queries = builder.generate_dataset(num_items=40, num_queries=10, show_progress=False)
        assert len(nodes) == 40
        assert len(queries) == 10

        # Export to Parquet
        builder.export_to_parquet(nodes, parquet_path)
        assert os.path.exists(parquet_path)
        table = pq.read_table(parquet_path)
        assert table.num_rows == 40
        assert "mrl_d0_64" in table.column_names
        assert "mrl_d1_128" in table.column_names
        assert "mrl_d2_768" in table.column_names

        # Export to SQLite
        builder.export_to_sqlite(nodes, sqlite_path)
        assert os.path.exists(sqlite_path)
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM projects")
        count = cur.fetchone()[0]
        assert count == 40

        cur.execute("SELECT COUNT(*) FROM project_embeddings")
        emb_count = cur.fetchone()[0]
        assert emb_count == 40
        conn.close()

        # Export queries
        builder.export_queries_to_parquet(queries, queries_path)
        assert os.path.exists(queries_path)
        json_path = queries_path.replace(".parquet", ".json")
        assert os.path.exists(json_path)
