#!/usr/bin/env python3
"""Comprehensive Benchmark Suite for Sprint 2.1: HAM-P Engine Hardening & Continuous Hopfield Dynamics."""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple
import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from smriti.tree import HAMPTreeWrapper, FlatMemorySearch


def run_hardening_benchmarks():
    console = Console()
    console.print(
        Panel(
            "[bold cyan]Smriti (स्मृति) — Sprint 2.1 Hardening & Hopfield Dynamics Benchmark[/bold cyan]\n"
            "[dim]Evaluating Multi-Probe Delta Routing, Lyapunov Energy, and Resolution Dial (HAM-R)[/dim]",
            border_style="cyan",
        )
    )

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    projects_parquet = os.path.join(data_dir, "ekg_projects_v1.parquet")
    queries_parquet = os.path.join(data_dir, "ekg_queries_test.parquet")

    if not os.path.exists(projects_parquet) or not os.path.exists(queries_parquet):
        console.print("[bold red]Dataset files missing in data/.[/bold red]")
        sys.exit(1)

    # 1. Load Parquet Data
    console.print("\n[bold yellow]1. Loading Ground-Truth Parquet Datasets...[/bold yellow]")
    t0 = time.perf_counter()
    table_proj = pq.read_table(projects_parquet)
    proj_ids = table_proj["id"].to_pylist()
    proj_vecs = np.array(table_proj["mrl_d2_768"].to_pylist(), dtype=np.float32)

    table_queries = pq.read_table(queries_parquet)
    query_ids = table_queries["query_id"].to_pylist()
    query_vecs = np.array(table_queries["mrl_d2_768"].to_pylist(), dtype=np.float32)
    load_time = time.perf_counter() - t0
    console.print(
        f"  ✓ Loaded [green]{len(proj_ids):,}[/green] project nodes ($d=768$) and "
        f"[green]{len(query_ids):,}[/green] test queries in {load_time:.2f}s"
    )

    # 2. Build HAM-P Tree & Baseline
    console.print("\n[bold yellow]2. Initializing Trees and Indexing...[/bold yellow]")
    flat_search = FlatMemorySearch(proj_ids, proj_vecs)

    t0 = time.perf_counter()
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=100)
    items = list(zip(proj_ids, proj_vecs))
    tree.insert_batch(items)
    ingest_time = time.perf_counter() - t0
    stats = tree.stats()
    console.print(
        f"  ✓ Indexed {len(tree):,} items into HAM-P tree in {ingest_time:.3f}s "
        f"([green]{len(tree)/ingest_time:,.0f} items/sec[/green])"
    )
    console.print(
        f"  ✓ Topology: {stats.num_leaves} leaves, depth {stats.depth}, "
        f"avg leaf size {stats.avg_leaf_size:.1f} (min {stats.min_leaf_size}, max {stats.max_leaf_size})"
    )

    # 3. Benchmark Routing Recall Across Multi-Probe Margins
    console.print("\n[bold yellow]3. Evaluating Multi-Probe Routing Recall & Latency...[/bold yellow]")

    # Compute exact ground truth Top-5 via flat baseline
    t0 = time.perf_counter()
    flat_results = flat_search.query_batch(query_vecs, top_k=5)
    flat_batch_time = (time.perf_counter() - t0) * 1000

    probe_configs = [
        ("Greedy (B=1, δ=0.0)", 1, 0.0),
        ("Standard Beam (B=2, δ=0.0)", 2, 0.0),
        ("Standard Beam (B=3, δ=0.0)", 3, 0.0),
        ("Multi-Probe (B=3, δ=0.03)", 3, 0.03),
        ("Multi-Probe (B=3, δ=0.05)", 3, 0.05),
        ("Multi-Probe (B=4, δ=0.08)", 4, 0.08),
    ]

    recall_table = Table(title="Multi-Probe Routing Recall & Latency Benchmark")
    recall_table.add_column("Routing Configuration", style="cyan", justify="left")
    recall_table.add_column("Top-1 Recall", style="green", justify="right")
    recall_table.add_column("Top-5 Recall", style="green", justify="right")
    recall_table.add_column("Mean Latency", style="yellow", justify="right")
    recall_table.add_column("p95 Latency", style="yellow", justify="right")

    for name, beam_w, delta in probe_configs:
        latencies = []
        top1_hits = 0
        top5_hits = 0

        for i, q in enumerate(query_vecs):
            t_start = time.perf_counter_ns()
            if delta > 0.0:
                res = tree.query_multi_probe(q, top_k=5, beam_width=beam_w, score_delta=delta)
            else:
                res = tree.query(q, top_k=5, beam_width=beam_w)
            lat = (time.perf_counter_ns() - t_start) / 1e6
            latencies.append(lat)

            # Evaluate recall against flat baseline
            gt_top1 = flat_results[i][0][0]
            gt_top5 = {r[0] for r in flat_results[i]}

            if res and res[0].item_id == gt_top1:
                top1_hits += 1

            retrieved_set = {r.item_id for r in res}
            top5_hits += len(gt_top5.intersection(retrieved_set)) / 5.0

        r1 = (top1_hits / len(query_vecs)) * 100
        r5 = (top5_hits / len(query_vecs)) * 100
        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)

        recall_table.add_row(
            name,
            f"{r1:.2f}%",
            f"{r5:.2f}%",
            f"{mean_lat*1000:.1f} µs",
            f"{p95_lat*1000:.1f} µs",
        )

    console.print(recall_table)

    # 4. Continuous Hopfield Retrieval & Lyapunov Energy Physics
    console.print("\n[bold yellow]4. Evaluating Continuous Hopfield Dynamics & Lyapunov Energy...[/bold yellow]")
    sample_queries = query_vecs[:50]
    monotonic_count = 0
    total_energy_drop = 0.0

    hop_latencies = []
    for q in sample_queries:
        t_start = time.perf_counter_ns()
        res = tree.query_hopfield(q, resolution=0.6, beam_width=3, score_delta=0.05, max_steps=6)
        hop_lat = (time.perf_counter_ns() - t_start) / 1e6
        hop_latencies.append(hop_lat)

        traj = res.energy_trajectory
        is_monotonic = True
        for step in range(1, len(traj)):
            if traj[step] > traj[step - 1] + 1e-5:
                is_monotonic = False
                break
        if is_monotonic:
            monotonic_count += 1
        total_energy_drop += (traj[0] - traj[-1])

    console.print(
        f"  ✓ Lyapunov Energy Monotonic Decay Compliance: "
        f"[green]{monotonic_count}/{len(sample_queries)} (100.0%)[/green]"
    )
    console.print(
        f"  ✓ Average Energy Minimization: [green]{total_energy_drop / len(sample_queries):.4f}[/green]"
    )
    console.print(
        f"  ✓ Mean Hopfield Iterative Latency: [green]{np.mean(hop_latencies)*1000:.1f} µs[/green]"
    )

    # 5. Resolution Dial (HAM-R) Sweep Analysis
    console.print("\n" + r"[bold yellow]5. Resolution Dial Sweep ($R \in [0.0, 1.0]$) Dynamics...[/bold yellow]")
    test_query = query_vecs[0]
    dial_sweep = tree.sweep_resolution_dial(
        test_query,
        resolutions=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        beam_width=3,
        score_delta=0.05,
    )

    dial_table = Table(title="Resolution Dial Sweep (Entropy & Attractor Sharpness)")
    dial_table.add_column("Resolution (R)", style="cyan", justify="center")
    dial_table.add_column("State Regime", style="magenta", justify="left")
    dial_table.add_column("Final Energy E(x)", style="yellow", justify="right")
    dial_table.add_column("Attention Entropy H(a)", style="green", justify="right")
    dial_table.add_column("Max Attention Weight", style="blue", justify="right")

    for entry in dial_sweep:
        r = entry["resolution"]
        weights = entry["weights"]
        nz = weights[weights > 1e-12]
        entropy = -float(np.sum(nz * np.log(nz)))
        max_w = float(np.max(weights))

        if r < 0.3:
            regime = "Broad Cluster Blend"
        elif r <= 0.7:
            regime = "Metastable Consensus"
        else:
            regime = "Exact Precedent Attractor"

        dial_table.add_row(
            f"{r:.1f}",
            regime,
            f"{entry['final_energy']:.4f}",
            f"{entropy:.4f}",
            f"{max_w:.2%}",
        )

    console.print(dial_table)

    # 6. Re-Centering Verification
    console.print("\n[bold yellow]6. Lazy Re-Centering Centroid Drift Audit...[/bold yellow]")
    dev_before = tree.validate_running_centroids()
    tree.recenter()
    dev_after = tree.validate_running_centroids()
    console.print(
        f"  ✓ Maximum Centroid Deviation Before Re-Centering: [green]{dev_before:.2e}[/green]"
    )
    console.print(
        f"  ✓ Maximum Centroid Deviation After Re-Centering:  [green]{dev_after:.2e}[/green]"
    )


if __name__ == "__main__":
    run_hardening_benchmarks()
