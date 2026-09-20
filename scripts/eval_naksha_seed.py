#!/usr/bin/env python3
"""Initial integration test: run Smriti's HAM-P/HAM-R engine against Naksha's real seeded graph.

Reads the committed Naksha seed JSON directly (no SurrealDB/Docker required — see
claude/adr-smriti-integration.md in the Naksha Claude project for why this is decoupled
from the app stack). Flattens every domain record into a `[Domain: X] [Sub-domain: Y] ...`
text description, embeds it, builds an HAM-P tree, and runs a handful of sanity queries:

  1. Tree topology stats after streaming insertion of the whole seeded graph.
  2. Self-recall: does querying with a node's own vector return itself as top-1?
  3. Domain purity: for a sample of nodes, what fraction of their top-5 neighbors share
     the same Sangh/Kala/Krama/Vyapar domain? (tests whether the root router's domain
     split is doing anything sensible on real, small, heterogeneous data)
  4. Edge-discovery sanity check: for one known EKG edge (a Krama service that
     `supports_features` a Vyapar product_feature), is Smriti's associative distance
     between them smaller than a random cross-domain pair? This is the "does Smriti's
     nearness track the graph's confirmed edges at all" check from the ADR.
  5. One Resolution Dial sweep (R: 0 -> 1) on a single query, to see the energy/output
     behavior described in HAM-R.

NOTE on scale (see adr-smriti-integration.md): the seeded graph is ~53 nodes total across
all four domains, vs. the 3,500-node synthetic benchmark in Smriti's own README. This script
validates correctness/routing/plumbing on real data — it is NOT a recall/latency benchmark;
the tree is too small for that comparison to mean anything yet.

Usage:
    # from the smriti/ repo root, with its .venv activated
    python scripts/eval_naksha_seed.py                       # real Nomic embeddings
    python scripts/eval_naksha_seed.py --fallback-embeddings  # no model download, quick smoke test
    python scripts/eval_naksha_seed.py --naksha-seed-dir /path/to/naksha/naksha/seed/data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from smriti.embeddings.mrl_pipeline import MRLEmbeddingPipeline
from smriti.tree.hamp_tree import HAMPTreeWrapper

console = Console()

# id -> Naksha module. Keep in sync with sangh.md / kala.md / kram.md / vyapar.md.
DOMAIN_MAP = {
    "business_units": ("Sangh", "business_unit"),
    "departments": ("Sangh", "department"),
    "teams": ("Sangh", "team"),
    "roles": ("Sangh", "role"),
    "employees": ("Sangh", "employee"),
    "career_tracks": ("Kala", "career_track"),
    "skills": ("Kala", "skill"),
    "expectations": ("Kala", "expectation"),
    "technologies": ("Krama", "technology"),
    "data_stores": ("Krama", "data_store"),
    "message_buses": ("Krama", "message_bus"),
    "services": ("Krama", "service"),
    "projects": ("Krama", "project"),
    "capabilities": ("Vyapar", "capability"),
    "product_lines": ("Vyapar", "product_line"),
    "products": ("Vyapar", "product"),
    "product_features": ("Vyapar", "product_feature"),
    "market_segments": ("Vyapar", "market_segment"),
    "customers": ("Vyapar", "customer"),
}


def record_to_text(record: dict[str, Any], domain: str, subdomain: str) -> str:
    """Serialize a record's non-id fields into the `[Domain: X] [Sub-domain: Y] ...` shape
    the fallback embedder (and, informally, a real text embedder too) expects."""
    parts = [f"[Domain: {domain}] [Sub-domain: {subdomain}]"]
    name = record.get("name") or record.get("title") or record.get("id", "")
    parts.append(str(name))
    for k, v in record.items():
        if k in ("id", "name", "title"):
            continue
        if isinstance(v, (str, int, float)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list) and v and isinstance(v[0], str):
            parts.append(f"{k}: {', '.join(v)}")
    return " | ".join(parts)


def load_seed_records(seed_dir: Path) -> list[tuple[str, str, str, str]]:
    """Returns list of (item_id, domain, subdomain, text)."""
    records: list[tuple[str, str, str, str]] = []

    for fname in ("org_skills.json", "systems.json", "business.json"):
        fpath = seed_dir / fname
        if not fpath.exists():
            console.print(f"[yellow]Skipping missing {fpath}[/yellow]")
            continue
        data = json.loads(fpath.read_text())
        for table_key, rows in data.items():
            if table_key not in DOMAIN_MAP:
                continue  # edge/relation tables (feature_segments, feature_releases, ...)
            domain, subdomain = DOMAIN_MAP[table_key]
            for row in rows:
                rid = row.get("id", f"{subdomain}:{len(records)}")
                records.append((rid, domain, subdomain, record_to_text(row, domain, subdomain)))

    # sample_reviews.json is a flat list, not a domain dict
    reviews_path = seed_dir / "sample_reviews.json"
    if reviews_path.exists():
        reviews = json.loads(reviews_path.read_text())
        for i, row in enumerate(reviews):
            rid = row.get("id", f"review:{i}")
            records.append((rid, "Kala", "review", record_to_text(row, "Kala", "review")))

    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--naksha-seed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "naksha" / "naksha" / "seed" / "data",
        help="Path to naksha/naksha/seed/data (sibling checkout by default)",
    )
    ap.add_argument(
        "--fallback-embeddings",
        action="store_true",
        help="Use the deterministic pseudo-MRL fallback instead of downloading nomic-embed-text-v1.5",
    )
    args = ap.parse_args()

    console.print(Panel(
        "[bold cyan]Smriti x Naksha — Initial Integration Test[/bold cyan]\n"
        "[dim]Real seeded graph, not the synthetic 3,500-node benchmark — see adr-smriti-integration.md[/dim]",
        border_style="cyan",
    ))

    if not args.naksha_seed_dir.exists():
        console.print(f"[bold red]Seed dir not found: {args.naksha_seed_dir}[/bold red]")
        console.print("Pass --naksha-seed-dir explicitly if naksha/ isn't checked out next to smriti/.")
        raise SystemExit(1)

    records = load_seed_records(args.naksha_seed_dir)
    console.print(f"Loaded [bold]{len(records)}[/bold] real Naksha nodes from {args.naksha_seed_dir}")

    by_domain: dict[str, int] = {}
    for _, domain, _, _ in records:
        by_domain[domain] = by_domain.get(domain, 0) + 1
    dom_table = Table(title="Nodes by Naksha module")
    dom_table.add_column("Module")
    dom_table.add_column("Count", justify="right")
    for d, c in sorted(by_domain.items()):
        dom_table.add_row(d, str(c))
    console.print(dom_table)

    # 1. Embed
    console.print("\n[bold yellow]1. Embedding (MRL d0=64/d1=128/d2=768)...[/bold yellow]")
    pipeline = MRLEmbeddingPipeline(use_deterministic_fallback=args.fallback_embeddings)
    if pipeline.model is None and not args.fallback_embeddings:
        console.print(f"[yellow]Model load failed ({pipeline._init_error}); using deterministic fallback.[/yellow]")
    texts = [t for _, _, _, t in records]
    embeddings = pipeline.encode_text_batch(texts, show_progress_bar=False)

    # 2. Build tree via streaming insertion (mirrors Naksha's own eventual ingestion pattern)
    console.print("\n[bold yellow]2. Building HAM-P tree via streaming insertion...[/bold yellow]")
    tree = HAMPTreeWrapper(d0=64, d1=128, d2=768, max_leaf_capacity=25)
    id_domain = {}
    id_vec = {}
    for (item_id, domain, subdomain, _), emb in zip(records, embeddings):
        vec = np.array(emb.d2_768, dtype=np.float32)
        tree.insert(item_id, vec)
        id_domain[item_id] = (domain, subdomain)
        id_vec[item_id] = vec

    stats = tree.stats()
    stats_table = Table(title="HAM-P Tree Topology (real Naksha seed)")
    stats_table.add_column("Metric")
    stats_table.add_column("Value", style="bold green")
    stats_table.add_row("Total items indexed", str(stats.num_items))
    stats_table.add_row("Total leaf nodes", str(stats.num_leaves))
    stats_table.add_row("Tree depth", str(stats.depth))
    stats_table.add_row("Avg leaf size", f"{stats.avg_leaf_size:.1f}")
    console.print(stats_table)

    # 3. Self-recall check
    console.print("\n[bold yellow]3. Self-recall check (query with own vector -> expect self as top-1)...[/bold yellow]")
    random.seed(7)
    sample_ids = random.sample(list(id_vec.keys()), min(10, len(id_vec)))
    self_hits = 0
    for item_id in sample_ids:
        results = tree.query(id_vec[item_id], top_k=1, beam_width=3)
        if results and results[0].item_id == item_id:
            self_hits += 1
    console.print(f"Self-recall: [bold]{self_hits}/{len(sample_ids)}[/bold] "
                  f"({'PASS' if self_hits == len(sample_ids) else 'CHECK — see note below'})")
    if self_hits < len(sample_ids):
        console.print("[dim]Less than perfect self-recall on this few nodes is a routing/beam-width "
                      "signal worth digging into, not necessarily a bug — widen beam_width and re-check.[/dim]")

    # 4. Domain purity of top-5 neighbors
    console.print("\n[bold yellow]4. Domain purity — do top-5 neighbors share the query's module?...[/bold yellow]")
    purity_scores = []
    for item_id in sample_ids:
        results = tree.query(id_vec[item_id], top_k=5, beam_width=3)
        own_domain = id_domain[item_id][0]
        neighbor_domains = [id_domain[r.item_id][0] for r in results if r.item_id in id_domain]
        if neighbor_domains:
            purity = sum(1 for d in neighbor_domains if d == own_domain) / len(neighbor_domains)
            purity_scores.append(purity)
    if purity_scores:
        console.print(f"Mean domain purity across {len(purity_scores)} sampled queries: "
                       f"[bold]{np.mean(purity_scores):.2f}[/bold] (1.0 = neighbors always same module)")

    # 5. Edge-discovery sanity check: a real Krama<->Vyapar edge should be "close"
    console.print("\n[bold yellow]5. Edge-discovery sanity check...[/bold yellow]")
    known_pair = ("service:api_gateway", "product_feature:real_time_recommendations")
    if known_pair[0] in id_vec and known_pair[1] in id_vec:
        edge_dist = np.linalg.norm(id_vec[known_pair[0]] - id_vec[known_pair[1]])
        krama_ids = [i for i, (d, _) in id_domain.items() if d == "Krama"]
        vyapar_ids = [i for i, (d, _) in id_domain.items() if d == "Vyapar"]
        random_dists = []
        for _ in range(20):
            a, b = random.choice(krama_ids), random.choice(vyapar_ids)
            random_dists.append(np.linalg.norm(id_vec[a] - id_vec[b]))
        console.print(f"Known edge {known_pair[0]} <-> {known_pair[1]}: distance = {edge_dist:.4f}")
        console.print(f"Random Krama<->Vyapar pair distance: mean = {np.mean(random_dists):.4f}, "
                       f"min = {np.min(random_dists):.4f}")
        console.print("[dim]With deterministic-fallback embeddings this mostly tests plumbing, not "
                      "semantics — rerun with real embeddings (drop --fallback-embeddings) for a "
                      "meaningful signal here.[/dim]")
    else:
        console.print("[yellow]Known edge pair not found in this seed snapshot — skipping.[/yellow]")

    # 6. Resolution dial sweep on one query
    console.print("\n[bold yellow]6. Resolution Dial sweep (R: 0.0 -> 1.0) on one query...[/bold yellow]")
    probe_id = sample_ids[0]
    sweep = tree.sweep_resolution_dial(id_vec[probe_id], resolutions=[0.0, 0.25, 0.5, 0.75, 1.0])
    sweep_table = Table(title=f"Resolution sweep for {probe_id}")
    sweep_table.add_column("R")
    sweep_table.add_column("Final energy", justify="right")
    sweep_table.add_column("# candidates", justify="right")
    for s in sweep:
        sweep_table.add_row(f"{s['resolution']:.2f}", f"{s['final_energy']:.4f}", str(len(s["candidate_ids"])))
    console.print(sweep_table)

    console.print("\n[bold green]Done.[/bold green] This is a correctness/plumbing pass on real data, "
                  "not the recall/latency benchmark — see adr-smriti-integration.md for what's still needed "
                  "before that comparison means anything.")


if __name__ == "__main__":
    main()
