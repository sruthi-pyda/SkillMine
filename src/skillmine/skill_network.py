"""
Phase 3 — Skill co-occurrence network and Louvain community detection.

Usage (from project root):
    python3 -m src.skillmine.skill_network

Outputs written to output/network/:
    skill_nodes.csv        — one row per skill node (frequency, community, degree)
    skill_edges.csv        — one row per edge (co_count, pmi, ppmi_norm, weight)
    skill_communities.csv  — one row per community (size, members, label, top titles)
    skill_network.html     — interactive PyVis graph coloured by community

PMI weighting
─────────────
For skills a and b, across N posting baskets:

    PMI(a, b) = log2( P(a,b) / (P(a) · P(b)) )

where:
    P(a,b) = co_count(a,b) / N    (fraction of baskets containing both)
    P(a)   = freq(a) / N          (fraction of baskets containing a)
    P(b)   = freq(b) / N          (fraction of baskets containing b)

Treatment of zero / negative PMI:
    PMI ≤ 0 means the pair co-occurs no more often than chance.
    Such edges are excluded from the graph entirely; only positive-PMI pairs
    become edges.  This prevents spurious connections between very common
    skills (e.g. communication + python) that happen to appear in the same
    postings just because both are frequent, not because they are semantically
    related.

Rare-skill safeguard:
    Edges are only built for pairs whose raw co_count ≥ NET_MIN_CO_COUNT (20).
    Very low co-counts produce statistically unreliable PMI estimates even when
    mathematically positive.

PPMI normalisation for edge weight:
    ppmi      = max(PMI, 0)
    ppmi_norm = ppmi / max_ppmi_in_graph   (maps to [0, 1])
    The edge weight stored in NetworkX and used for viz is ppmi_norm.
"""

import sys
import math
import logging
import random
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import networkx as nx
import community as community_louvain   # python-louvain

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.skillmine.config import (
    CLEAN_CSV, TRANSACTIONS_CSV,
    NETWORK_DIR, NODES_CSV, EDGES_CSV, COMMUNITIES_CSV, NETWORK_HTML,
    NET_MIN_CO_COUNT, NET_MIN_PMI, NET_VIZ_MIN_FREQ,
    NET_VIZ_MIN_EDGE_WEIGHT, NET_VIZ_PRES_EDGE_WEIGHT,
    NET_LOUVAIN_RESOLUTION, NET_MIN_COMMUNITY_SIZE, NET_TITLES_PER_COMMUNITY,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Community colour palette (one per community; cycles if > 12 communities)
COMMUNITY_COLOURS = [
    "#E8512B", "#2E6E8E", "#27AE60", "#8E44AD", "#F39C12",
    "#16A085", "#C0392B", "#2980B9", "#D35400", "#1ABC9C",
    "#7F8C8D", "#2C3E50",
]


# ── Data loading ──────────────────────────────────────────────────────────

def load_baskets() -> tuple[dict[str, frozenset], dict[str, int], int]:
    """
    Returns:
        job_baskets  — {job_id: frozenset of canonical skills}
        skill_freq   — {skill: count of postings containing it}
        N            — total number of baskets (postings with ≥1 skill)
    """
    txn = pd.read_csv(TRANSACTIONS_CSV, dtype=str)

    job_baskets: dict[str, frozenset] = {}
    for job_id, group in txn.groupby("job_id"):
        job_baskets[job_id] = frozenset(group["skill"])

    skill_freq: dict[str, int] = txn.groupby("skill").size().to_dict()
    N = len(job_baskets)

    log.info("Loaded %d baskets, %d unique skills", N, len(skill_freq))
    return job_baskets, skill_freq, N


def load_titles() -> dict[str, str]:
    """Return {job_id: title_lower} from the clean postings CSV."""
    clean = pd.read_csv(CLEAN_CSV, dtype=str, usecols=["job_id", "title"])
    return {row.job_id: str(row.title).lower() for row in clean.itertuples()}


# ── Edge construction ─────────────────────────────────────────────────────

def build_cooccurrence(
    job_baskets: dict[str, frozenset],
) -> Counter:
    """Count how many baskets contain each ordered pair (a < b)."""
    pair_counts: Counter = Counter()
    for skills in job_baskets.values():
        for a, b in combinations(sorted(skills), 2):
            pair_counts[(a, b)] += 1
    return pair_counts


def compute_pmi(
    pair_counts: Counter,
    skill_freq: dict[str, int],
    N: int,
    min_co_count: int = NET_MIN_CO_COUNT,
) -> list[dict]:
    """
    Compute PMI for every pair that meets min_co_count.
    Returns a list of edge dicts; negative-PMI pairs are excluded.

    Each dict contains:
        skill_a, skill_b, co_count, p_ab, p_a, p_b, pmi, ppmi
    """
    edges = []
    for (a, b), co_count in pair_counts.items():
        if co_count < min_co_count:
            continue
        p_ab = co_count / N
        p_a  = skill_freq[a] / N
        p_b  = skill_freq[b] / N
        pmi  = math.log2(p_ab / (p_a * p_b))
        ppmi = max(pmi, 0.0)
        if ppmi <= NET_MIN_PMI:
            continue
        edges.append({
            "skill_a":  a,
            "skill_b":  b,
            "co_count": co_count,
            "p_ab":     round(p_ab, 6),
            "p_a":      round(p_a, 6),
            "p_b":      round(p_b, 6),
            "pmi":      round(pmi, 4),
            "ppmi":     round(ppmi, 4),
        })
    return edges


def normalise_ppmi(edges: list[dict]) -> list[dict]:
    """Add ppmi_norm = ppmi / max_ppmi, mapping edge weights to [0, 1]."""
    if not edges:
        return edges
    max_ppmi = max(e["ppmi"] for e in edges)
    for e in edges:
        e["ppmi_norm"] = round(e["ppmi"] / max_ppmi, 4) if max_ppmi > 0 else 0.0
    return edges


# ── NetworkX graph ────────────────────────────────────────────────────────

def build_graph(
    skill_freq: dict[str, int],
    edges: list[dict],
) -> nx.Graph:
    """
    Build an undirected weighted NetworkX graph.
    Node attributes: frequency
    Edge attributes: co_count, pmi, ppmi, ppmi_norm (= weight)
    """
    G = nx.Graph()

    for skill, freq in skill_freq.items():
        G.add_node(skill, frequency=freq)

    for e in edges:
        G.add_edge(
            e["skill_a"], e["skill_b"],
            co_count=e["co_count"],
            pmi=e["pmi"],
            ppmi=e["ppmi"],
            weight=e["ppmi_norm"],
        )

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ── Louvain community detection ───────────────────────────────────────────

def detect_communities(G: nx.Graph, seed: int = RANDOM_SEED) -> dict[str, int]:
    """
    Run Louvain community detection on G.
    Returns {skill: community_id}.
    Notes:
      - Uses 'weight' edge attribute (ppmi_norm) so tighter skill clusters
        get higher weight.
      - Resolution=1.0 is the standard default; documented in config.
      - python-louvain is stochastic; the seed fixes the random state for
        reproducibility within the same library version.
    """
    random.seed(seed)
    partition = community_louvain.best_partition(
        G,
        weight="weight",
        resolution=NET_LOUVAIN_RESOLUTION,
        random_state=seed,
    )
    n_communities = len(set(partition.values()))
    log.info("Louvain: %d communities (resolution=%.1f, seed=%d)",
             n_communities, NET_LOUVAIN_RESOLUTION, seed)
    return partition


# ── Community summaries ───────────────────────────────────────────────────

def summarise_communities(
    partition: dict[str, int],
    skill_freq: dict[str, int],
    job_baskets: dict[str, frozenset],
    titles: dict[str, str],
) -> list[dict]:
    """
    For each community produce:
      - community_id, size
      - member_skills (sorted by frequency desc)
      - top_frequent_skills (top 5 by frequency)
      - top_titles (top job titles from postings dominated by community skills)
      - suggested_label (cautious, evidence-based)
    """
    # Group skills by community
    community_skills: dict[int, list[str]] = defaultdict(list)
    for skill, cid in partition.items():
        community_skills[cid].append(skill)

    # For each posting, find which community most of its skills belong to
    posting_community: dict[str, int] = {}
    for job_id, basket in job_baskets.items():
        if not basket:
            continue
        counts = Counter(partition[s] for s in basket if s in partition)
        if counts:
            posting_community[job_id] = counts.most_common(1)[0][0]

    # Gather job titles per community
    community_titles: dict[int, Counter] = defaultdict(Counter)
    for job_id, cid in posting_community.items():
        title = titles.get(job_id, "")
        if title:
            community_titles[cid][title] += 1

    summaries = []
    for cid, skills in sorted(community_skills.items()):
        if len(skills) < NET_MIN_COMMUNITY_SIZE:
            continue

        skills_by_freq = sorted(skills, key=lambda s: skill_freq.get(s, 0), reverse=True)
        top5 = skills_by_freq[:5]
        top_titles = [t for t, _ in community_titles[cid].most_common(NET_TITLES_PER_COMMUNITY)]

        label = _suggest_label(skills_by_freq, skill_freq)

        summaries.append({
            "community_id":      cid,
            "size":              len(skills),
            "member_skills":     "; ".join(skills_by_freq),
            "top_frequent_skills": "; ".join(top5),
            "top_job_titles":    "; ".join(top_titles[:5]),
            "suggested_label":   label,
            "label_note":        "data-derived; interpret with caution",
        })

    return summaries


# Label dictionary: canonical skills → candidate family labels.
# Derived from the actual community memberships observed at resolution=1.5.
# Labels are cautious: they reflect skill families found in this dataset, not
# definitive career-path categories.
_LABEL_HINTS = {
    frozenset(["python", "machine learning", "scikit-learn",
               "tensorflow", "pytorch", "deep learning", "nlp",
               "generative ai", "r", "c++", "matlab"]): "ML / AI",
    frozenset(["kubernetes", "gcp", "docker", "linux",
               "bash", "terraform"]): "Cloud Infrastructure / DevOps",
    frozenset(["aws", "azure", "django", "flask"]): "Cloud + Web Deployment",
    frozenset(["spark", "hadoop", "kafka", "databricks",
               "airflow", "scala", "snowflake", "etl",
               "data warehousing", "dbt", "data modeling"]): "Data Engineering",
    frozenset(["sql", "excel", "tableau", "power bi",
               "data visualization", "statistics",
               "sas", "looker", "sap", "salesforce",
               "ms office", "powerpoint"]): "Data Analysis / BI",
    frozenset(["java", "javascript", "c#", "react", "node.js",
               "typescript", "angular", "vue", "mongodb",
               "redis", "selenium", "git", "ci/cd",
               "agile", "jira", "rest api"]): "Software Engineering",
}


def _suggest_label(skills_by_freq: list[str], skill_freq: dict[str, int]) -> str:
    """
    Return a cautious label based on overlap with known skill families.
    Falls back to listing the top 3 skills.
    """
    skill_set = set(skills_by_freq)
    best_label, best_overlap = "mixed", 0
    for hint_set, label in _LABEL_HINTS.items():
        overlap = len(skill_set & hint_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    # Require at least 2 strong overlapping skills to assign a non-generic label
    if best_overlap < 2:
        top3 = ", ".join(skills_by_freq[:3])
        return f"mixed ({top3})"
    return best_label


# ── CSV exporters ─────────────────────────────────────────────────────────

def export_nodes(G: nx.Graph, partition: dict[str, int]) -> pd.DataFrame:
    rows = []
    for node in G.nodes():
        rows.append({
            "skill":        node,
            "frequency":    G.nodes[node].get("frequency", 0),
            "degree":       G.degree(node),
            "weighted_degree": round(sum(d["weight"] for _, _, d in G.edges(node, data=True)), 4),
            "community_id": partition.get(node, -1),
        })
    df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    df.to_csv(NODES_CSV, index=False)
    log.info("Written: %s (%d nodes)", NODES_CSV, len(df))
    return df


def export_edges(edges: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(edges)
    if df.empty:
        df.to_csv(EDGES_CSV, index=False)
        log.warning("Written empty edges CSV: %s", EDGES_CSV)
        return df
    df = df.sort_values("ppmi", ascending=False)
    df.to_csv(EDGES_CSV, index=False)
    log.info("Written: %s (%d edges)", EDGES_CSV, len(df))
    return df


def export_communities(summaries: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(summaries)
    df.to_csv(COMMUNITIES_CSV, index=False)
    log.info("Written: %s (%d communities)", COMMUNITIES_CSV, len(df))
    return df


# ── PyVis interactive graph ───────────────────────────────────────────────

def build_pyvis_html(
    G: nx.Graph,
    partition: dict[str, int],
    skill_freq: dict[str, int],
    min_freq: int = NET_VIZ_MIN_FREQ,
    min_edge_weight: float = NET_VIZ_MIN_EDGE_WEIGHT,
    out_path: Path | None = None,
) -> None:
    """
    Generate an interactive HTML graph coloured by Louvain community.

    Filtering applied for legibility:
      - Nodes with skill_freq < min_freq are excluded from the viz graph
        (but remain in the underlying analysis).
      - Edges with ppmi_norm < min_edge_weight are excluded.
        Default: NET_VIZ_MIN_EDGE_WEIGHT (0.05) — standard viz.
        Use NET_VIZ_PRES_EDGE_WEIGHT (0.20) for a presentation-quality version.

    Node size: proportional to log(frequency).
    Edge width: proportional to ppmi_norm × 10.
    Colour: by community_id.

    The full analytical edge/node data is always preserved in the CSV exports.
    """
    from pyvis.network import Network

    if out_path is None:
        out_path = NETWORK_HTML
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    # Assign colours
    community_ids = sorted(set(partition.values()))
    colour_map = {
        cid: COMMUNITY_COLOURS[i % len(COMMUNITY_COLOURS)]
        for i, cid in enumerate(community_ids)
    }

    # Build filtered subgraph for viz
    nodes_to_include = {
        n for n in G.nodes()
        if skill_freq.get(n, 0) >= min_freq
    }
    H = G.subgraph(nodes_to_include).copy()
    # Remove weak edges (threshold is configurable)
    weak_edges = [(u, v) for u, v, d in H.edges(data=True)
                  if d.get("weight", 0) < min_edge_weight]
    H.remove_edges_from(weak_edges)
    # Remove isolated nodes (after edge pruning)
    H.remove_nodes_from(list(nx.isolates(H)))

    log.info("PyVis graph: %d nodes, %d edges (after viz filter)",
             H.number_of_nodes(), H.number_of_edges())

    net = Network(
        height="750px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3,
                   spring_length=150, spring_strength=0.05)

    for node in H.nodes():
        freq  = skill_freq.get(node, 1)
        size  = max(10, math.log(freq + 1) * 5)
        cid   = partition.get(node, 0)
        colour = colour_map.get(cid, "#888888")
        net.add_node(
            node,
            label=node,
            title=(f"<b>{node}</b><br>"
                   f"Frequency: {freq}<br>"
                   f"Community: {cid}"),
            size=size,
            color=colour,
        )

    for u, v, data in H.edges(data=True):
        w = data.get("weight", 0.01)
        net.add_edge(
            u, v,
            value=w,
            title=(f"{u} ↔ {v}<br>"
                   f"Co-occurrences: {data.get('co_count', '?')}<br>"
                   f"PMI: {data.get('pmi', '?')}<br>"
                   f"PPMI (norm): {w:.3f}"),
            width=max(0.5, w * 6),
            color={"color": "#888888", "opacity": 0.5},
        )

    # Add legend as a title annotation
    net.set_options(json.dumps({
        "interaction": {"hover": True, "tooltipDelay": 100},
        "physics":     {"enabled": True},
        "nodes": {"font": {"size": 13}},
    }))

    net.write_html(str(out_path))
    log.info("Written: %s  (%d nodes, %d edges, min_edge_weight=%.2f)",
             out_path, H.number_of_nodes(), H.number_of_edges(), min_edge_weight)

    # Also write a filtered-node report
    excluded = [n for n in G.nodes() if skill_freq.get(n, 0) < min_freq]
    if excluded:
        log.info("Nodes excluded from viz (freq < %d): %s",
                 min_freq, ", ".join(sorted(excluded)))


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_phase3() -> dict:
    """Full Phase 3 pipeline. Returns results dict for testing."""
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    job_baskets, skill_freq, N = load_baskets()
    titles = load_titles()

    # 1. Build co-occurrence counts
    pair_counts = build_cooccurrence(job_baskets)
    log.info("Total co-occurrence pairs: %d", len(pair_counts))

    # 2. Compute PMI, apply filters, normalise
    edges = compute_pmi(pair_counts, skill_freq, N)
    log.info("Edges after min_co_count=%d and PMI>0 filters: %d",
             NET_MIN_CO_COUNT, len(edges))
    edges = normalise_ppmi(edges)

    # 3. Build NetworkX graph
    G = build_graph(skill_freq, edges)

    # 4. Louvain community detection
    partition = detect_communities(G, seed=RANDOM_SEED)
    community_sizes = Counter(partition.values())
    log.info("Community sizes: %s", dict(sorted(community_sizes.items())))

    # 5. Community summaries
    summaries = summarise_communities(partition, skill_freq, job_baskets, titles)

    # 6. Export tables
    nodes_df  = export_nodes(G, partition)
    edges_df  = export_edges(edges)
    comm_df   = export_communities(summaries)

    # 7. PyVis HTML — standard (ppmi_norm ≥ 0.05) and presentation (ppmi_norm ≥ 0.20)
    build_pyvis_html(G, partition, skill_freq)
    build_pyvis_html(G, partition, skill_freq,
                     min_edge_weight=NET_VIZ_PRES_EDGE_WEIGHT,
                     out_path=NETWORK_DIR / "skill_network_presentation.html")

    # 8. Graph quality metrics
    components = list(nx.connected_components(G))
    log.info("Graph connectivity: %d connected component(s)", len(components))
    if len(components) == 1:
        log.info("Graph is fully connected.")
    else:
        sizes = sorted([len(c) for c in components], reverse=True)
        log.info("Component sizes: %s", sizes)

    results = {
        "n_nodes":           G.number_of_nodes(),
        "n_edges":           G.number_of_edges(),
        "n_communities":     len(set(partition.values())),
        "community_sizes":   dict(community_sizes),
        "partition":         partition,
        "G":                 G,
        "edges":             edges,
        "summaries":         summaries,
        "nodes_df":          nodes_df,
        "edges_df":          edges_df,
        "comm_df":           comm_df,
    }
    return results


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_phase3()

    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Graph:       {results['n_nodes']} nodes, {results['n_edges']} edges")
    print(f"  Communities: {results['n_communities']}")
    print()

    print("Community summaries:")
    for s in results["summaries"]:
        print(f"\n  Community {s['community_id']} — {s['suggested_label']} "
              f"({s['size']} skills)")
        print(f"    Members:   {s['member_skills']}")
        print(f"    Top skills:{s['top_frequent_skills']}")
        print(f"    Top titles:{s['top_job_titles']}")
        print(f"    Note:      {s['label_note']}")

    print()
    print("Top 10 edges by PMI:")
    edf = results["edges_df"].head(10)
    for _, row in edf.iterrows():
        print(f"  {row.skill_a:22s} ↔ {row.skill_b:22s}  "
              f"co={row.co_count:4d}  PMI={row.pmi:.3f}  ppmi_norm={row.ppmi_norm:.3f}")
