"""
Tests for Phase 3 — Skill co-occurrence network and Louvain communities.

Run from project root:
    python3 -m pytest tests/test_skill_network.py -v
"""

import sys
import math
from pathlib import Path
from collections import Counter

import pytest
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.skillmine.skill_network import (
    build_cooccurrence,
    compute_pmi,
    normalise_ppmi,
    build_graph,
    detect_communities,
    summarise_communities,
    _suggest_label,
)
from src.skillmine.config import (
    TRANSACTIONS_CSV, CLEAN_CSV, NODES_CSV, EDGES_CSV, COMMUNITIES_CSV,
    NET_MIN_CO_COUNT, RANDOM_SEED,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

# 20 baskets that guarantee controlled co-occurrences
BASKETS = {
    f"j{i:02d}": frozenset(skills)
    for i, skills in enumerate([
        ["python", "sql", "aws"],
        ["python", "sql", "azure"],
        ["python", "sql"],
        ["python", "sql", "ml"],
        ["python", "aws", "azure"],
        ["sql", "aws", "azure"],
        ["java", "spring"],
        ["java", "spring", "sql"],
        ["java", "spring", "aws"],
        ["java", "spring", "azure"],
        ["python", "ml"],
        ["python", "sql", "ml"],
        ["ml", "aws"],
        ["python", "sql", "aws"],
        ["java", "sql"],
        ["python", "azure"],
        ["java", "spring", "sql"],
        ["python", "sql", "aws"],
        ["java", "spring"],
        ["ml", "python"],
    ])
}

N = len(BASKETS)  # 20
# python: 14, sql: 12, java: 7, spring: 7, aws: 7, azure: 5, ml: 5
SKILL_FREQ = Counter()
for basket in BASKETS.values():
    SKILL_FREQ.update(basket)


# ── Edge construction tests ───────────────────────────────────────────────

class TestBuildCooccurrence:
    def test_returns_counter(self):
        pc = build_cooccurrence(BASKETS)
        assert isinstance(pc, Counter)

    def test_pairs_are_sorted(self):
        """All pairs (a, b) should have a < b lexicographically."""
        pc = build_cooccurrence(BASKETS)
        for a, b in pc.keys():
            assert a < b, f"Unsorted pair: ({a}, {b})"

    def test_python_sql_count(self):
        """python+sql appear together in baskets 0,1,2,3,7(?),11,13,17 = 8 baskets."""
        pc = build_cooccurrence(BASKETS)
        # Count manually from BASKETS
        expected = sum(
            1 for basket in BASKETS.values()
            if "python" in basket and "sql" in basket
        )
        assert pc[("python", "sql")] == expected

    def test_no_self_pairs(self):
        pc = build_cooccurrence(BASKETS)
        for a, b in pc.keys():
            assert a != b

    def test_empty_baskets_returns_empty(self):
        pc = build_cooccurrence({})
        assert len(pc) == 0

    def test_single_skill_baskets_returns_empty(self):
        single = {"j0": frozenset(["python"]), "j1": frozenset(["sql"])}
        pc = build_cooccurrence(single)
        assert len(pc) == 0


class TestComputePmi:
    def test_returns_list_of_dicts(self):
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        assert isinstance(edges, list)
        if edges:
            assert isinstance(edges[0], dict)

    def test_required_keys(self):
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        for e in edges:
            assert {"skill_a", "skill_b", "co_count", "pmi", "ppmi"} <= e.keys()

    def test_no_negative_ppmi(self):
        """ppmi = max(pmi, 0) so all ppmi values must be >= 0."""
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        for e in edges:
            assert e["ppmi"] >= 0

    def test_pairs_below_min_co_count_excluded(self):
        pc = build_cooccurrence(BASKETS)
        edges_low  = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        edges_high = compute_pmi(pc, SKILL_FREQ, N, min_co_count=5)
        assert len(edges_low) >= len(edges_high)

    def test_pmi_formula_correct(self):
        """Verify PMI(python, sql) against hand calculation."""
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        row = next((e for e in edges
                    if set([e["skill_a"], e["skill_b"]]) == {"python", "sql"}), None)
        if row is None:
            pytest.skip("python-sql pair not in edges")
        co = sum(1 for b in BASKETS.values() if "python" in b and "sql" in b)
        p_ab = co / N
        p_a  = SKILL_FREQ["python"] / N
        p_b  = SKILL_FREQ["sql"] / N
        expected_pmi = math.log2(p_ab / (p_a * p_b))
        assert abs(row["pmi"] - expected_pmi) < 0.01

    def test_negative_pmi_edges_excluded(self):
        """No returned edge should have pmi <= NET_MIN_PMI (0)."""
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        for e in edges:
            assert e["pmi"] > 0, (
                f"Edge {e['skill_a']}-{e['skill_b']} has PMI={e['pmi']} ≤ 0")

    def test_empty_pair_counts_returns_empty(self):
        edges = compute_pmi(Counter(), {}, 1, min_co_count=1)
        assert edges == []


class TestNormalisePpmi:
    def test_max_ppmi_norm_is_one(self):
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        edges = normalise_ppmi(edges)
        if edges:
            max_norm = max(e["ppmi_norm"] for e in edges)
            assert abs(max_norm - 1.0) < 1e-6

    def test_all_ppmi_norm_in_unit_interval(self):
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        edges = normalise_ppmi(edges)
        for e in edges:
            assert 0.0 <= e["ppmi_norm"] <= 1.0 + 1e-9

    def test_empty_list_returns_empty(self):
        assert normalise_ppmi([]) == []


class TestBuildGraph:
    def _make_edges(self):
        pc = build_cooccurrence(BASKETS)
        edges = compute_pmi(pc, SKILL_FREQ, N, min_co_count=1)
        return normalise_ppmi(edges)

    def test_returns_networkx_graph(self):
        G = build_graph(SKILL_FREQ, self._make_edges())
        assert isinstance(G, nx.Graph)

    def test_all_skills_are_nodes(self):
        G = build_graph(SKILL_FREQ, self._make_edges())
        for skill in SKILL_FREQ:
            assert skill in G.nodes

    def test_frequency_node_attribute_set(self):
        G = build_graph(SKILL_FREQ, self._make_edges())
        for n in G.nodes():
            assert "frequency" in G.nodes[n]
            assert G.nodes[n]["frequency"] == SKILL_FREQ[n]

    def test_edge_weight_is_ppmi_norm(self):
        edges = self._make_edges()
        G = build_graph(SKILL_FREQ, edges)
        for e in edges:
            a, b = e["skill_a"], e["skill_b"]
            if G.has_edge(a, b):
                assert abs(G[a][b]["weight"] - e["ppmi_norm"]) < 1e-6

    def test_no_self_loops(self):
        G = build_graph(SKILL_FREQ, self._make_edges())
        assert nx.number_of_selfloops(G) == 0

    def test_undirected(self):
        G = build_graph(SKILL_FREQ, self._make_edges())
        assert not G.is_directed()

    def test_empty_edges_builds_node_only_graph(self):
        G = build_graph({"python": 10, "sql": 8}, [])
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 0


class TestDetectCommunities:
    def _make_graph(self):
        pc = build_cooccurrence(BASKETS)
        edges = normalise_ppmi(compute_pmi(pc, SKILL_FREQ, N, min_co_count=1))
        return build_graph(SKILL_FREQ, edges)

    def test_returns_dict(self):
        G = self._make_graph()
        part = detect_communities(G, seed=RANDOM_SEED)
        assert isinstance(part, dict)

    def test_all_nodes_assigned(self):
        G = self._make_graph()
        part = detect_communities(G, seed=RANDOM_SEED)
        assert set(part.keys()) == set(G.nodes())

    def test_community_ids_are_integers(self):
        G = self._make_graph()
        part = detect_communities(G, seed=RANDOM_SEED)
        for cid in part.values():
            assert isinstance(cid, int)

    def test_at_least_two_communities(self):
        """With distinct skill clusters, expect >1 community."""
        G = self._make_graph()
        part = detect_communities(G, seed=RANDOM_SEED)
        assert len(set(part.values())) >= 2

    def test_reproducible_with_same_seed(self):
        G = self._make_graph()
        part1 = detect_communities(G, seed=42)
        part2 = detect_communities(G, seed=42)
        assert part1 == part2


class TestSuggestLabel:
    def test_ml_family(self):
        skills = ["python", "tensorflow", "pytorch", "deep learning", "nlp"]
        label = _suggest_label(skills, {s: 100 for s in skills})
        assert "ML" in label or "AI" in label

    def test_cloud_family(self):
        skills = ["aws", "docker", "kubernetes", "terraform", "ci/cd"]
        label = _suggest_label(skills, {s: 100 for s in skills})
        assert "Cloud" in label or "DevOps" in label

    def test_fallback_for_single_match(self):
        skills = ["python"]  # Only 1 match → mixed fallback
        label = _suggest_label(skills, {"python": 100})
        assert "mixed" in label

    def test_returns_string(self):
        label = _suggest_label(["python", "sql"], {"python": 100, "sql": 50})
        assert isinstance(label, str)


class TestOutputFiles:
    """Integration tests: verify Phase 3 outputs exist and are valid."""

    @pytest.mark.skipif(not NODES_CSV.exists(), reason="Phase 3 not yet run")
    def test_nodes_csv_has_required_columns(self):
        df = pd.read_csv(NODES_CSV)
        assert {"skill", "frequency", "community_id", "degree"} <= set(df.columns)

    @pytest.mark.skipif(not NODES_CSV.exists(), reason="Phase 3 not yet run")
    def test_nodes_csv_row_count(self):
        """Node count should equal the skill dictionary size.
        Updated from 61 → 65 after Phase 5 dictionary expansion (go, erp, swift, api)."""
        df = pd.read_csv(NODES_CSV)
        # Verify all nodes are present — exact count matches current dictionary size
        from src.skillmine.config import CLEAN_CSV
        import pandas as _pd
        trans = _pd.read_csv(
            str(CLEAN_CSV).replace("skillmine_clean.csv", "skillmine_transactions.csv"),
            dtype=str
        )
        expected = trans["skill"].nunique()
        assert len(df) == expected, (
            f"Expected {expected} skill nodes (one per canonical skill), got {len(df)}"
        )

    @pytest.mark.skipif(not NODES_CSV.exists(), reason="Phase 3 not yet run")
    def test_no_null_community_ids(self):
        df = pd.read_csv(NODES_CSV)
        assert df["community_id"].notna().all()

    @pytest.mark.skipif(not EDGES_CSV.exists(), reason="Phase 3 not yet run")
    def test_edges_csv_has_required_columns(self):
        df = pd.read_csv(EDGES_CSV)
        assert {"skill_a", "skill_b", "co_count", "pmi", "ppmi", "ppmi_norm"} <= set(df.columns)

    @pytest.mark.skipif(not EDGES_CSV.exists(), reason="Phase 3 not yet run")
    def test_all_ppmi_norm_valid(self):
        df = pd.read_csv(EDGES_CSV)
        assert (df["ppmi_norm"] >= 0).all()
        assert (df["ppmi_norm"] <= 1.0001).all()

    @pytest.mark.skipif(not EDGES_CSV.exists(), reason="Phase 3 not yet run")
    def test_no_negative_pmi_edges(self):
        df = pd.read_csv(EDGES_CSV)
        assert (df["pmi"] > 0).all()

    @pytest.mark.skipif(not COMMUNITIES_CSV.exists(), reason="Phase 3 not yet run")
    def test_communities_csv_has_required_columns(self):
        df = pd.read_csv(COMMUNITIES_CSV)
        assert {"community_id", "size", "member_skills",
                "top_frequent_skills", "suggested_label"} <= set(df.columns)

    @pytest.mark.skipif(not COMMUNITIES_CSV.exists(), reason="Phase 3 not yet run")
    def test_community_sizes_positive(self):
        df = pd.read_csv(COMMUNITIES_CSV)
        assert (df["size"] >= 2).all()

    @pytest.mark.skipif(
        not (Path("output/network/skill_network.html").exists()),
        reason="Phase 3 not yet run"
    )
    def test_html_file_exists_and_nonempty(self):
        html = Path("output/network/skill_network.html").read_text()
        assert len(html) > 1000
        assert "visjs" in html.lower() or "network" in html.lower()
