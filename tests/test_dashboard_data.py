"""
Tests for Phase 6 — dashboard data layer.

Run from project root:
    python3 -m pytest tests/test_dashboard_data.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.skillmine import dashboard_data as dd
from src.skillmine.config import (
    CLEAN_CSV, TRANSACTIONS_CSV, TABLES_DIR, EDGES_CSV, CLUSTER_ASSIGNMENTS_CSV,
    NETWORK_PRES_HTML,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def toy_txn():
    # 4 skill-bearing postings; posting j5 has no skills
    return pd.DataFrame({
        "job_id": ["j1", "j1", "j2", "j2", "j3", "j4", "j4"],
        "skill":  ["python", "sql", "python", "aws", "sql", "python", "sql"],
    })


@pytest.fixture
def toy_clean():
    return pd.DataFrame({
        "job_id": ["j1", "j2", "j3", "j4", "j5"],
        "title": ["Data Analyst", "Cloud Engineer", "Data Analyst", "Data Engineer", "Engineer"],
        "company_name": list("ABCDE"),
        "location": ["NY"] * 5,
        "listed_date": ["2024-04-05", "2024-04-06", "2024-04-07", "2024-04-08", "2024-04-20"],
        "listed_month": ["2024-04-01"] * 5,
    })


@pytest.fixture
def toy_rules():
    df = pd.DataFrame({
        "antecedents": ["python", "sql", "python, sql", "aws"],
        "consequents": ["sql", "python", "aws", "python"],
        "support":     [0.50, 0.50, 0.10, 0.25],
        "support_count": [2, 2, 1, 1],
        "confidence":  [0.67, 0.67, 0.20, 1.00],
        "lift":        [1.3, 1.3, 0.8, 1.5],
    })
    df["n_antecedents"] = df["antecedents"].map(lambda s: len(dd.parse_itemset(s)))
    df["n_consequents"] = df["consequents"].map(lambda s: len(dd.parse_itemset(s)))
    df["rule"] = df["antecedents"] + "  →  " + df["consequents"]
    return df


# ── Itemsets and rule filtering ───────────────────────────────────────────

class TestParseItemset:
    def test_single(self):
        assert dd.parse_itemset("python") == frozenset({"python"})

    def test_multi_with_spaces(self):
        assert dd.parse_itemset("docker,  gcp") == frozenset({"docker", "gcp"})

    def test_keeps_technical_names(self):
        assert dd.parse_itemset("c++, ci/cd, node.js") == frozenset({"c++", "ci/cd", "node.js"})

    @pytest.mark.parametrize("blank", ["", "   ", None, float("nan")])
    def test_blank_is_empty(self, blank):
        assert dd.parse_itemset(blank) == frozenset()


class TestFilterRules:
    def test_thresholds(self, toy_rules):
        out = dd.filter_rules(toy_rules, min_support=0.2, min_confidence=0.5, min_lift=1.2)
        assert set(out["rule"]) == {"python  →  sql", "sql  →  python", "aws  →  python"}

    def test_skill_either_side(self, toy_rules):
        out = dd.filter_rules(toy_rules, skills=["aws"])
        assert len(out) == 2

    def test_skill_antecedent_only(self, toy_rules):
        out = dd.filter_rules(toy_rules, skills=["aws"], side="antecedent")
        assert out["rule"].tolist() == ["aws  →  python"]

    def test_skill_consequent_only(self, toy_rules):
        out = dd.filter_rules(toy_rules, skills=["python"], side="consequent")
        assert set(out["rule"]) == {"sql  →  python", "aws  →  python"}

    def test_multiple_skills_must_all_match(self, toy_rules):
        out = dd.filter_rules(toy_rules, skills=["python", "aws"])
        assert set(out["rule"]) == {"python, sql  →  aws", "aws  →  python"}

    def test_skill_not_substring_matched(self, toy_rules):
        # "sq" must not match "sql"
        assert dd.filter_rules(toy_rules, skills=["sq"]).empty

    def test_max_items(self, toy_rules):
        out = dd.filter_rules(toy_rules, max_items=2)
        assert "python, sql  →  aws" not in set(out["rule"])

    def test_empty_result_keeps_columns(self, toy_rules):
        out = dd.filter_rules(toy_rules, min_lift=100)
        assert out.empty and list(out.columns) == list(toy_rules.columns)

    @pytest.mark.parametrize("side", ["either", "antecedent", "consequent"])
    def test_empty_after_thresholds_with_skill_filter(self, toy_rules, side):
        # Regression: an empty frame + skill filter used to drop every column.
        out = dd.filter_rules(toy_rules, min_lift=100, skills=["python"], side=side)
        assert out.empty and list(out.columns) == list(toy_rules.columns)

    def test_bad_side_raises(self, toy_rules):
        with pytest.raises(ValueError):
            dd.filter_rules(toy_rules, skills=["sql"], side="left")


class TestSkillRules:
    def test_split_by_side_and_sorted(self, toy_rules):
        out = dd.skill_rules(toy_rules, "python")
        assert out["as_antecedent"]["rule"].tolist() == ["python  →  sql", "python, sql  →  aws"]
        assert out["as_consequent"]["rule"].tolist() == ["aws  →  python", "sql  →  python"]

    def test_unknown_skill(self, toy_rules):
        out = dd.skill_rules(toy_rules, "cobol")
        assert out["as_antecedent"].empty and out["as_consequent"].empty


# ── Frequencies, neighbours, clusters ─────────────────────────────────────

class TestSkillFrequencies:
    def test_counts_and_denominators(self, toy_txn):
        f = dd.skill_frequencies(toy_txn, n_postings=5).set_index("skill")
        assert f.loc["python", "postings"] == 3
        assert f.loc["python", "share_all"] == pytest.approx(3 / 5)
        assert f.loc["python", "share_skilled"] == pytest.approx(3 / 4)

    def test_duplicate_pairs_counted_once(self, toy_txn):
        dup = pd.concat([toy_txn, toy_txn.iloc[[0]]])
        f = dd.skill_frequencies(dup, 5).set_index("skill")
        assert f.loc["python", "postings"] == 3

    def test_sorted_desc(self, toy_txn):
        f = dd.skill_frequencies(toy_txn, 5)
        assert f["postings"].is_monotonic_decreasing


class TestSkillNeighbours:
    def test_both_edge_directions(self):
        edges = pd.DataFrame({"skill_a": ["aws", "python"], "skill_b": ["python", "sql"],
                              "co_count": [30, 50], "pmi": [1.0, 0.5], "ppmi_norm": [1.0, 0.5]})
        nb = dd.skill_neighbours(edges, "python")
        assert nb["partner"].tolist() == ["aws", "sql"]

    def test_isolated_skill(self):
        edges = pd.DataFrame({"skill_a": ["aws"], "skill_b": ["python"], "co_count": [30],
                              "pmi": [1.0], "ppmi_norm": [1.0]})
        assert dd.skill_neighbours(edges, "excel").empty


class TestClusterShares:
    def test_share_of_cluster(self, toy_txn):
        assign = pd.DataFrame({"job_id": ["j1", "j2", "j3", "j4"], "cluster_id": [0, 1, 0, 0]})
        s = dd.cluster_skill_shares(toy_txn, assign).set_index(["cluster_id", "skill"])
        assert s.loc[(0, "sql"), "postings"] == 3
        assert s.loc[(0, "sql"), "share"] == pytest.approx(1.0)
        assert s.loc[(0, "python"), "share"] == pytest.approx(2 / 3)
        assert s.loc[(1, "aws"), "share"] == pytest.approx(1.0)

    def test_distribution_sums_to_one(self, toy_txn):
        assign = pd.DataFrame({"job_id": ["j1", "j2", "j3", "j4"], "cluster_id": [0, 1, 0, 0]})
        summ = pd.DataFrame({"cluster_id": [0, 1], "display_label": ["C0 · A", "C1 · B"]})
        d = dd.skill_cluster_distribution(dd.cluster_skill_shares(toy_txn, assign), summ, "python")
        assert d["share_of_skill"].sum() == pytest.approx(1.0)
        assert d.iloc[0]["display_label"] == "C0 · A"


# ── Overview and trends ───────────────────────────────────────────────────

class TestOverviewStats:
    def test_counts(self, toy_clean, toy_txn):
        ba = pd.DataFrame({"stage": ["Before cleaning", "After cleaning"], "rows": [10, 5],
                           "cols": [31, 27], "duplicate_text": [3, 0]})
        s = dd.overview_stats(toy_clean, toy_txn, ba)
        assert s["postings"] == 5
        assert s["skilled_postings"] == 4
        assert s["zero_skill_postings"] == 1
        assert s["n_skills"] == 3
        assert s["n_pairs"] == 7
        assert s["n_months"] == 1
        assert s["raw_rows"] == 10


class TestTrendStatus:
    def test_single_month_unavailable(self, toy_clean):
        st = dd.trend_status(toy_clean, min_months=6)
        assert st["available"] is False
        assert st["n_months"] == 1
        assert "2024-04-05" in st["reason"] and "2024-04-20" in st["reason"]

    def test_enough_months_available(self):
        months = [f"2024-{m:02d}-01" for m in range(1, 8)]
        clean = pd.DataFrame({"listed_month": months, "listed_date": months})
        assert dd.trend_status(clean, min_months=6)["available"] is True

    def test_threshold_is_inclusive(self):
        months = [f"2024-{m:02d}-01" for m in range(1, 7)]
        clean = pd.DataFrame({"listed_month": months, "listed_date": months})
        assert dd.trend_status(clean, min_months=6)["available"] is True
        assert dd.trend_status(clean, min_months=7)["available"] is False


# ── Missing artifacts ─────────────────────────────────────────────────────

class TestMissingArtifact:
    def test_carries_command(self, tmp_path):
        with pytest.raises(dd.MissingArtifact) as exc:
            dd.load_rules("nope", tables_dir=tmp_path)
        assert exc.value.command == dd.RULES_CMD
        assert exc.value.path.name == "rules_nope.csv"

    def test_is_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dd.load_clean(tmp_path / "missing.csv")

    def test_signature_handles_missing(self, tmp_path):
        assert dd.file_signature(tmp_path / "missing.csv") == (None,)


# ── Real artifacts (skipped when outputs have not been generated) ─────────

needs_outputs = pytest.mark.skipif(
    not (CLEAN_CSV.exists() and TRANSACTIONS_CSV.exists()),
    reason="pipeline outputs not generated")


@needs_outputs
class TestRealArtifacts:
    def test_transactions_are_subset_of_clean(self):
        clean, txn = dd.load_clean(), dd.load_transactions()
        assert set(txn["job_id"]) <= set(clean["job_id"])

    def test_no_duplicate_pairs(self):
        assert not dd.load_transactions().duplicated(["job_id", "skill"]).any()

    def test_zero_skill_matches_before_after(self):
        s = dd.overview_stats(dd.load_clean(), dd.load_transactions(), dd.load_before_after())
        after = dd.load_before_after().set_index("stage").loc["After cleaning"]
        assert s["zero_skill_postings"] == after["missing_skills"]
        assert s["postings"] == after["rows"]

    def test_current_data_has_no_trend(self):
        # Guard: the dashboard must not claim trends from the April-2024 snapshot.
        assert dd.trend_status(dd.load_clean())["available"] is False

    def test_rules_match_summary(self):
        summ = dd.load_rules_summary()
        for row in summ.itertuples():
            r = dd.load_rules(row.run, TABLES_DIR)
            assert len(r) == row.n_rules
            assert (r["n_baskets"] == row.n_baskets).all()

    @pytest.mark.skipif(not CLUSTER_ASSIGNMENTS_CSV.exists(), reason="clusters not generated")
    def test_clustered_postings_are_skill_bearing(self):
        a, txn = dd.load_cluster_assignments(), dd.load_transactions()
        assert set(a["job_id"]) == set(txn["job_id"])

    @pytest.mark.skipif(not EDGES_CSV.exists(), reason="network not generated")
    def test_edges_reference_known_skills(self):
        e, txn = dd.load_edges(), dd.load_transactions()
        assert set(e["skill_a"]) | set(e["skill_b"]) <= set(txn["skill"])

    @pytest.mark.skipif(not NETWORK_PRES_HTML.exists(), reason="network not generated")
    def test_network_html_self_contained(self):
        html = dd.load_network_html(presentation=True)
        assert 'src="lib/' not in html
        assert "node_modules" not in html
        assert "function neighbourhoodHighlight" in html
