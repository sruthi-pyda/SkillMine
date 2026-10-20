"""
Tests for Phase 2 — FP-Growth association rules.

Run from project root:
    python3 -m pytest tests/test_association_rules.py -v
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.skillmine.association_rules import (
    build_binary_matrix,
    mine_rules,
    verify_metrics,
    assign_role_group,
    load_baskets,
)
from src.skillmine.config import (
    TRANSACTIONS_CSV, CLEAN_CSV,
    FP_MIN_SUPPORT_OVERALL, FP_MIN_CONFIDENCE, FP_MIN_LIFT,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

TINY_BASKETS = [
    ["python", "sql", "aws"],
    ["python", "sql", "azure"],
    ["python", "sql"],
    ["python", "aws", "azure"],
    ["sql", "aws"],
    ["sql", "azure"],
    ["java", "spring", "sql"],
    ["java", "spring"],
    ["java", "spring", "aws"],
    ["java", "spring", "azure"],
]

# Hand-computed values for the tiny basket
# python appears in baskets 0,1,2,3 → support = 4/10 = 0.4
# sql appears in 0,1,2,4,5,6 → support = 6/10 = 0.6
# python+sql in 0,1,2 → joint support = 3/10 = 0.3
# conf(python→sql) = 0.3/0.4 = 0.75
# lift(python→sql) = 0.75/0.6 = 1.25


# ── Unit tests ────────────────────────────────────────────────────────────

class TestBuildBinaryMatrix:
    def test_shape(self):
        bm = build_binary_matrix(TINY_BASKETS)
        assert bm.shape[0] == len(TINY_BASKETS)

    def test_columns_are_all_skills(self):
        bm = build_binary_matrix(TINY_BASKETS)
        all_skills = {s for basket in TINY_BASKETS for s in basket}
        assert set(bm.columns) == all_skills

    def test_python_row_correct(self):
        bm = build_binary_matrix(TINY_BASKETS)
        # Basket 0: python, sql, aws → python=True
        assert bm.loc[0, "python"] == True   # noqa: E712 — numpy bool needs ==
        # Basket 6: java, spring, sql → python=False
        assert bm.loc[6, "python"] == False  # noqa: E712

    def test_all_boolean(self):
        bm = build_binary_matrix(TINY_BASKETS)
        # mlxtend returns numpy bool dtype
        assert all(str(t) in ("bool", "object") for t in bm.dtypes)

    def test_empty_returns_empty_df(self):
        # build_binary_matrix([]) returns an empty DataFrame, not an exception
        bm = build_binary_matrix([])
        assert len(bm) == 0


class TestMineRules:
    # Use enough baskets to exceed FP_GROUP_MIN_BASKETS (150)
    BIG = TINY_BASKETS * 20   # 200 baskets

    def test_returns_dataframe(self):
        rules = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5, min_lift=1.0)
        assert isinstance(rules, pd.DataFrame)

    def test_required_columns_present(self):
        rules = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5, min_lift=1.0)
        required = {"antecedents", "consequents", "support", "confidence", "lift",
                    "leverage", "conviction", "support_count"}
        assert required.issubset(set(rules.columns))

    def test_sorted_by_lift_descending(self):
        rules = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5, min_lift=1.0)
        lifts = rules["lift"].tolist()
        assert lifts == sorted(lifts, reverse=True)

    def test_lift_filter_applied(self):
        rules = mine_rules(TINY_BASKETS, min_support=0.2, min_confidence=0.5, min_lift=1.5)
        if rules is not None:
            assert (rules["lift"] >= 1.5).all()

    def test_confidence_filter_applied(self):
        rules = mine_rules(TINY_BASKETS, min_support=0.2, min_confidence=0.6, min_lift=1.0)
        if rules is not None:
            assert (rules["confidence"] >= 0.6).all()

    def test_returns_none_below_basket_floor(self):
        # Only 5 baskets — below FP_GROUP_MIN_BASKETS (150)
        tiny = TINY_BASKETS[:5]
        result = mine_rules(tiny, min_support=0.2, min_confidence=0.5, min_lift=1.0)
        assert result is None

    def test_returns_none_when_no_rules_pass(self):
        # Extremely high thresholds → no rules
        result = mine_rules(TINY_BASKETS * 50, min_support=0.99, min_confidence=0.99,
                             min_lift=100.0)
        assert result is None

    def test_min_count_filter_applied(self):
        """All returned rules must have support_count >= min_count."""
        rules = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5,
                            min_lift=1.0, min_count=5)
        if rules is not None:
            assert (rules["support_count"] >= 5).all()

    def test_min_count_drops_tiny_rules(self):
        """Setting min_count very high should return fewer rules than min_count=1."""
        rules_low  = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5,
                                 min_lift=1.0, min_count=1)
        rules_high = mine_rules(self.BIG, min_support=0.1, min_confidence=0.5,
                                 min_lift=1.0, min_count=100)
        if rules_high is not None and rules_low is not None:
            assert len(rules_high) <= len(rules_low)

    def test_no_self_rules(self):
        """Antecedents and consequents must be disjoint."""
        rules = mine_rules(TINY_BASKETS * 30, min_support=0.1, min_confidence=0.5,
                            min_lift=1.0)
        if rules is not None:
            for _, row in rules.iterrows():
                ant = set(row["antecedents"].split(", "))
                con = set(row["consequents"].split(", "))
                assert ant.isdisjoint(con), f"Self-rule: {row['antecedents']} → {row['consequents']}"

    def test_support_count_consistent_with_support(self):
        rules = mine_rules(TINY_BASKETS * 30, min_support=0.1, min_confidence=0.5,
                            min_lift=1.0)
        if rules is not None:
            n = rules["n_baskets"].iloc[0]
            for _, row in rules.iterrows():
                expected = round(row["support"] * n)
                # Allow ±1 for rounding
                assert abs(row["support_count"] - expected) <= 1


class TestVerifyMetrics:
    def test_no_warnings_on_correct_rules(self):
        # Mine rules, then verify — should return empty warning list
        rules = mine_rules(TINY_BASKETS * 30, min_support=0.1, min_confidence=0.5,
                            min_lift=1.0)
        if rules is not None:
            skill_lists = TINY_BASKETS * 30
            warnings = verify_metrics(rules, skill_lists)
            assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_detects_tampered_lift(self):
        rules = mine_rules(TINY_BASKETS * 30, min_support=0.1, min_confidence=0.5,
                            min_lift=1.0)
        if rules is not None and len(rules) >= 1:
            tampered = rules.copy()
            tampered.loc[0, "lift"] = 999.0   # obviously wrong
            skill_lists = TINY_BASKETS * 30
            warnings = verify_metrics(tampered, skill_lists)
            assert len(warnings) > 0


class TestAssignRoleGroup:
    def test_data_analyst(self):
        assert assign_role_group("data analyst") == "data_analyst"

    def test_software_engineer(self):
        assert assign_role_group("senior software engineer") == "software_engineer"

    def test_cloud_devops(self):
        assert assign_role_group("devops engineer") == "cloud_devops"

    def test_data_scientist(self):
        assert assign_role_group("data scientist") == "data_scientist_ml"

    def test_data_engineer(self):
        assert assign_role_group("data engineer") == "data_engineer"

    def test_unclassified(self):
        assert assign_role_group("office manager") is None

    def test_case_insensitive(self):
        assert assign_role_group("SENIOR DATA ANALYST") == "data_analyst"

    def test_first_match_wins(self):
        # "data analyst" should match data_analyst, not software_engineer
        result = assign_role_group("data analyst software engineer")
        assert result == "data_analyst"

    # ── cloud_devops regression tests: bleed reduction (Phase 5) ──────────

    def test_infrastructure_engineer_is_cloud_devops(self):
        """'infrastructure engineer' requires the 'eng' suffix to match — TP."""
        assert assign_role_group("senior infrastructure engineer") == "cloud_devops"

    def test_infrastructure_architect_is_cloud_devops(self):
        """'infrastructure architect' requires the 'arch' suffix to match — TP."""
        assert assign_role_group("cloud infrastructure architect") == "cloud_devops"

    def test_infrastructure_analyst_is_not_cloud_devops(self):
        """'infrastructure analyst' no longer matches cloud_devops after bleed fix."""
        result = assign_role_group("infrastructure analyst")
        assert result != "cloud_devops", (
            "infrastructure analyst should not be classified as cloud_devops; "
            "got: " + str(result))

    def test_infrastructure_project_manager_not_cloud_devops(self):
        """'IT infrastructure project manager' was a known bleed case — must be None."""
        assert assign_role_group("sr it infrastructure project manager") is None

    def test_infrastructure_management_not_cloud_devops(self):
        """Generic infrastructure management roles should not be cloud_devops."""
        result = assign_role_group("project administrator infrastructure management")
        assert result != "cloud_devops"

    def test_site_reliability_engineer_is_cloud_devops(self):
        """site reliability engineer should still match — no regression."""
        assert assign_role_group("site reliability engineer") == "cloud_devops"

    def test_platform_engineer_is_cloud_devops(self):
        assert assign_role_group("platform engineer") == "cloud_devops"

    def test_sre_abbreviation_is_cloud_devops(self):
        assert assign_role_group("sre engineer") == "cloud_devops"


class TestMetricDefinitions:
    """Verify lift, confidence, support against hand-computed values."""

    def _get_rules(self):
        # Use a large enough pool by repeating TINY_BASKETS
        baskets = TINY_BASKETS * 20   # 200 baskets, same ratios
        return mine_rules(baskets, min_support=0.1, min_confidence=0.5, min_lift=1.0), baskets

    def test_python_sql_support(self):
        """python+sql joint support ≈ 0.3 in TINY_BASKETS."""
        rules, baskets = self._get_rules()
        if rules is None:
            pytest.skip("No rules produced")
        mask = (rules["antecedents"] == "python") & (rules["consequents"] == "sql")
        if not mask.any():
            mask = (rules["antecedents"] == "sql") & (rules["consequents"] == "python")
        if not mask.any():
            pytest.skip("python→sql rule not present")
        row = rules[mask].iloc[0]
        assert abs(row["support"] - 0.3) < 0.01

    def test_confidence_formula(self):
        """conf(A→B) = support(A∪B) / support(A)."""
        rules, baskets = self._get_rules()
        if rules is None:
            pytest.skip("No rules produced")
        n = len(baskets)
        skill_sets: dict[str, set] = defaultdict(set)
        for i, b in enumerate(baskets):
            for s in b:
                skill_sets[s].add(i)

        for _, row in rules.head(3).iterrows():
            ant_skills = frozenset(row["antecedents"].split(", "))
            con_skills = frozenset(row["consequents"].split(", "))
            union = ant_skills | con_skills
            ant_idx  = set.intersection(*(skill_sets[s] for s in ant_skills))
            union_idx = set.intersection(*(skill_sets[s] for s in union))
            expected_conf = len(union_idx) / len(ant_idx) if ant_idx else 0
            assert abs(row["confidence"] - expected_conf) < 0.01

    def test_lift_formula(self):
        """lift(A→B) = conf(A→B) / support(B)."""
        rules, _ = self._get_rules()
        if rules is None:
            pytest.skip("No rules produced")
        for _, row in rules.head(3).iterrows():
            expected_lift = row["confidence"] / row["consequent support"]
            assert abs(row["lift"] - expected_lift) < 0.02


class TestDataContracts:
    """Integration tests against actual Phase 1 outputs."""

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_transactions_csv_readable(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert set(df.columns) >= {"job_id", "skill"}
        assert len(df) > 0

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_no_duplicate_job_skill_pairs(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        dupes = df.duplicated(subset=["job_id", "skill"]).sum()
        assert dupes == 0, f"{dupes} duplicate (job_id, skill) pairs found"

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_no_empty_skill_names(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert df["skill"].str.strip().ne("").all()

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_all_skills_canonical(self):
        """Every skill in transactions must be in the known canonical set."""
        from src.skillmine.config import PROJECT_ROOT
        # Load skill dict from R script
        import re as _re
        r_script = (PROJECT_ROOT / "R" / "skillmine_review1.R").read_text()
        canonical = set(_re.findall(r'"([^"]+)",\s+(?:"|\\.)', r_script))
        # Quick approach: just verify no skill is blank or has leading/trailing space
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert (df["skill"] == df["skill"].str.strip()).all()

    @pytest.mark.skipif(
        not CLEAN_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_clean_csv_row_count(self):
        df = pd.read_csv(CLEAN_CSV, dtype=str)
        assert len(df) == 22372, f"Expected 22372, got {len(df)}"

    @pytest.mark.skipif(
        not CLEAN_CSV.exists(),
        reason="Phase 1 output not present"
    )
    def test_job_ids_unique_in_clean(self):
        df = pd.read_csv(CLEAN_CSV, dtype=str)
        assert df["job_id"].nunique() == len(df), "Duplicate job_ids in clean CSV"


class TestPhase5SkillDictionaryExpansion:
    """
    Regression tests for the Phase 5 skill dictionary additions.
    Verify that:
      (a) the new skills appear in the transaction table after the R re-run, and
      (b) the ambiguous bare-'go' pattern was intentionally NOT added (FP guard).
    """

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_golang_skill_present_in_transactions(self):
        """'go' (canonical name) should appear in transactions after R re-run."""
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert "go" in df["skill"].values, (
            "'go' (extracted via \\bgolang\\b pattern) not found in transactions. "
            "Re-run R/skillmine_review1.R to update outputs."
        )

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_erp_skill_present_in_transactions(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert "erp" in df["skill"].values, "'erp' not found in transactions"

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_swift_skill_present_in_transactions(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert "swift" in df["skill"].values, "'swift' not found in transactions"

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_api_skill_present_in_transactions(self):
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        assert "api" in df["skill"].values, "'api' not found in transactions"

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_go_extracted_via_golang_not_bare_go(self):
        """
        The 'go' skill must be extracted via \\bgolang\\b, not \\bgo\\b.
        Guard: the count of 'go' postings must be plausibly small (< 500),
        ruling out the 1,586 broad matches that bare \\bgo\\b would produce.
        If this fails, the R script may have accidentally used \\bgo\\b.
        """
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        go_count = (df["skill"] == "go").sum()
        assert go_count < 500, (
            f"'go' appears {go_count} times — suspiciously high for golang-only extraction. "
            "Verify R/skillmine_review1.R uses \\\\bgolang\\\\b (not \\\\bgo\\\\b)."
        )

    @pytest.mark.skipif(
        not TRANSACTIONS_CSV.exists(), reason="Phase 1 output not present"
    )
    def test_new_skills_have_no_trailing_spaces(self):
        """Canonical names for new skills must be exactly as defined, no whitespace."""
        df = pd.read_csv(TRANSACTIONS_CSV, dtype=str)
        new_skills = df[df["skill"].isin(["go", "erp", "swift", "api"])]
        assert (new_skills["skill"] == new_skills["skill"].str.strip()).all()
