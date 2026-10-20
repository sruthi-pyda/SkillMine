"""
Tests for Phase 4 — K-Means posting clusters / role archetypes.

Run from project root:
    python3 -m pytest tests/test_posting_clusters.py -v
"""

import sys
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.skillmine.posting_clusters import (
    _skill_tokenizer,
    _suggest_archetype,
    build_tfidf_matrix,
    evaluate_k_range,
    export_assignments,
    export_silhouette,
    export_summaries,
    fit_clusters,
    load_posting_skills,
    summarise_clusters,
)
from src.skillmine.config import (
    CLEAN_CSV,
    CLUSTER_ASSIGNMENTS_CSV,
    CLUSTER_SILHOUETTE_CSV,
    CLUSTER_SUMMARIES_CSV,
    KMEANS_K_MIN,
    RANDOM_SEED,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

# Minimal clean-CSV-like data with clear cluster structure:
#   Group A: python-heavy ML/AI roles (12 postings)
#   Group B: java-heavy software engineering (12 postings)
#   Group C: sql/excel-heavy data analysis (12 postings)
_GROUP_A = "python;machine learning;scikit-learn;tensorflow;nlp"
_GROUP_B = "java;spring;git;rest api;agile;javascript"
_GROUP_C = "sql;excel;tableau;power bi;statistics;data visualization"

TINY_POSTINGS = pd.DataFrame(
    [
        {"job_id": f"j{i:03d}",
         "title": title,
         "company_name": "TestCo",
         "experience_level": "Mid",
         "work_type": "Full-time",
         "skills": skills,
         "n_skills": len(skills.split(";"))}
        for i, (title, skills) in enumerate(
            [("data scientist", _GROUP_A)] * 12
            + [("software engineer", _GROUP_B)] * 12
            + [("data analyst", _GROUP_C)] * 12
        )
    ]
)


def _make_tfidf():
    X, vec = build_tfidf_matrix(TINY_POSTINGS)
    return X, vec


# ── TestSkillTokenizer ─────────────────────────────────────────────────────

class TestSkillTokenizer:
    def test_single_skill(self):
        assert _skill_tokenizer("python") == ["python"]

    def test_multiple_skills(self):
        result = _skill_tokenizer("python;sql;aws")
        assert result == ["python", "sql", "aws"]

    def test_multiword_skill_preserved(self):
        result = _skill_tokenizer("machine learning;ms office")
        assert "machine learning" in result
        assert "ms office" in result

    def test_strips_whitespace(self):
        result = _skill_tokenizer(" python ; sql ")
        assert result == ["python", "sql"]

    def test_empty_string(self):
        assert _skill_tokenizer("") == []

    def test_no_empty_tokens(self):
        result = _skill_tokenizer("python;;sql")
        assert "" not in result


# ── TestLoadPostingSkills ──────────────────────────────────────────────────

class TestLoadPostingSkills:
    def _make_csv(self, tmp_path: Path) -> Path:
        df = pd.DataFrame({
            "job_id":      ["j1", "j2", "j3", "j4"],
            "title":       ["Data Scientist", "SWE", "Analyst", "HR Manager"],
            "company_name":["A", "B", "C", "D"],
            "skills":      ["python;sql", "java;git", "", ""],
            "n_skills":    [2, 2, 0, 0],
        })
        p = tmp_path / "clean.csv"
        df.to_csv(p, index=False)
        return p

    def test_returns_dataframe(self, tmp_path):
        df = load_posting_skills(self._make_csv(tmp_path))
        assert isinstance(df, pd.DataFrame)

    def test_zero_skill_postings_excluded(self, tmp_path):
        df = load_posting_skills(self._make_csv(tmp_path))
        assert (df["n_skills"] > 0).all()

    def test_correct_row_count(self, tmp_path):
        df = load_posting_skills(self._make_csv(tmp_path))
        assert len(df) == 2  # j1 and j2 have skills

    def test_required_columns_present(self, tmp_path):
        df = load_posting_skills(self._make_csv(tmp_path))
        assert {"job_id", "title", "skills", "n_skills"} <= set(df.columns)

    def test_job_ids_unique(self, tmp_path):
        df = load_posting_skills(self._make_csv(tmp_path))
        assert df["job_id"].nunique() == len(df)


# ── TestBuildTfidfMatrix ───────────────────────────────────────────────────

class TestBuildTfidfMatrix:
    def test_returns_sparse_matrix(self):
        X, vec = _make_tfidf()
        assert issparse(X) or hasattr(X, "toarray")

    def test_shape_correct(self):
        X, vec = _make_tfidf()
        assert X.shape[0] == len(TINY_POSTINGS)
        assert X.shape[1] > 0

    def test_vocabulary_contains_skills(self):
        _, vec = _make_tfidf()
        vocab = set(vec.get_feature_names_out())
        assert "python" in vocab
        assert "machine learning" in vocab  # multi-word preserved
        assert "sql" in vocab

    def test_multiword_skill_single_feature(self):
        """'machine learning' must appear as one feature, not two."""
        _, vec = _make_tfidf()
        vocab = set(vec.get_feature_names_out())
        assert "machine learning" in vocab
        # Should NOT be split into "machine" and "learning" separately
        assert "machine" not in vocab or "learning" not in vocab

    def test_no_negative_values(self):
        X, _ = _make_tfidf()
        assert X.data.min() >= 0

    def test_empty_df_returns_empty_matrix(self):
        empty = TINY_POSTINGS.iloc[0:0].copy()
        X, _ = build_tfidf_matrix(empty)
        assert X.shape[0] == 0

    def test_l2_normalised_rows(self):
        """Each row should have unit L2 norm (TF-IDF with norm='l2')."""
        X, _ = _make_tfidf()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).flatten())
        # Rows with at least one skill should have norm ≈ 1
        nonzero_mask = norms > 0
        assert np.allclose(norms[nonzero_mask], 1.0, atol=1e-6)


def issparse(x):
    return sp.issparse(x)


# ── TestEvaluateKRange ─────────────────────────────────────────────────────

class TestEvaluateKRange:
    def test_returns_list_of_dicts(self):
        X, _ = _make_tfidf()
        results = evaluate_k_range(X, k_range=range(2, 5), seed=RANDOM_SEED)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_required_keys(self):
        X, _ = _make_tfidf()
        results = evaluate_k_range(X, k_range=[3], seed=RANDOM_SEED)
        assert {"k", "silhouette_score", "n_samples", "note"} <= set(results[0].keys())

    def test_length_equals_k_range(self):
        X, _ = _make_tfidf()
        results = evaluate_k_range(X, k_range=range(2, 6), seed=RANDOM_SEED)
        assert len(results) == 4

    def test_silhouette_in_valid_range(self):
        X, _ = _make_tfidf()
        results = evaluate_k_range(X, k_range=range(2, 5), seed=RANDOM_SEED)
        for r in results:
            if r["silhouette_score"] is not None:
                assert -1.0 <= r["silhouette_score"] <= 1.0

    def test_k_exceeds_n_samples_handled(self):
        """k ≥ n should not raise; result should note it."""
        tiny = TINY_POSTINGS.head(3)
        X, _ = build_tfidf_matrix(tiny)
        results = evaluate_k_range(X, k_range=[10], seed=RANDOM_SEED)
        # Should return result with None score or note about range
        assert len(results) == 1
        r = results[0]
        assert r["silhouette_score"] is None or r["note"] != "ok"

    def test_reproducible_same_seed(self):
        X, _ = _make_tfidf()
        r1 = evaluate_k_range(X, k_range=range(2, 5), seed=42)
        r2 = evaluate_k_range(X, k_range=range(2, 5), seed=42)
        scores1 = [r["silhouette_score"] for r in r1]
        scores2 = [r["silhouette_score"] for r in r2]
        assert scores1 == scores2


# ── TestFitClusters ────────────────────────────────────────────────────────

class TestFitClusters:
    def test_returns_ndarray(self):
        X, _ = _make_tfidf()
        labels = fit_clusters(X, k=3, seed=RANDOM_SEED)
        assert isinstance(labels, np.ndarray)

    def test_label_count_matches_input(self):
        X, _ = _make_tfidf()
        labels = fit_clusters(X, k=3, seed=RANDOM_SEED)
        assert len(labels) == X.shape[0]

    def test_label_range(self):
        k = 3
        X, _ = _make_tfidf()
        labels = fit_clusters(X, k=k, seed=RANDOM_SEED)
        assert set(labels) == set(range(k))

    def test_reproducible_same_seed(self):
        X, _ = _make_tfidf()
        l1 = fit_clusters(X, k=3, seed=42)
        l2 = fit_clusters(X, k=3, seed=42)
        np.testing.assert_array_equal(l1, l2)

    def test_different_seeds_may_differ(self):
        """Not guaranteed, but different seeds often give different label orders."""
        X, _ = _make_tfidf()
        l1 = fit_clusters(X, k=3, seed=0)
        l2 = fit_clusters(X, k=3, seed=99)
        # Labels may be permuted even if the partition is equivalent;
        # we just verify both are valid
        assert len(l1) == len(l2) == X.shape[0]

    def test_cluster_count_matches_k(self):
        # TINY_POSTINGS has 3 distinct skill profiles; test only k ≤ 3.
        # With k > distinct points KMeans may produce fewer clusters (sklearn
        # ConvergenceWarning) — that edge case is a data-quality issue, not a
        # code bug, so we don't test it here.
        X, _ = _make_tfidf()
        for k in [2, 3]:
            labels = fit_clusters(X, k=k, seed=RANDOM_SEED)
            assert len(set(labels)) == k


# ── TestSummariseClusters ──────────────────────────────────────────────────

class TestSummariseClusters:
    def _make_summary(self, k=3):
        X, vec = _make_tfidf()
        labels = fit_clusters(X, k=k, seed=RANDOM_SEED)
        return summarise_clusters(labels, TINY_POSTINGS, vec, X), labels

    def test_returns_list(self):
        summaries, _ = self._make_summary()
        assert isinstance(summaries, list)

    def test_correct_number_of_clusters(self):
        k = 3
        summaries, _ = self._make_summary(k)
        assert len(summaries) == k

    def test_required_keys(self):
        summaries, _ = self._make_summary()
        for s in summaries:
            assert {"cluster_id", "size", "top_skills",
                    "rep_titles", "suggested_label"} <= set(s.keys())

    def test_sizes_sum_to_total(self):
        summaries, _ = self._make_summary()
        assert sum(s["size"] for s in summaries) == len(TINY_POSTINGS)

    def test_top_skills_nonempty(self):
        summaries, _ = self._make_summary()
        for s in summaries:
            assert len(s["top_skills"]) > 0

    def test_suggested_label_is_string(self):
        summaries, _ = self._make_summary()
        for s in summaries:
            assert isinstance(s["suggested_label"], str)
            assert len(s["suggested_label"]) > 0

    def test_cluster_ids_are_unique(self):
        summaries, _ = self._make_summary()
        ids = [s["cluster_id"] for s in summaries]
        assert len(ids) == len(set(ids))


# ── TestSuggestArchetype ───────────────────────────────────────────────────

class TestSuggestArchetype:
    def test_ml_family(self):
        skills = ["python", "machine learning", "tensorflow", "pytorch", "deep learning"]
        label = _suggest_archetype(skills)
        assert "ML" in label or "AI" in label

    def test_data_engineering_family(self):
        skills = ["spark", "kafka", "airflow", "etl", "hadoop", "databricks"]
        label = _suggest_archetype(skills)
        assert "Data Engineering" in label

    def test_cloud_devops_family(self):
        skills = ["aws", "docker", "kubernetes", "terraform", "ci/cd"]
        label = _suggest_archetype(skills)
        assert "Cloud" in label or "DevOps" in label

    def test_software_engineering_family(self):
        # Use backend-focused skills only (no frontend tokens that would fire
        # Web/Frontend Engineering first: no javascript/react/typescript/angular/vue/node.js)
        skills = ["java", "spring", "git", "agile", "c#", "jira", "rest api"]
        label = _suggest_archetype(skills)
        assert "Software" in label

    def test_data_analysis_family(self):
        skills = ["sql", "excel", "tableau", "power bi", "statistics"]
        label = _suggest_archetype(skills)
        assert "Data Analysis" in label or "BI" in label

    def test_fallback_for_single_match(self):
        label = _suggest_archetype(["python"])
        assert isinstance(label, str)

    def test_returns_string(self):
        assert isinstance(_suggest_archetype([]), str)

    # ── Phase 5 regression tests for updated archetype ordering ──────────

    def test_erp_requires_two_signals(self):
        """Enterprise/ERP requires both sap AND erp (threshold=2) to avoid
        mislabelling clusters where sap appears incidentally."""
        # Only sap present — should NOT fire Enterprise/ERP
        label_sap_only = _suggest_archetype(
            ["javascript", "react", "sql", "java", "api", "c#", "sap", "jira"])
        assert "Enterprise" not in label_sap_only, (
            "Single 'sap' in a web cluster must not trigger Enterprise/ERP; "
            f"got: {label_sap_only}")

    def test_erp_fires_when_both_sap_and_erp_present(self):
        """When both sap and erp are present, Enterprise/ERP should be labelled."""
        label = _suggest_archetype(["ms office", "excel", "sap", "erp", "sql", "r"])
        assert "Enterprise" in label or "ERP" in label, (
            f"sap+erp should trigger Enterprise/ERP; got: {label}")

    def test_web_frontend_before_enterprise_erp(self):
        """Web/Frontend must fire before Enterprise/ERP in the ordering."""
        # These skills represent a SW cluster with incidental sap — must be Web/Frontend
        label = _suggest_archetype(
            ["javascript", "react", "java", "sql", "api", "c#", "sap", "erp"])
        # sap+erp present (threshold=2 would fire) but Web/Frontend fires first
        assert "Web" in label or "Frontend" in label, (
            f"javascript+react should win over sap+erp in ordering; got: {label}")

    def test_new_skill_api_tokenized_correctly(self):
        """'api' was added to the skill dictionary in Phase 5; verify tokenizer handles it."""
        from src.skillmine.posting_clusters import _skill_tokenizer
        tokens = _skill_tokenizer("python;sql;api;rest api")
        assert "api" in tokens
        assert "rest api" in tokens

    def test_new_skill_erp_tokenized_correctly(self):
        from src.skillmine.posting_clusters import _skill_tokenizer
        assert "erp" in _skill_tokenizer("sap;erp;ms office")

    def test_new_skill_go_tokenized_correctly(self):
        """'go' canonical name must survive round-trip through tokenizer."""
        from src.skillmine.posting_clusters import _skill_tokenizer
        assert "go" in _skill_tokenizer("python;go;kubernetes")

    def test_new_skill_swift_tokenized_correctly(self):
        from src.skillmine.posting_clusters import _skill_tokenizer
        assert "swift" in _skill_tokenizer("swift;objective-c;ios")


# ── TestExports ────────────────────────────────────────────────────────────

class TestExports:
    def test_export_assignments(self, tmp_path):
        X, vec = _make_tfidf()
        labels = fit_clusters(X, k=3, seed=RANDOM_SEED)
        out = tmp_path / "posting_clusters.csv"
        df = export_assignments(TINY_POSTINGS, labels, out_path=out)
        assert out.exists()
        assert "cluster_id" in df.columns
        assert len(df) == len(TINY_POSTINGS)

    def test_export_summaries(self, tmp_path):
        X, vec = _make_tfidf()
        labels = fit_clusters(X, k=3, seed=RANDOM_SEED)
        summaries = summarise_clusters(labels, TINY_POSTINGS, vec, X)
        out = tmp_path / "cluster_summaries.csv"
        df = export_summaries(summaries, out_path=out)
        assert out.exists()
        assert len(df) == 3

    def test_export_silhouette(self, tmp_path):
        X, _ = _make_tfidf()
        results = evaluate_k_range(X, k_range=range(2, 4), seed=RANDOM_SEED)
        out = tmp_path / "silhouette_scores.csv"
        df = export_silhouette(results, out_path=out)
        assert out.exists()
        assert "k" in df.columns
        assert "silhouette_score" in df.columns

    def test_export_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "nested" / "dir" / "out.csv"
        X, vec = _make_tfidf()
        labels = fit_clusters(X, k=3, seed=RANDOM_SEED)
        export_assignments(TINY_POSTINGS, labels, out_path=nested)
        assert nested.exists()


# ── TestOutputFiles (integration) ─────────────────────────────────────────

class TestOutputFiles:
    """Integration tests: verify Phase 4 outputs exist and are valid."""

    @pytest.mark.skipif(not CLUSTER_ASSIGNMENTS_CSV.exists(), reason="Phase 4 not yet run")
    def test_assignments_has_required_columns(self):
        df = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
        assert {"job_id", "title", "cluster_id", "n_skills"} <= set(df.columns)

    @pytest.mark.skipif(not CLUSTER_ASSIGNMENTS_CSV.exists(), reason="Phase 4 not yet run")
    def test_assignments_no_null_cluster_ids(self):
        df = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV)
        assert df["cluster_id"].notna().all()

    @pytest.mark.skipif(not CLUSTER_ASSIGNMENTS_CSV.exists(), reason="Phase 4 not yet run")
    def test_assignments_job_ids_unique(self):
        df = pd.read_csv(CLUSTER_ASSIGNMENTS_CSV, dtype={"job_id": str})
        assert df["job_id"].nunique() == len(df)

    @pytest.mark.skipif(not CLUSTER_SUMMARIES_CSV.exists(), reason="Phase 4 not yet run")
    def test_summaries_has_required_columns(self):
        df = pd.read_csv(CLUSTER_SUMMARIES_CSV)
        assert {"cluster_id", "size", "top_skills",
                "rep_titles", "suggested_label"} <= set(df.columns)

    @pytest.mark.skipif(not CLUSTER_SUMMARIES_CSV.exists(), reason="Phase 4 not yet run")
    def test_summaries_sizes_positive(self):
        df = pd.read_csv(CLUSTER_SUMMARIES_CSV)
        assert (df["size"] > 0).all()

    @pytest.mark.skipif(not CLUSTER_SUMMARIES_CSV.exists(), reason="Phase 4 not yet run")
    def test_summaries_labels_nonempty(self):
        df = pd.read_csv(CLUSTER_SUMMARIES_CSV)
        assert df["suggested_label"].notna().all()
        assert (df["suggested_label"].str.strip() != "").all()

    @pytest.mark.skipif(not CLUSTER_SILHOUETTE_CSV.exists(), reason="Phase 4 not yet run")
    def test_silhouette_has_required_columns(self):
        df = pd.read_csv(CLUSTER_SILHOUETTE_CSV)
        assert {"k", "silhouette_score"} <= set(df.columns)

    @pytest.mark.skipif(not CLUSTER_SILHOUETTE_CSV.exists(), reason="Phase 4 not yet run")
    def test_silhouette_scores_valid_range(self):
        df = pd.read_csv(CLUSTER_SILHOUETTE_CSV)
        valid = df["silhouette_score"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    @pytest.mark.skipif(not CLUSTER_SILHOUETTE_CSV.exists(), reason="Phase 4 not yet run")
    def test_silhouette_covers_expected_k_range(self):
        from src.skillmine.config import KMEANS_K_MIN, KMEANS_K_MAX
        df = pd.read_csv(CLUSTER_SILHOUETTE_CSV)
        assert df["k"].min() <= KMEANS_K_MIN
        assert df["k"].max() >= KMEANS_K_MAX
