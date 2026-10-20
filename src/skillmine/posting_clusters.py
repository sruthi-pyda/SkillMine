"""
Phase 4 — K-Means posting clusters / role archetypes.

Pipeline:
  1. Load skill-bearing postings from Phase 1 clean CSV.
  2. Build TF-IDF matrix over per-posting skill sets (semicolon-separated).
  3. Evaluate KMeans for k = KMEANS_K_MIN..KMEANS_K_MAX using silhouette scores.
  4. Select k using silhouette + interpretability; fit final model.
  5. Summarise each cluster: top TF-IDF skills, representative titles, archetype label.
  6. Export posting-to-cluster assignments, cluster summaries, and silhouette metrics.

Run from project root:
    python3 -m src.skillmine.posting_clusters
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .config import (
    CLEAN_CSV,
    CLUSTER_ASSIGNMENTS_CSV,
    CLUSTER_SILHOUETTE_CSV,
    CLUSTER_SUMMARIES_CSV,
    CLUSTERS_DIR,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    KMEANS_MAX_ITER,
    KMEANS_MIN_CLUSTER_SIZE,
    KMEANS_N_INIT,
    KMEANS_TOP_SKILLS,
    KMEANS_TOP_TITLES,
    RANDOM_SEED,
)

# ── Archetype families for cluster labeling ────────────────────────────────
# Ordered from most specific to most general.  Each entry: (label, keywords, threshold).
# threshold = minimum number of family keywords that must appear in cluster's top skills.
# More specific / exclusive skills (linux, sap) use threshold=1 so they fire early.
_ARCHETYPE_FAMILIES: list[tuple[str, list[str], int]] = [
    # Ordering principle: check families with exclusive, high-frequency signals first.
    # A family with threshold=1 only fires on truly domain-exclusive skills (linux, sap).
    # Families with threshold=2 require at least two co-present signals.
    #
    # Key risk: "sap" is a valid ERP signal but can appear incidentally in web/SW
    # postings (e.g. SAP integrations).  We check Web/Frontend and Data Engineering
    # BEFORE Enterprise/ERP so that a cluster dominated by javascript+react is not
    # mislabelled simply because "sap" appears further down the centroid ranking.
    # Ordering rationale:
    # • Families with highly exclusive signals use threshold=1 (Systems/Infra).
    # • Web/Frontend and Cloud/DevOps are checked early so that incidental
    #   "sap" occurrences in SW/DevOps clusters don't trigger Enterprise/ERP.
    # • Enterprise/ERP uses threshold=2 (requires BOTH sap AND erp to fire),
    #   is placed AFTER web/SW/cloud but BEFORE Statistical Analysis — this
    #   prevents clusters where sap+erp dominate from being mislabelled as
    #   "Statistical Analysis" just because r/statistics also appear.
    ("Systems / Infrastructure",
     ["linux", "bash", "c++", "kernel", "embedded"],
     1),
    ("Data Engineering",
     ["spark", "kafka", "airflow", "etl", "hadoop", "databricks",
      "snowflake", "data warehousing", "scala", "dbt", "data modeling"],
     2),
    ("ML / AI",
     ["machine learning", "tensorflow", "pytorch", "deep learning",
      "nlp", "scikit-learn", "generative ai"],
     2),
    ("Web / Frontend Engineering",
     ["javascript", "react", "typescript", "angular", "vue", "node.js"],
     2),
    ("Cloud / DevOps",
     ["docker", "kubernetes", "terraform", "ci/cd", "gcp", "aws", "azure"],
     2),
    # Enterprise/ERP: threshold=2 (both sap AND erp must appear) prevents
    # incidental single-skill matches in SW/web clusters.
    ("Enterprise / ERP",
     ["sap", "erp"],
     2),
    # Statistical Analysis: checked after ERP so sap+erp clusters are already
    # captured and won't incorrectly match here via r/statistics.
    ("Statistical Analysis",
     ["r", "statistics", "sas", "matlab"],
     2),
    ("Software Engineering",
     ["java", "git", "agile", "c#", "spring", "rest api", "selenium",
      "redis", "mongodb", "flask", "django", "jira"],
     2),
    # "powerpoint" omitted from BI — too generic to be a reliable signal.
    ("Data Analysis / BI",
     ["sql", "excel", "tableau", "power bi", "looker", "data visualization",
      "statistics"],
     2),
    ("General / Business",
     ["communication", "ms office", "powerpoint"],
     2),
]


# ── Core functions ─────────────────────────────────────────────────────────

def _skill_tokenizer(text: str) -> list[str]:
    """Split semicolon-separated skill string; preserve multi-word skill tokens."""
    return [s.strip() for s in text.split(";") if s.strip()]


def load_posting_skills(clean_csv: Path = CLEAN_CSV) -> pd.DataFrame:
    """
    Load clean postings, return only those with ≥1 extracted skill.

    Returns DataFrame with columns: job_id, title, company_name,
    experience_level, work_type, skills (raw string), n_skills.
    Zero-skill postings are excluded (documented in run_phase4 stats).
    """
    df = pd.read_csv(clean_csv, dtype={"job_id": str})
    keep_cols = ["job_id", "title", "company_name", "experience_level",
                 "work_type", "skills", "n_skills"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    df["skills"] = df["skills"].fillna("").str.strip()
    # Only keep postings with at least one skill
    df = df[df["skills"] != ""].reset_index(drop=True)
    # Ensure n_skills is int
    if "n_skills" in df.columns:
        df["n_skills"] = df["n_skills"].fillna(0).astype(int)
        df = df[df["n_skills"] > 0].reset_index(drop=True)
    return df


def build_tfidf_matrix(posting_df: pd.DataFrame):
    """
    Build sparse TF-IDF matrix from per-posting skill strings.

    Uses a custom tokenizer that splits on ';' so that multi-word skills
    (e.g. 'machine learning', 'ms office') are treated as single tokens.

    Returns:
        X          : sparse CSR matrix (n_postings × n_skills)
        vectorizer : fitted TfidfVectorizer
    """
    if posting_df.empty:
        vectorizer = TfidfVectorizer(
            tokenizer=_skill_tokenizer, token_pattern=None, lowercase=False
        )
        import scipy.sparse as sp
        return sp.csr_matrix((0, 0)), vectorizer

    vectorizer = TfidfVectorizer(
        tokenizer=_skill_tokenizer,
        token_pattern=None,   # required when providing a custom tokenizer
        lowercase=False,
        norm="l2",
    )
    X = vectorizer.fit_transform(posting_df["skills"])
    return X, vectorizer


def evaluate_k_range(
    X,
    k_range: range | list[int] | None = None,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Fit KMeans for each k in k_range and compute silhouette score.

    Silhouette requires ≥ 2 clusters and ≥ 2 samples per cluster.
    k values that fail these constraints are skipped with a note.

    Returns list of dicts: {k, silhouette_score, n_samples, note}.
    """
    if k_range is None:
        k_range = range(KMEANS_K_MIN, KMEANS_K_MAX + 1)

    n = X.shape[0]
    results = []
    for k in k_range:
        if k < 2 or k >= n:
            results.append({"k": k, "silhouette_score": None,
                             "n_samples": n, "note": "k out of valid range"})
            continue
        km = KMeans(
            n_clusters=k,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=seed,
        )
        labels = km.fit_predict(X)
        # Check that every cluster has ≥ 2 members
        counts = Counter(labels)
        if min(counts.values()) < 2:
            results.append({"k": k, "silhouette_score": None,
                             "n_samples": n, "note": "singleton cluster present"})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = silhouette_score(X, labels, metric="euclidean",
                                     random_state=seed)
        results.append({"k": k, "silhouette_score": round(float(score), 6),
                        "n_samples": n, "note": "ok"})
    return results


def _select_k(silhouette_results: list[dict]) -> int:
    """
    Select k from silhouette results using a parsimony + interpretability rule.

    When silhouette scores increase monotonically (common with sparse TF-IDF data),
    the highest k is not necessarily the most useful: it over-fragments into clusters
    with duplicate role-archetype labels.  Instead we apply a relative threshold:
    choose the *smallest* k whose silhouette falls within 5 percentage points of
    the best observed score.  This aligns with the master plan guidance to prefer
    interpretable clusters over marginally higher scores.

    Example: if scores are 0.167 (k=6) … 0.232 (k=12), threshold=0.05 means we
    accept any k with score ≥ 0.182 → smallest qualifying k is 8.
    """
    valid = [(r["k"], r["silhouette_score"])
             for r in silhouette_results if r["silhouette_score"] is not None]
    if not valid:
        return KMEANS_K_MIN
    best_score = max(s for _, s in valid)
    # 5-point relative gap allows meaningful parsimony on monotone score curves
    plateau_threshold = 0.05
    candidates = [(k, s) for k, s in valid if (best_score - s) <= plateau_threshold]
    return min(candidates, key=lambda x: x[0])[0]


def fit_clusters(X, k: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Fit final KMeans with chosen k, return integer label array."""
    km = KMeans(
        n_clusters=k,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=seed,
    )
    return km.fit_predict(X)


def _suggest_archetype(top_skills: list[str]) -> str:
    """
    Return an evidence-based archetype label for a cluster given its top skills.

    Families are checked in specificity order (most exclusive first). The first
    family whose per-family threshold is met wins. Falls back to 'Mixed / General'
    if no family threshold is reached.
    """
    top_lower = {s.lower() for s in top_skills}
    for label, keywords, threshold in _ARCHETYPE_FAMILIES:
        count = sum(1 for kw in keywords if kw.lower() in top_lower)
        if count >= threshold:
            return label
    return "Mixed / General"


def summarise_clusters(
    labels: np.ndarray,
    posting_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X,
    top_n_skills: int = KMEANS_TOP_SKILLS,
    top_n_titles: int = KMEANS_TOP_TITLES,
) -> list[dict]:
    """
    Build per-cluster summary: size, top TF-IDF skills, representative titles,
    archetype label.

    Cluster centroids in TF-IDF space capture the most characteristic skills.
    Job titles in each cluster provide human-interpretable context.
    """
    feature_names = vectorizer.get_feature_names_out()
    posting_df = posting_df.copy()
    posting_df["cluster_id"] = labels

    # Compute per-cluster centroid in TF-IDF space
    # X is sparse; sum each cluster's rows then normalise
    k = len(np.unique(labels))
    summaries = []

    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        cluster_size = int(mask.sum())
        if cluster_size == 0:
            continue

        # Centroid = mean TF-IDF vector across cluster members
        X_cluster = X[mask]
        if issparse(X_cluster):
            centroid = np.asarray(X_cluster.mean(axis=0)).flatten()
        else:
            centroid = X_cluster.mean(axis=0)

        # Top skills by centroid weight
        top_idx = np.argsort(centroid)[::-1][:top_n_skills]
        top_skills = [feature_names[i] for i in top_idx if centroid[i] > 0]

        # Representative titles by frequency in this cluster
        titles_in_cluster = posting_df.loc[mask, "title"].dropna()
        title_counts = Counter(titles_in_cluster.str.lower().tolist())
        rep_titles = [t for t, _ in title_counts.most_common(top_n_titles)]

        archetype = _suggest_archetype(top_skills)

        summaries.append({
            "cluster_id":       cid,
            "size":             cluster_size,
            "top_skills":       ";".join(top_skills),
            "rep_titles":       ";".join(rep_titles),
            "suggested_label":  archetype,
        })
    return summaries


# ── Export helpers ─────────────────────────────────────────────────────────

def export_assignments(
    posting_df: pd.DataFrame,
    labels: np.ndarray,
    out_path: Path = CLUSTER_ASSIGNMENTS_CSV,
) -> pd.DataFrame:
    """Export posting-to-cluster assignments CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = posting_df[["job_id", "title", "company_name", "n_skills"]].copy()
    df["cluster_id"] = labels
    df.to_csv(out_path, index=False)
    return df


def export_summaries(
    summaries: list[dict],
    out_path: Path = CLUSTER_SUMMARIES_CSV,
) -> pd.DataFrame:
    """Export cluster summaries CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(out_path, index=False)
    return df


def export_silhouette(
    results: list[dict],
    out_path: Path = CLUSTER_SILHOUETTE_CSV,
) -> pd.DataFrame:
    """Export silhouette scores CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_phase4(
    clean_csv: Path = CLEAN_CSV,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Full Phase 4 pipeline.

    Returns a stats dict for reporting.
    """
    print("Phase 4 — K-Means Posting Clusters")
    print("=" * 50)

    # 1. Load data
    print("\n[1/6] Loading posting skills …")
    posting_df = load_posting_skills(clean_csv)
    n_postings = len(posting_df)
    print(f"  Skill-bearing postings: {n_postings:,}")

    # 2. Build TF-IDF
    print("\n[2/6] Building TF-IDF matrix …")
    X, vectorizer = build_tfidf_matrix(posting_df)
    n_features = X.shape[1]
    print(f"  Matrix: {X.shape[0]:,} × {n_features} (sparse={issparse(X)})")

    # 3. Evaluate k range
    k_range = range(KMEANS_K_MIN, KMEANS_K_MAX + 1)
    print(f"\n[3/6] Evaluating k = {KMEANS_K_MIN}–{KMEANS_K_MAX} …")
    silhouette_results = evaluate_k_range(X, k_range, seed=seed)
    for r in silhouette_results:
        score_str = f"{r['silhouette_score']:.4f}" if r["silhouette_score"] is not None else "n/a"
        print(f"  k={r['k']:2d}  silhouette={score_str}  [{r['note']}]")

    # 4. Select k
    chosen_k = _select_k(silhouette_results)
    # Interpretability override: if best-scoring k creates any cluster smaller
    # than KMEANS_MIN_CLUSTER_SIZE, try the next-smaller k.
    while chosen_k > KMEANS_K_MIN:
        test_labels = fit_clusters(X, chosen_k, seed=seed)
        min_size = min(Counter(test_labels).values())
        if min_size >= KMEANS_MIN_CLUSTER_SIZE:
            break
        chosen_k -= 1

    print(f"\n[4/6] Selected k = {chosen_k}")
    valid_scores = [(r["k"], r["silhouette_score"]) for r in silhouette_results
                    if r["silhouette_score"] is not None]
    if valid_scores:
        best_k, best_s = max(valid_scores, key=lambda x: x[1])
        chosen_s = next((r["silhouette_score"] for r in silhouette_results
                         if r["k"] == chosen_k), None)
        print(f"  Best silhouette: k={best_k} → {best_s:.4f}")
        print(f"  Chosen silhouette: k={chosen_k} → {chosen_s:.4f}")
        print(f"  (Parsimony: smallest k within 0.005 of best; min cluster ≥ {KMEANS_MIN_CLUSTER_SIZE})")

    # 5. Fit final model
    print(f"\n[5/6] Fitting final KMeans (k={chosen_k}, seed={seed}) …")
    labels = fit_clusters(X, chosen_k, seed=seed)
    cluster_sizes = sorted(Counter(labels).values(), reverse=True)
    print(f"  Cluster sizes: {cluster_sizes}")

    # 6. Summarise and export
    print("\n[6/6] Summarising clusters and exporting …")
    summaries = summarise_clusters(labels, posting_df, vectorizer, X)
    for s in summaries:
        print(f"  C{s['cluster_id']:2d}  {s['suggested_label']:<28}  "
              f"n={s['size']:5,}  top: {s['top_skills'][:60]}")

    export_assignments(posting_df, labels)
    export_summaries(summaries)
    export_silhouette(silhouette_results)

    print(f"\n  Outputs written to {CLUSTERS_DIR}/")
    print("  posting_clusters.csv   — posting-to-cluster assignments")
    print("  cluster_summaries.csv  — cluster descriptions and archetype labels")
    print("  silhouette_scores.csv  — silhouette score for each evaluated k")

    chosen_score = next((r["silhouette_score"] for r in silhouette_results
                         if r["k"] == chosen_k), None)

    return {
        "n_postings_clustered": n_postings,
        "n_features":           n_features,
        "k_range":              f"{KMEANS_K_MIN}–{KMEANS_K_MAX}",
        "chosen_k":             chosen_k,
        "chosen_silhouette":    chosen_score,
        "cluster_sizes":        cluster_sizes,
        "summaries":            summaries,
    }


if __name__ == "__main__":
    run_phase4()
