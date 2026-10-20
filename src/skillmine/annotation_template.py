"""
Phase 5 — Skill extraction annotation template generator.

Produces a CSV template for human review of the skill extraction quality.
A human annotator fills in 'actual_skills' and 'notes' for each sampled posting.
After annotation, precision/recall/F1 can be computed with evaluate_annotations().

Run from project root:
    python3 -m src.skillmine.annotation_template
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLEAN_CSV, OUTPUT_DIR, RANDOM_SEED

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_POSTINGS_CSV   = OUTPUT_DIR.parent / "data" / "linkedin_primary" / "postings.csv"
ANNOTATION_TEMPLATE_CSV = OUTPUT_DIR / "validation" / "skill_extraction_review.csv"
ANNOTATION_RESULTS_CSV  = OUTPUT_DIR / "validation" / "skill_extraction_metrics.csv"

SAMPLE_SIZE    = 100
# Stratification strata by n_skills
STRATA = [
    ("zero_skill",    lambda df: df["n_skills"] == 0,     25),
    ("low_skill",     lambda df: df["n_skills"].between(1, 2), 30),
    ("medium_skill",  lambda df: df["n_skills"].between(3, 5), 30),
    ("high_skill",    lambda df: df["n_skills"] >= 6,     15),
]

INSTRUCTIONS = """
==============================================================
SkillMine – Skill Extraction Quality Review Template
==============================================================
Instructions for annotators
------------------------------
1. For each row, read the 'description' and 'skills_desc' text
   in columns D and E.
2. In column 'actual_skills', list all genuine technical skills
   you can identify in the text, separated by semicolons.
   Use canonical names where possible (e.g. "python", "sql",
   "machine learning", "power bi"). If no technical skills are
   mentioned, leave the cell as "none".
3. In column 'notes', record any relevant observations
   (e.g. "contains golang but not currently in dictionary",
    "auto-extraction missed tensorflow", "false positive: ml
    matched 'email' but meant machine learning").
4. Do not modify any column except 'actual_skills' and 'notes'.
5. Save the file after completing each batch of rows.

Precision  = (auto-extracted ∩ actual) / auto-extracted
Recall     = (auto-extracted ∩ actual) / actual
F1         = 2 × precision × recall / (precision + recall)

These metrics will be computed after you have completed
all rows and saved the file.  Run:
    python3 -m src.skillmine.annotation_template --evaluate
==============================================================
"""


# ── Sampling ───────────────────────────────────────────────────────────────

def _stratified_sample(clean_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Sample SAMPLE_SIZE postings stratified by skill count.

    Strata and target counts:
      zero_skill   (n_skills == 0) : 25   — captures extraction misses
      low_skill    (1–2 skills)    : 30   — typical short baskets
      medium_skill (3–5 skills)    : 30   — moderate skill richness
      high_skill   (≥ 6 skills)    : 15   — complex postings

    All strata use the same fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for label, condition, target in STRATA:
        pool = clean_df[condition(clean_df)]
        n = min(target, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        sample = pool.iloc[idx].copy()
        sample["stratum"] = label
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def build_template(
    clean_csv: Path = CLEAN_CSV,
    raw_csv: Path = RAW_POSTINGS_CSV,
    out_path: Path = ANNOTATION_TEMPLATE_CSV,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build and save the annotation template CSV.

    Columns exported:
      row_num            — sequential row for reference
      job_id             — stable identifier
      title              — original job title
      stratum            — sampling stratum (for later stratified recall)
      auto_skills        — skills extracted by the pipeline (semicolon-sep)
      n_auto_skills      — count of auto-extracted skills
      description        — raw job description (from primary data)
      skills_desc        — LinkedIn-supplied skills description field
      actual_skills      — [BLANK — annotator fills this in]
      notes              — [BLANK — annotator fills this in]

    Personal identifiers (company names, applicant data) are excluded.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading clean CSV …")
    clean = pd.read_csv(clean_csv, dtype={"job_id": str})

    print("[2/4] Stratified sampling …")
    sample = _stratified_sample(clean, seed=seed)
    print(f"  Sample: {len(sample)} postings  "
          f"({sample['stratum'].value_counts().to_dict()})")

    print("[3/4] Joining raw descriptions …")
    sample_ids = set(sample["job_id"])
    raw_chunks = []
    for chunk in pd.read_csv(
        raw_csv,
        usecols=["job_id", "description", "skills_desc"],
        dtype={"job_id": str},
        chunksize=20000,
    ):
        sub = chunk[chunk["job_id"].isin(sample_ids)]
        if len(sub):
            raw_chunks.append(sub)
    if not raw_chunks:
        raise FileNotFoundError(f"Raw postings not found at {raw_csv}")
    raw = pd.concat(raw_chunks).drop_duplicates("job_id")

    # The clean CSV already has skills_desc; drop it before merge to avoid
    # duplicate column names (skills_desc_x / skills_desc_y) from the join.
    sample_slim = sample.drop(columns=["skills_desc"], errors="ignore")
    merged = sample_slim.merge(raw, on="job_id", how="left")
    merged["description"] = merged["description"].fillna("")
    merged["skills_desc"]  = merged["skills_desc"].fillna("")

    print("[4/4] Writing annotation template …")
    template = pd.DataFrame({
        "row_num":       range(1, len(merged) + 1),
        "job_id":        merged["job_id"],
        "title":         merged["title"],
        "stratum":       merged["stratum"],
        "auto_skills":   merged["skills"].fillna("none"),
        "n_auto_skills": merged["n_skills"].fillna(0).astype(int),
        "description":   merged["description"],
        "skills_desc":   merged["skills_desc"],
        "actual_skills": "",   # annotator fills in
        "notes":         "",   # annotator fills in
    })
    template.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(template)} rows)")
    print()
    print(INSTRUCTIONS)
    return template


# ── Evaluation ─────────────────────────────────────────────────────────────

def _parse_skills(cell: str) -> frozenset[str]:
    """Parse semicolon-separated skill string, normalised to lower-case."""
    if pd.isna(cell) or str(cell).strip().lower() in ("", "none", "nan"):
        return frozenset()
    return frozenset(s.strip().lower() for s in str(cell).split(";") if s.strip())


def evaluate_annotations(
    template_path: Path = ANNOTATION_TEMPLATE_CSV,
    results_path: Path  = ANNOTATION_RESULTS_CSV,
) -> dict | None:
    """
    Compute precision, recall, and F1 from a completed annotation template.

    The 'actual_skills' column must be filled in by a human annotator.
    Rows where 'actual_skills' is blank are treated as unannotated and
    reported separately rather than counted as metrics.

    Returns a dict with aggregate metrics and per-row detail, or None if
    the template has no annotations.
    """
    df = pd.read_csv(template_path, dtype=str)
    df["actual_skills"] = df["actual_skills"].fillna("")
    df["auto_skills"]   = df["auto_skills"].fillna("")

    annotated = df[df["actual_skills"].str.strip().ne("")].copy()
    unannotated = df[df["actual_skills"].str.strip().eq("")]

    if len(annotated) == 0:
        print(f"No annotations found in {template_path}.")
        print("Fill in the 'actual_skills' column and re-run.")
        return None

    # Per-row metrics
    rows = []
    for _, row in annotated.iterrows():
        auto   = _parse_skills(row["auto_skills"])
        actual = _parse_skills(row["actual_skills"])

        tp = len(auto & actual)
        fp = len(auto - actual)
        fn = len(actual - auto)

        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec  = tp / (tp + fn) if (tp + fn) > 0 else None
        f1   = (2 * prec * rec / (prec + rec)
                if prec is not None and rec is not None
                   and (prec + rec) > 0 else None)

        rows.append({
            "row_num":    row["row_num"],
            "job_id":     row["job_id"],
            "stratum":    row["stratum"],
            "auto_skills":   row["auto_skills"],
            "actual_skills": row["actual_skills"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall":    round(rec,  4) if rec  is not None else None,
            "f1":        round(f1,   4) if f1   is not None else None,
        })

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(results_path, index=False)

    # Macro-average (mean across postings)
    macro_prec = detail_df["precision"].dropna().mean()
    macro_rec  = detail_df["recall"].dropna().mean()
    macro_f1   = detail_df["f1"].dropna().mean()

    # Micro-average (pooled counts)
    total_tp = detail_df["tp"].sum()
    total_fp = detail_df["fp"].sum()
    total_fn = detail_df["fn"].sum()
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0)

    metrics = {
        "n_annotated":        len(annotated),
        "n_unannotated":      len(unannotated),
        "macro_precision":    round(macro_prec, 4),
        "macro_recall":       round(macro_rec,  4),
        "macro_f1":           round(macro_f1,   4),
        "micro_precision":    round(micro_prec, 4),
        "micro_recall":       round(micro_rec,  4),
        "micro_f1":           round(micro_f1,   4),
    }

    print("=== Skill Extraction Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")
    print(f"\n  Per-row detail saved to: {results_path}")
    return metrics


if __name__ == "__main__":
    import sys
    if "--evaluate" in sys.argv:
        evaluate_annotations()
    else:
        build_template()
