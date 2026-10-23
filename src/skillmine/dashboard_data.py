"""
Phase 6 — Data access layer for the Streamlit dashboard.

Every function here reads *precomputed* artifacts from output/ (Phases 1–5)
and derives display tables from them.  Nothing here re-runs mining
algorithms.  The module has no Streamlit dependency so it can be unit-tested.

Missing artifacts raise MissingArtifact, which carries the command that
regenerates the file so the UI can show an actionable message.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.skillmine.config import (
    CLEAN_CSV, TRANSACTIONS_CSV, TABLES_DIR, BEFORE_AFTER_CSV, RULES_SUMMARY_CSV,
    NODES_CSV, EDGES_CSV, COMMUNITIES_CSV, NETWORK_HTML, NETWORK_PRES_HTML,
    PYVIS_LIB_DIR, CLUSTER_ASSIGNMENTS_CSV, CLUSTER_SUMMARIES_CSV,
    CLUSTER_SILHOUETTE_CSV, TREND_MIN_MONTHS,
)

R_CMD     = "Rscript R/skillmine_review1.R"
RULES_CMD = "python -m src.skillmine.association_rules"
NET_CMD   = "python -m src.skillmine.skill_network"
CLUST_CMD = "python -m src.skillmine.posting_clusters"


class MissingArtifact(FileNotFoundError):
    """A required output file does not exist yet."""

    def __init__(self, path: Path, command: str):
        self.path = Path(path)
        self.command = command
        super().__init__(f"Missing {self.path} — generate it with: {command}")


def require(path: Path, command: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise MissingArtifact(path, command)
    return path


def file_signature(*paths: Path) -> tuple:
    """Modification times of the given files; used as a cache key so the
    dashboard reloads automatically when the pipeline rewrites an artifact."""
    return tuple(Path(p).stat().st_mtime if Path(p).exists() else None for p in paths)


# ── Loaders ───────────────────────────────────────────────────────────────

def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, R_CMD), dtype={"job_id": str})


def load_transactions(path: Path = TRANSACTIONS_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, R_CMD), dtype=str)


def load_before_after(path: Path = BEFORE_AFTER_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, R_CMD))


def load_rules_summary(path: Path = RULES_SUMMARY_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, RULES_CMD))


def load_rules(role: str, tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """
    Load the *full* rule table for one run ("overall" or a role group).

    The *_top50.csv / *_top20.csv files are deliberately not used: the role
    top-50 files were found to be left over from an earlier run (different
    n_baskets than the full tables and rules_summary.csv).  Top-N is derived
    here from the full table instead.
    """
    df = pd.read_csv(require(Path(tables_dir) / f"rules_{role}.csv", RULES_CMD))
    df["n_antecedents"] = df["antecedents"].map(lambda s: len(parse_itemset(s)))
    df["n_consequents"] = df["consequents"].map(lambda s: len(parse_itemset(s)))
    df["rule"] = df["antecedents"] + "  →  " + df["consequents"]
    return df


def load_nodes(path: Path = NODES_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, NET_CMD))


def load_edges(path: Path = EDGES_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, NET_CMD))


def load_communities(path: Path = COMMUNITIES_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, NET_CMD))


def load_cluster_assignments(path: Path = CLUSTER_ASSIGNMENTS_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, CLUST_CMD), dtype={"job_id": str})


def load_cluster_summaries(path: Path = CLUSTER_SUMMARIES_CSV) -> pd.DataFrame:
    df = pd.read_csv(require(path, CLUST_CMD))
    # Several clusters can share an archetype label (e.g. three "Enterprise / ERP"
    # clusters), so always carry the id alongside it.
    df["display_label"] = "C" + df["cluster_id"].astype(str) + " · " + df["suggested_label"]
    return df


def load_silhouette(path: Path = CLUSTER_SILHOUETTE_CSV) -> pd.DataFrame:
    return pd.read_csv(require(path, CLUST_CMD))


def load_network_html(presentation: bool = True,
                      lib_dir: Path = PYVIS_LIB_DIR) -> str:
    """
    Return the PyVis HTML, made self-contained for embedding in an iframe.

    PyVis references lib/bindings/utils.js by a relative path (which does not
    resolve inside Streamlit's component iframe) and stray ../node_modules
    template tags.  The local vis-network copy under lib/ is inlined in place
    of the CDN tags when present, so the graph also renders offline.
    """
    path = require(NETWORK_PRES_HTML if presentation else NETWORK_HTML, NET_CMD)
    html = path.read_text(encoding="utf-8")
    lib_dir = Path(lib_dir)

    html = re.sub(r'<link[^>]*node_modules[^>]*>\s*', "", html)
    html = re.sub(r'<script[^>]*node_modules[^>]*>\s*</script>\s*', "", html)

    def inline_script(pattern: str, local: Path) -> None:
        nonlocal html
        if local.exists():
            body = local.read_text(encoding="utf-8")
            html = re.sub(pattern, lambda _: f"<script>{body}</script>", html)

    def inline_style(pattern: str, local: Path) -> None:
        nonlocal html
        if local.exists():
            body = local.read_text(encoding="utf-8")
            html = re.sub(pattern, lambda _: f"<style>{body}</style>", html)

    inline_script(r'<script[^>]*src="lib/bindings/utils\.js"[^>]*>\s*</script>',
                  lib_dir / "bindings" / "utils.js")
    inline_script(r'<script[^>]*src="[^"]*vis-network[^"]*\.js"[^>]*>\s*</script>',
                  lib_dir / "vis-9.1.2" / "vis-network.min.js")
    inline_style(r'<link[^>]*href="[^"]*vis-network[^"]*\.css"[^>]*>',
                 lib_dir / "vis-9.1.2" / "vis-network.css")
    return html


# ── Itemset helpers ───────────────────────────────────────────────────────

def parse_itemset(s: str) -> frozenset:
    """'docker, gcp' → frozenset({'docker', 'gcp'})"""
    if not isinstance(s, str) or not s.strip():
        return frozenset()
    return frozenset(x.strip() for x in s.split(",") if x.strip())


def filter_rules(
    rules: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
    skills: list[str] | None = None,
    side: str = "either",
    max_items: int | None = None,
) -> pd.DataFrame:
    """
    Filter a rule table.

    skills — keep rules mentioning ALL of these skills on the chosen side
             ("antecedent", "consequent" or "either").
    max_items — maximum total skills in the rule (antecedent + consequent).
    """
    if side not in {"either", "antecedent", "consequent"}:
        raise ValueError(f"side must be either/antecedent/consequent, got {side!r}")
    df = rules[(rules["support"] >= min_support)
               & (rules["confidence"] >= min_confidence)
               & (rules["lift"] >= min_lift)]
    if max_items is not None:
        df = df[df["n_antecedents"] + df["n_consequents"] <= max_items]
    if skills:
        wanted = set(skills)
        ant = df["antecedents"].map(parse_itemset)
        con = df["consequents"].map(parse_itemset)
        if side == "antecedent":
            mask = ant.map(lambda a: wanted <= a)
        elif side == "consequent":
            mask = con.map(lambda c: wanted <= c)
        else:
            mask = pd.Series([wanted <= (a | c) for a, c in zip(ant, con)],
                             index=df.index, dtype=bool)
        # Boolean Series, not a list: df[[]] on an empty frame would select
        # zero *columns* rather than zero rows.
        df = df[mask.astype(bool)]
    return df


# ── Overview ──────────────────────────────────────────────────────────────

def overview_stats(clean: pd.DataFrame, txn: pd.DataFrame,
                   before_after: pd.DataFrame) -> dict:
    """Headline numbers, all computed from the current artifacts."""
    before = before_after[before_after["stage"] == "Before cleaning"].iloc[0]
    dates = pd.to_datetime(clean["listed_date"], errors="coerce")
    skilled = txn["job_id"].nunique()
    n = len(clean)
    return {
        "raw_rows":            int(before["rows"]),
        "raw_cols":            int(before["cols"]),
        "duplicate_text":      int(before["duplicate_text"]),
        "postings":            n,
        "skilled_postings":    skilled,
        "zero_skill_postings": n - skilled,
        "n_skills":            txn["skill"].nunique(),
        "n_pairs":             len(txn),
        "mean_skills":         len(txn) / n if n else 0.0,
        "date_min":            dates.min(),
        "date_max":            dates.max(),
        "n_months":            clean["listed_month"].nunique(),
    }


# ── Skill-level views ─────────────────────────────────────────────────────

def skill_frequencies(txn: pd.DataFrame, n_postings: int) -> pd.DataFrame:
    """
    One row per skill:
      postings       — number of postings mentioning the skill
      share_all      — postings / all filtered postings (incl. zero-skill)
      share_skilled  — postings / postings with ≥1 skill (the FP-Growth /
                       network basket population)
    """
    n_skilled = txn["job_id"].nunique()
    freq = (txn.drop_duplicates(["job_id", "skill"])
               .groupby("skill").size().rename("postings").reset_index())
    freq["share_all"] = freq["postings"] / n_postings if n_postings else 0.0
    freq["share_skilled"] = freq["postings"] / n_skilled if n_skilled else 0.0
    return freq.sort_values(["postings", "skill"], ascending=[False, True],
                            ignore_index=True)


def skill_rules(rules: pd.DataFrame, skill: str, top: int = 10) -> dict:
    """Strongest rules (by lift) where the skill is the antecedent side vs the
    consequent side.  Returns {"as_antecedent": df, "as_consequent": df}."""
    by_lift = rules.sort_values(["lift", "support"], ascending=False)
    return {
        "as_antecedent": filter_rules(by_lift, skills=[skill], side="antecedent").head(top),
        "as_consequent": filter_rules(by_lift, skills=[skill], side="consequent").head(top),
    }


def skill_neighbours(edges: pd.DataFrame, skill: str, top: int = 10) -> pd.DataFrame:
    """Network partners of a skill, strongest normalised PPMI first."""
    a = edges[edges["skill_a"] == skill].rename(columns={"skill_b": "partner"})
    b = edges[edges["skill_b"] == skill].rename(columns={"skill_a": "partner"})
    both = pd.concat([a, b], ignore_index=True)
    cols = ["partner", "co_count", "pmi", "ppmi_norm"]
    if both.empty:
        return pd.DataFrame(columns=cols)
    return (both[cols].sort_values(["ppmi_norm", "co_count"], ascending=False)
                      .head(top).reset_index(drop=True))


def cluster_skill_shares(txn: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Long table: cluster_id, skill, postings, cluster_size, share
    where share = postings in the cluster mentioning the skill / cluster size.
    """
    sizes = assignments.groupby("cluster_id").size().rename("cluster_size")
    joined = (txn.drop_duplicates(["job_id", "skill"])
                 .merge(assignments[["job_id", "cluster_id"]], on="job_id", how="inner"))
    counts = joined.groupby(["cluster_id", "skill"]).size().rename("postings").reset_index()
    counts = counts.merge(sizes, left_on="cluster_id", right_index=True)
    counts["share"] = counts["postings"] / counts["cluster_size"]
    return counts


def skill_cluster_distribution(shares: pd.DataFrame, summaries: pd.DataFrame,
                               skill: str) -> pd.DataFrame:
    """For one skill: how many postings in each cluster mention it."""
    df = shares[shares["skill"] == skill]
    df = df.merge(summaries[["cluster_id", "display_label"]], on="cluster_id", how="left")
    total = df["postings"].sum()
    df = df.assign(share_of_skill=df["postings"] / total if total else 0.0)
    return df.sort_values("postings", ascending=False, ignore_index=True)


def postings_with_skill(clean: pd.DataFrame, txn: pd.DataFrame, skill: str,
                        n: int = 25) -> pd.DataFrame:
    """Sample of historical postings mentioning the skill (deterministic order)."""
    ids = set(txn.loc[txn["skill"] == skill, "job_id"])
    cols = [c for c in ["title", "company_name", "location", "experience_level",
                        "work_type", "listed_date"] if c in clean.columns]
    return (clean[clean["job_id"].isin(ids)]
            .sort_values(["listed_date", "job_id"], ascending=[False, True])
            [cols].head(n).reset_index(drop=True))


# ── Trends ────────────────────────────────────────────────────────────────

def trend_status(clean: pd.DataFrame, min_months: int = TREND_MIN_MONTHS) -> dict:
    """
    Decide whether monthly-share trend analysis is supported by the data.
    Never returns available=True unless there are ≥ min_months distinct months.
    """
    months = sorted(clean["listed_month"].dropna().unique().tolist())
    dates = pd.to_datetime(clean["listed_date"], errors="coerce")
    available = len(months) >= min_months
    if available:
        reason = f"{len(months)} distinct months available."
    else:
        reason = (f"Only {len(months)} distinct month(s) in the data "
                  f"({dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}); "
                  f"at least {min_months} are needed to fit a trend.")
    return {"available": available, "n_months": len(months), "months": months,
            "min_months": min_months, "date_min": dates.min(),
            "date_max": dates.max(), "reason": reason}
