"""
Phase 2 — FP-Growth association rules.

Usage (from project root):
    python3 -m src.skillmine.association_rules

Outputs written to output/tables/:
    rules_overall.csv          — all rules, full dataset, ranked by lift
    rules_overall_top20.csv    — top 20 rules by lift, for quick review
    rules_<role>.csv           — rules per role group
    rules_summary.csv          — one row per run: thresholds, counts, top rule
"""

import sys
import re
import csv
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.skillmine.config import (
    CLEAN_CSV, TRANSACTIONS_CSV, TABLES_DIR,
    FP_MIN_SUPPORT_OVERALL, FP_MIN_CONFIDENCE, FP_MIN_LIFT,
    FP_MIN_SUPPORT_GROUP, FP_MIN_CONFIDENCE_GROUP, FP_MIN_LIFT_GROUP,
    FP_GROUP_MIN_BASKETS, FP_MIN_COUNT, ROLE_GROUPS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────

def load_baskets(transactions_csv: Path = TRANSACTIONS_CSV,
                 clean_csv: Path = CLEAN_CSV) -> pd.DataFrame:
    """
    Return a DataFrame with one row per posting that has ≥1 extracted skill.
    Columns: job_id, skills (frozenset), title_lower
    """
    txn = pd.read_csv(transactions_csv, dtype=str)
    clean = pd.read_csv(clean_csv, dtype=str)

    # Build posting → skill set (deduplicated)
    baskets = (txn.groupby("job_id")["skill"]
                  .apply(frozenset)
                  .reset_index()
                  .rename(columns={"skill": "skills"}))

    # Join title for role-group assignment
    titles = clean[["job_id", "title"]].copy()
    titles["title_lower"] = titles["title"].str.lower().fillna("")
    baskets = baskets.merge(titles[["job_id", "title_lower"]], on="job_id", how="left")

    log.info("Loaded %d baskets from %d postings", len(baskets), len(clean))
    return baskets


def assign_role_group(title: str) -> str | None:
    """Return the first matching role group name, or None.
    Input is lowercased internally so callers may pass raw or pre-lowered titles.
    """
    title_lower = title.lower()
    for group, pattern in ROLE_GROUPS.items():
        if re.search(pattern, title_lower):
            return group
    return None


# ── Binary matrix builder ─────────────────────────────────────────────────

def build_binary_matrix(skill_lists: list[list[str]]) -> pd.DataFrame:
    """Encode list-of-lists as a binary DataFrame for mlxtend."""
    te = TransactionEncoder()
    te_array = te.fit_transform(skill_lists)
    return pd.DataFrame(te_array, columns=te.columns_)


# ── Rule mining ───────────────────────────────────────────────────────────

def mine_rules(
    skill_lists: list[list[str]],
    min_support: float,
    min_confidence: float,
    min_lift: float,
    label: str = "overall",
    min_count: int = FP_MIN_COUNT,
) -> pd.DataFrame | None:
    """
    Run FP-Growth + association_rules on the given skill lists.
    Returns a clean DataFrame of rules, or None if no rules pass thresholds.
    """
    n_baskets = len(skill_lists)
    log.info("[%s] Mining %d baskets | min_support=%.3f min_conf=%.2f min_lift=%.2f",
             label, n_baskets, min_support, min_confidence, min_lift)

    if n_baskets < FP_GROUP_MIN_BASKETS:
        log.warning("[%s] Only %d baskets — below floor of %d; skipping.",
                    label, n_baskets, FP_GROUP_MIN_BASKETS)
        return None

    bm = build_binary_matrix(skill_lists)

    try:
        itemsets = fpgrowth(bm, min_support=min_support, use_colnames=True)
    except Exception as exc:
        log.error("[%s] fpgrowth failed: %s", label, exc)
        return None

    log.info("[%s] Frequent itemsets: %d", label, len(itemsets))

    if len(itemsets) == 0:
        log.warning("[%s] No frequent itemsets at min_support=%.3f", label, min_support)
        return None

    try:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(itemsets),
        )
    except Exception as exc:
        log.error("[%s] association_rules failed: %s", label, exc)
        return None

    # Apply lift filter
    rules = rules[rules["lift"] >= min_lift].copy()

    if len(rules) == 0:
        log.warning("[%s] No rules survive lift >= %.2f", label, min_lift)
        return None

    # Apply absolute minimum count guard (prevents high-lift / tiny-count rules)
    # Derive support_count here before the final rename pass
    rules["support_count"] = (rules["support"] * n_baskets).round().astype(int)
    before_count_filter = len(rules)
    rules = rules[rules["support_count"] >= min_count].copy()
    if before_count_filter > len(rules):
        log.info("[%s] Dropped %d rules with support_count < %d (high-lift/low-count)",
                 label, before_count_filter - len(rules), min_count)
    if len(rules) == 0:
        log.warning("[%s] No rules survive min_count >= %d", label, min_count)
        return None

    log.info("[%s] Rules after filters: %d", label, len(rules))

    # Convert frozensets to readable strings
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    # Add leverage and conviction (support_count already set above)
    # leverage = support(A∪B) - support(A)*support(B)  [already in mlxtend]
    # conviction = (1 - consequent_support) / (1 - confidence)  [already in mlxtend]
    # mlxtend ≥0.22 returns leverage and conviction natively; guard for older versions
    if "leverage" not in rules.columns:
        rules["leverage"] = (rules["support"]
                             - rules["antecedent support"] * rules["consequent support"])
    if "conviction" not in rules.columns:
        denom = 1 - rules["confidence"]
        rules["conviction"] = ((1 - rules["consequent support"])
                                / denom.replace(0, float("nan")))

    # Round for readability
    for col in ["support", "confidence", "lift", "leverage",
                "antecedent support", "consequent support"]:
        if col in rules.columns:
            rules[col] = rules[col].round(4)
    rules["conviction"] = rules["conviction"].round(3)

    # Sort by lift descending, then confidence descending
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules["rank"] = rules.index + 1
    rules["role_group"] = label
    rules["n_baskets"] = n_baskets
    rules["min_support_used"] = min_support

    # Select and order final columns
    cols = [
        "rank", "role_group", "n_baskets", "min_support_used",
        "antecedents", "consequents",
        "support", "support_count",
        "antecedent support", "consequent support",
        "confidence", "lift", "leverage", "conviction",
    ]
    return rules[[c for c in cols if c in rules.columns]]


# ── Sanity-check helpers ──────────────────────────────────────────────────

def verify_metrics(rules_df: pd.DataFrame, skill_lists: list[list[str]]) -> list[str]:
    """
    Hand-check the top 5 rules by recomputing support, confidence, lift
    from first principles. Returns a list of warning strings (empty = clean).
    """
    warnings = []
    n = len(skill_lists)
    # Build skill→posting set map
    skill_sets: dict[str, set] = defaultdict(set)
    for i, basket in enumerate(skill_lists):
        for s in basket:
            skill_sets[s].add(i)

    for _, row in rules_df.head(5).iterrows():
        ant = frozenset(row["antecedents"].split(", "))
        con = frozenset(row["consequents"].split(", "))
        union = ant | con

        # Postings containing all antecedent skills
        ant_set = set.intersection(*(skill_sets[s] for s in ant)) if ant else set(range(n))
        union_set = set.intersection(*(skill_sets[s] for s in union)) if union else set()
        con_set = set.intersection(*(skill_sets[s] for s in con)) if con else set(range(n))

        sup_calc   = len(union_set) / n
        conf_calc  = len(union_set) / len(ant_set) if ant_set else 0
        lift_calc  = conf_calc / (len(con_set) / n) if con_set else 0

        sup_diff  = abs(sup_calc  - row["support"])
        conf_diff = abs(conf_calc - row["confidence"])
        lift_diff = abs(lift_calc - row["lift"])

        if sup_diff > 0.005:
            warnings.append(f"Row {row['rank']}: support mismatch: "
                            f"computed={sup_calc:.4f} reported={row['support']}")
        if conf_diff > 0.005:
            warnings.append(f"Row {row['rank']}: confidence mismatch: "
                            f"computed={conf_calc:.4f} reported={row['confidence']}")
        if lift_diff > 0.05:
            warnings.append(f"Row {row['rank']}: lift mismatch: "
                            f"computed={lift_calc:.4f} reported={row['lift']}")

    return warnings


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_phase2(
    min_support_overall: float = FP_MIN_SUPPORT_OVERALL,
    min_support_group:   float = FP_MIN_SUPPORT_GROUP,
    min_confidence:      float = FP_MIN_CONFIDENCE,
    min_lift:            float = FP_MIN_LIFT,
) -> dict:
    """
    Full Phase 2 pipeline. Returns a results dict for testing and reporting.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    baskets_df = load_baskets()
    baskets_df["role_group"] = baskets_df["title_lower"].apply(assign_role_group)

    # Convert frozensets to sorted lists for mlxtend
    all_skill_lists = [sorted(s) for s in baskets_df["skills"]]

    results = {}
    summary_rows = []

    # ── Overall run ───────────────────────────────────────────────────────
    rules_overall = mine_rules(
        all_skill_lists,
        min_support=min_support_overall,
        min_confidence=min_confidence,
        min_lift=min_lift,
        label="overall",
    )

    if rules_overall is not None:
        # Metric verification on top 5
        checks = verify_metrics(rules_overall, all_skill_lists)
        if checks:
            for w in checks:
                log.warning("METRIC CHECK: %s", w)
        else:
            log.info("Metric verification passed for top 5 overall rules.")

        out_path = TABLES_DIR / "rules_overall.csv"
        rules_overall.to_csv(out_path, index=False)
        log.info("Written: %s (%d rules)", out_path, len(rules_overall))

        top20_path = TABLES_DIR / "rules_overall_top20.csv"
        rules_overall.head(20).to_csv(top20_path, index=False)
        log.info("Written: %s", top20_path)

        summary_rows.append({
            "run": "overall",
            "n_baskets": len(all_skill_lists),
            "min_support": min_support_overall,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
            "n_rules": len(rules_overall),
            "top_rule": (f"{rules_overall.iloc[0]['antecedents']} → "
                         f"{rules_overall.iloc[0]['consequents']} "
                         f"(lift={rules_overall.iloc[0]['lift']})"),
        })
        results["overall"] = rules_overall
        results["metric_warnings"] = checks
    else:
        log.warning("No overall rules generated.")
        results["overall"] = None
        results["metric_warnings"] = []

    # ── Per role-group runs ───────────────────────────────────────────────
    group_rules = {}
    for group_name in ROLE_GROUPS:
        group_df = baskets_df[baskets_df["role_group"] == group_name]
        skill_lists = [sorted(s) for s in group_df["skills"]]

        log.info("[%s] %d postings, %d with skills",
                 group_name, len(baskets_df[baskets_df["role_group"] == group_name]),
                 len(skill_lists))

        grules = mine_rules(
            skill_lists,
            min_support=min_support_group,
            min_confidence=min_confidence,
            min_lift=min_lift,
            label=group_name,
        )

        if grules is not None:
            out_path = TABLES_DIR / f"rules_{group_name}.csv"
            grules.to_csv(out_path, index=False)
            log.info("Written: %s (%d rules)", out_path, len(grules))
            summary_rows.append({
                "run": group_name,
                "n_baskets": len(skill_lists),
                "min_support": min_support_group,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "n_rules": len(grules),
                "top_rule": (f"{grules.iloc[0]['antecedents']} → "
                             f"{grules.iloc[0]['consequents']} "
                             f"(lift={grules.iloc[0]['lift']})"),
            })
        else:
            summary_rows.append({
                "run": group_name,
                "n_baskets": len(skill_lists),
                "min_support": min_support_group,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "n_rules": 0,
                "top_rule": "none",
            })
        group_rules[group_name] = grules

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / "rules_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Written: %s", summary_path)

    results["group_rules"] = group_rules
    results["summary"] = summary_df
    return results


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_phase2()

    print("\n" + "=" * 70)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 70)
    print(results["summary"].to_string(index=False))

    if results["overall"] is not None:
        df = results["overall"]
        print(f"\nTop 10 overall rules (ranked by lift, {len(df)} total):")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(df[cols].head(10).to_string(index=False))

    if results["metric_warnings"]:
        print("\nMETRIC WARNINGS:")
        for w in results["metric_warnings"]:
            print(" ", w)
    else:
        print("\nMetric verification: PASSED")
