"""
SkillMine dashboard pages (Phase 6).  Navigation lives in streamlit_app.py.

All pages read precomputed artifacts from output/ through
src/skillmine/dashboard_data.py.  No mining algorithm is re-run here.
"""

import sys
from functools import wraps
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.skillmine import dashboard_data as dd
from src.skillmine.config import (
    CLEAN_CSV, TRANSACTIONS_CSV, TABLES_DIR, BEFORE_AFTER_CSV, RULES_SUMMARY_CSV,
    NODES_CSV, EDGES_CSV, COMMUNITIES_CSV, NETWORK_HTML, NETWORK_PRES_HTML,
    CLUSTER_ASSIGNMENTS_CSV, CLUSTER_SUMMARIES_CSV, CLUSTER_SILHOUETTE_CSV,
    FP_MIN_COUNT, NET_MIN_CO_COUNT, NET_VIZ_MIN_FREQ, NET_VIZ_MIN_EDGE_WEIGHT,
    NET_VIZ_PRES_EDGE_WEIGHT, DASH_SMALL_GROUP_BASKETS, DASH_LARGE_RULESET,
    ROLE_GROUPS,
)

ROLE_NAMES = {
    "overall":           "All roles",
    "data_analyst":      "Data analyst",
    "software_engineer": "Software engineer",
    "cloud_devops":      "Cloud / DevOps",
    "data_scientist_ml": "Data scientist / ML",
    "data_engineer":     "Data engineer",
}

# Cluster labels like "C2 · Web / Frontend Engineering" are long; Vega-Lite
# truncates axis labels at 180px by default.
LABEL_AXIS = alt.Axis(labelLimit=320)

SNAPSHOT_NOTE = ("Historical snapshot: LinkedIn postings listed "
                 "{start:%d %b %Y} – {end:%d %b %Y}. Not live data.")


# ── Cached loading ────────────────────────────────────────────────────────
# The file signature (mtimes) is part of the cache key, so re-running a
# pipeline step invalidates the cached copy automatically.

@st.cache_data(show_spinner="Loading outputs…")
def _load(name: str, signature: tuple, *args):
    return getattr(dd, name)(*args)


def load(name: str, *args, paths: tuple = ()):
    return _load(name, dd.file_signature(*paths), *args)


def clean():         return load("load_clean", paths=(CLEAN_CSV,))
def txn():           return load("load_transactions", paths=(TRANSACTIONS_CSV,))
def nodes():         return load("load_nodes", paths=(NODES_CSV,))
def edges():         return load("load_edges", paths=(EDGES_CSV,))
def communities():   return load("load_communities", paths=(COMMUNITIES_CSV,))
def assignments():   return load("load_cluster_assignments", paths=(CLUSTER_ASSIGNMENTS_CSV,))
def summaries():     return load("load_cluster_summaries", paths=(CLUSTER_SUMMARIES_CSV,))
def silhouette():    return load("load_silhouette", paths=(CLUSTER_SILHOUETTE_CSV,))
def rules_summary(): return load("load_rules_summary", paths=(RULES_SUMMARY_CSV,))


def rules(role: str):
    return load("load_rules", role, paths=(TABLES_DIR / f"rules_{role}.csv",))


@st.cache_data(show_spinner=False)
def _skill_freq(signature: tuple):
    return dd.skill_frequencies(txn(), len(clean()))


def skill_freq():
    return _skill_freq(dd.file_signature(CLEAN_CSV, TRANSACTIONS_CSV))


@st.cache_data(show_spinner=False)
def _cluster_shares(signature: tuple):
    return dd.cluster_skill_shares(txn(), assignments())


def cluster_shares():
    return _cluster_shares(dd.file_signature(TRANSACTIONS_CSV, CLUSTER_ASSIGNMENTS_CSV))


@st.cache_data(show_spinner=False)
def _network_html(signature: tuple, presentation: bool):
    return dd.load_network_html(presentation)


def network_html(presentation: bool):
    return _network_html(dd.file_signature(NETWORK_HTML, NETWORK_PRES_HTML), presentation)


def snapshot_caption():
    c = clean()
    dates = pd.to_datetime(c["listed_date"], errors="coerce")
    st.caption(SNAPSHOT_NOTE.format(start=dates.min(), end=dates.max()))


def guarded(page):
    """Turn missing-artifact errors into instructions instead of tracebacks."""
    @wraps(page)
    def wrapper():
        try:
            page()
        except dd.MissingArtifact as e:
            st.error(f"**Required output not found:** `{e.path}`")
            st.markdown("Generate it from the project root with:")
            st.code(e.command, language="bash")
            st.caption("See README.md → Setup for the full pipeline order "
                       "(R preprocessing → association rules → network → clusters).")
    return wrapper


# ── Pages ─────────────────────────────────────────────────────────────────

@guarded
def overview_page():
    st.title("SkillMine")
    st.markdown("**Mining skill associations from job postings.** Each posting is "
                "treated as a *basket* of skills extracted from its description.")

    s = dd.overview_stats(clean(), txn(), load("load_before_after", paths=(BEFORE_AFTER_CSV,)))

    c = st.columns(5)
    c[0].metric("Tech/data postings", f"{s['postings']:,}",
                help=f"Filtered from {s['raw_rows']:,} raw LinkedIn postings")
    c[1].metric("Skills in dictionary", s["n_skills"])
    c[2].metric("Posting–skill pairs", f"{s['n_pairs']:,}")
    c[3].metric("Mean skills / posting", f"{s['mean_skills']:.2f}",
                help="Averaged over all filtered postings, including zero-skill ones")
    c[4].metric("No skill found", f"{s['zero_skill_postings']:,}",
                help="Excluded from rules, network and clusters — "
                     f"those use the {s['skilled_postings']:,} postings with ≥1 skill")

    st.info(f"**Time coverage:** {s['date_min']:%d %b %Y} – {s['date_max']:%d %b %Y} "
            f"({s['n_months']} calendar month). This is a single snapshot, so the "
            "dashboard does not show skill trends over time.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Most common skills")
        top = skill_freq().head(20)
        chart = (alt.Chart(top).mark_bar()
                 .encode(x=alt.X("share_all:Q", title="Share of filtered postings",
                                 axis=alt.Axis(format="%")),
                         y=alt.Y("skill:N", sort="-x", title=None),
                         tooltip=["skill", alt.Tooltip("postings:Q", format=","),
                                  alt.Tooltip("share_all:Q", format=".1%", title="share")])
                 .properties(height=480))
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Pipeline")
        st.graphviz_chart("""
            digraph {
              rankdir=TB; node [shape=box, style="rounded", fontsize=11];
              raw [label="Raw LinkedIn postings"]; clean [label="Clean + filter (R)"];
              ext [label="Dictionary skill extraction"]; tx [label="Posting × skill baskets"];
              ar [label="FP-Growth rules"]; net [label="PMI network + Louvain"];
              km [label="TF-IDF + K-Means"]; dash [label="This dashboard"];
              raw -> clean -> ext -> tx; tx -> ar; tx -> net; tx -> km;
              ar -> dash; net -> dash; km -> dash;
            }""")
        st.subheader("Cleaning")
        st.markdown(f"""
| Step | Count |
|---|---:|
| Raw rows | {s['raw_rows']:,} |
| Duplicate title + company + description removed | {s['duplicate_text']:,} |
| Retained tech/data postings | {s['postings']:,} |
| … with ≥ 1 extracted skill | {s['skilled_postings']:,} |
""")

    st.subheader("Known limitations")
    st.markdown(f"""
- **Single time window.** All postings fall within {s['date_min']:%d %b} – {s['date_max']:%d %b %Y}; emerging-skill trends cannot be established (see *Emerging Skills*).
- **LinkedIn, US-heavy.** Results describe this source, not the whole job market.
- **Small skill dictionary** ({s['n_skills']} skills, regex-matched). {s['zero_skill_postings']:,} postings matched no skill. Short terms (e.g. *r*, *go*) can produce false positives.
- **Extractor precision not yet human-validated.** A 100-posting annotation sheet exists (`output/validation/skill_extraction_review.csv`) but has not been completed.
- **Broad title filter.** The tech/data title filter keeps some non-tech roles (e.g. building engineer, financial analyst), which appear in the clusters.
- **Associations are not causation**, and clusters are exploratory groupings, not occupational categories.
""")


@guarded
def rules_page():
    st.title("Association rules")
    st.markdown("Rules of the form **A → B**: postings that mention all skills in *A* "
                "also tend to mention all skills in *B*. Mined with FP-Growth, ranked by lift.")

    summ = rules_summary().set_index("run")
    roles = [r for r in ROLE_NAMES if r in summ.index]
    role = st.selectbox("Role group", roles,
                        format_func=lambda r: f"{ROLE_NAMES[r]}  ({int(summ.loc[r, 'n_baskets']):,} postings)")
    meta = summ.loc[role]
    df = rules(role)

    st.caption(f"Mined with min support {meta['min_support']:.0%}, min confidence "
               f"{meta['min_confidence']:.0%}, min lift {meta['min_lift']}, and at least "
               f"{FP_MIN_COUNT} supporting postings per rule.")
    if role != "overall":
        st.caption(f"Role group = first matching title pattern: `{ROLE_GROUPS[role]}`. "
                   "Postings matching no pattern appear only under *All roles*.")
    if meta["n_baskets"] < DASH_SMALL_GROUP_BASKETS:
        st.warning(f"**Small sample:** only {int(meta['n_baskets']):,} postings in this group. "
                   f"A {meta['min_support']:.0%} support threshold means a rule can rest on "
                   f"~{max(FP_MIN_COUNT, round(meta['min_support'] * meta['n_baskets']))} postings — "
                   "treat individual rules as unstable.")
    if len(df) > DASH_LARGE_RULESET:
        st.warning(f"**{len(df):,} rules.** Many are re-arrangements of the same few frequent "
                   "skill sets. Use *Max skills per rule* and the skill filter to keep it readable.")

    with st.expander("What do support, confidence and lift mean?"):
        st.markdown("""
- **Support** — share of postings in the group containing *every* skill in A and B. `Count` is the same thing as a number of postings.
- **Confidence** — of postings containing A, the share that also contain B. An estimate of P(B | A).
- **Lift** — confidence ÷ the overall share of postings containing B. Lift 1 = no association; lift 3 = B is three times more common when A is present. Lift is a ratio, **not a probability**.
- High lift built on a tiny count is fragile — always read lift alongside count.
- A rule shows co-occurrence in postings. It does not mean learning A causes employers to want B.
""")

    f1, f2, f3 = st.columns(3)
    min_sup = f1.slider("Min support", float(df["support"].min()), float(df["support"].max()),
                        float(df["support"].min()), step=0.001, format="%.3f")
    min_conf = f2.slider("Min confidence", 0.5, 1.0, 0.5, step=0.01)
    min_lift = f3.slider("Min lift", 1.0, float(max(df["lift"].max(), 1.0)), 1.0, step=0.1)

    g1, g2, g3, g4 = st.columns([3, 2, 2, 2])
    all_skills = sorted(set().union(*df["antecedents"].map(dd.parse_itemset),
                                    *df["consequents"].map(dd.parse_itemset)))
    skills = g1.multiselect("Rules containing skill(s)", all_skills)
    side = g2.radio("Where", ["either", "antecedent", "consequent"], horizontal=False,
                    format_func={"either": "Either side", "antecedent": "In A (if…)",
                                 "consequent": "In B (then…)"}.get)
    max_items_cap = int((df["n_antecedents"] + df["n_consequents"]).max())
    max_items = g3.slider("Max skills per rule", 2, max(max_items_cap, 2), min(3, max_items_cap) if max_items_cap >= 2 else 2)
    sort_by = g4.selectbox("Sort by", ["lift", "confidence", "support"])

    out = dd.filter_rules(df, min_sup, min_conf, min_lift, skills, side, max_items)
    out = out.sort_values([sort_by, "support"], ascending=False)

    st.markdown(f"**{len(out):,}** of {len(df):,} rules match.")
    if out.empty:
        st.info("No rules match these filters. Lower the thresholds, raise *Max skills per rule*, "
                "or remove a skill filter.")
        return
    st.dataframe(
        out[["antecedents", "consequents", "support", "support_count", "confidence", "lift"]].head(1000),
        hide_index=True, width="stretch",
        column_config={
            "antecedents":   st.column_config.TextColumn("If posting mentions (A)"),
            "consequents":   st.column_config.TextColumn("…it also tends to mention (B)"),
            "support":       st.column_config.NumberColumn("Support", format="%.3f"),
            "support_count": st.column_config.NumberColumn("Count", help="Postings containing A and B"),
            "confidence":    st.column_config.NumberColumn("Confidence", format="%.2f"),
            "lift":          st.column_config.NumberColumn("Lift", format="%.2f"),
        })
    if len(out) > 1000:
        st.caption("Showing the first 1,000 rows in the chosen sort order.")


@guarded
def network_page():
    st.title("Skill network")
    st.markdown("Each node is a skill; an edge links two skills that appear together in "
                "postings **more often than chance**. Colours are Louvain communities "
                "(skill families). Drag to explore; hover for details.")

    view = st.radio("View", ["presentation", "analytical"], horizontal=True,
                    format_func={"presentation": f"Presentation (strong links, PPMI ≥ {NET_VIZ_PRES_EDGE_WEIGHT})",
                                 "analytical": f"Analytical (PPMI ≥ {NET_VIZ_MIN_EDGE_WEIGHT})"}.get)
    st.iframe(network_html(view == "presentation"), height=780)

    with st.expander("How are edges weighted and filtered?"):
        st.markdown(f"""
- **Co-occurrences** — raw number of postings that mention both skills.
- **PMI** = log₂( P(a,b) / (P(a)·P(b)) ) over postings with ≥ 1 skill. Positive PMI means the pair appears together more often than if the skills were independent. PMI corrects for popularity: two very common skills can co-occur often yet have low PMI.
- **PPMI (norm)** — PMI with negatives set to 0, divided by the largest PMI in the graph, so it lies in 0–1. This is the edge weight.
- Edges need ≥ {NET_MIN_CO_COUNT} co-occurrences and PMI > 0. For readability the graph also hides skills in fewer than {NET_VIZ_MIN_FREQ} postings and edges below the PPMI threshold of the chosen view. The full edge list is used in the explorer below.
- Communities group skills that are densely linked; their names are suggestions derived from members and titles.
""")

    st.subheader("Skill families (Louvain communities)")
    comm = communities()
    st.dataframe(comm[["community_id", "suggested_label", "size", "member_skills", "top_job_titles"]],
                 hide_index=True, width="stretch",
                 column_config={"community_id": "ID", "suggested_label": "Suggested label",
                                "size": "Skills", "member_skills": "Member skills",
                                "top_job_titles": "Common job titles"})

    st.subheader("Strongest connections of a skill")
    n = nodes().sort_values("frequency", ascending=False)
    skill = st.selectbox("Skill", n["skill"].tolist())
    st.dataframe(dd.skill_neighbours(edges(), skill, top=15), hide_index=True,
                 width="stretch",
                 column_config={"partner": "Connected skill",
                                "co_count": st.column_config.NumberColumn("Co-occurrences"),
                                "pmi": st.column_config.NumberColumn("PMI", format="%.2f"),
                                "ppmi_norm": st.column_config.ProgressColumn("PPMI (norm)", min_value=0, max_value=1, format="%.2f")})


@guarded
def clusters_page():
    st.title("Posting clusters")
    st.markdown("Postings grouped by K-Means on TF-IDF vectors of their extracted skills. "
                "Each cluster is a candidate *role archetype*.")
    st.warning("Clusters are exploratory. Labels are suggestions derived from top skills, "
               "not ground-truth job categories, and the silhouette scores below indicate "
               "**weak separation** — clusters overlap substantially.")

    a, summ, sil = assignments(), summaries(), silhouette()
    k = a["cluster_id"].nunique()
    k_row = sil[sil["k"] == k]
    k_score = float(k_row["silhouette_score"].iloc[0]) if not k_row.empty else None
    best = sil.loc[sil["silhouette_score"].idxmax()]

    c = st.columns(4)
    c[0].metric("Clusters (k)", k)
    c[1].metric("Silhouette at chosen k", f"{k_score:.3f}" if k_score is not None else "n/a",
                help="Range −1 to 1; values near 0.2 mean weak structure")
    c[2].metric("Postings clustered", f"{len(a):,}")
    c[3].metric("Excluded (no skills)", f"{len(clean()) - len(a):,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Why k = %d?" % k)
        sil_chart = alt.Chart(sil).mark_line(point=True).encode(
            x=alt.X("k:O", title="k", axis=alt.Axis(labelAngle=0)), y=alt.Y("silhouette_score:Q", title="Silhouette",
                                                scale=alt.Scale(zero=False)),
            tooltip=["k", alt.Tooltip("silhouette_score:Q", format=".3f")])
        chosen = alt.Chart(sil[sil["k"] == k]).mark_point(size=180, filled=True, color="#E8512B").encode(
            x="k:O", y="silhouette_score:Q")
        st.altair_chart(sil_chart + chosen, width="stretch")
        st.caption(f"Scores rise steadily up to k = {int(best['k'])} ({best['silhouette_score']:.3f}). "
                   "The pipeline picks the smallest k within 0.05 of the best score (parsimony), "
                   "then steps down if any cluster has < 50 postings. The 0.05 margin is a design "
                   "choice, not a statistical test.")
    with right:
        st.subheader("Cluster sizes")
        st.altair_chart(alt.Chart(summ).mark_bar().encode(
            x=alt.X("size:Q", title="Postings"), y=alt.Y("display_label:N", sort="-x", title=None, axis=LABEL_AXIS),
            tooltip=["display_label", "size"]), width="stretch")

    st.subheader("Cluster summaries")
    st.dataframe(summ[["display_label", "size", "top_skills", "rep_titles"]], hide_index=True,
                 width="stretch",
                 column_config={"display_label": "Cluster", "size": "Postings",
                                "top_skills": "Top TF-IDF skills", "rep_titles": "Common titles"})

    st.subheader("Compare clusters")
    shares = cluster_shares().merge(summ[["cluster_id", "display_label"]], on="cluster_id")
    top_n = st.slider("Skills shown", 5, 30, 15)
    top_skills = skill_freq().head(top_n)["skill"].tolist()
    heat = shares[shares["skill"].isin(top_skills)]
    st.altair_chart(alt.Chart(heat).mark_rect().encode(
        x=alt.X("skill:N", sort=top_skills, title=None),
        y=alt.Y("display_label:N", title=None, axis=LABEL_AXIS),
        color=alt.Color("share:Q", title="Share of cluster", scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(format="%")),
        tooltip=["display_label", "skill", alt.Tooltip("share:Q", format=".1%"), "postings"],
    ).properties(height=320), width="stretch")
    st.caption("Cell = share of the cluster's postings that mention the skill. "
               "Skills ordered by overall frequency.")

    st.subheader("Inside a cluster")
    labels = dict(zip(summ["cluster_id"].astype(int), summ["display_label"]))
    cid = st.selectbox("Cluster", list(labels), format_func=lambda c: labels.get(int(c), f"C{c}"))
    titles = (a[a["cluster_id"] == cid]["title"].str.strip().str.lower()
              .value_counts().head(15).rename_axis("title").reset_index(name="postings"))
    st.dataframe(titles, hide_index=True, width="stretch")


@guarded
def explorer_page():
    st.title("Skill explorer")
    snapshot_caption()
    freq = skill_freq()
    skill = st.selectbox("Choose a skill", freq["skill"].tolist())
    row = freq[freq["skill"] == skill].iloc[0]
    node = nodes().set_index("skill")
    comm = communities().set_index("community_id")

    c = st.columns(3)
    c[0].metric("Postings mentioning it", f"{int(row['postings']):,}")
    c[1].metric("Share of all postings", f"{row['share_all']:.1%}",
                help=f"Out of {len(clean()):,} filtered postings")
    c[2].metric("Share of skill-bearing postings", f"{row['share_skilled']:.1%}",
                help="Denominator used for rule support and network PMI")
    # Family labels are too long for a metric tile, so they get their own line.
    if skill in node.index:
        cid = int(node.loc[skill, "community_id"])
        label = comm.loc[cid, "suggested_label"] if cid in comm.index else "—"
        st.markdown(f"**Skill family:** {label} *(Louvain community {cid})*")
    else:
        st.markdown("**Skill family:** — *(not in the network)*")

    st.subheader("Association rules")
    summ = rules_summary().set_index("run")
    role = st.selectbox("Role group", [r for r in ROLE_NAMES if r in summ.index],
                        format_func=ROLE_NAMES.get, key="explorer_role")
    sr = dd.skill_rules(rules(role), skill, top=10)
    cols = ["rule", "support", "support_count", "confidence", "lift"]
    cfg = {"rule": "Rule", "support": st.column_config.NumberColumn("Support", format="%.3f"),
           "support_count": "Count",
           "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
           "lift": st.column_config.NumberColumn("Lift", format="%.2f")}
    # Stacked rather than side by side: rule text is long and half-width tables
    # pushed confidence/lift out of view.
    st.markdown(f"**Postings with *{skill}* also tend to mention…**")
    if sr["as_antecedent"].empty:
        st.caption("No rule with this skill on the *if* side passes the mining thresholds.")
    else:
        st.dataframe(sr["as_antecedent"][cols], hide_index=True, width="stretch", column_config=cfg)
    st.markdown(f"**…skills that predict *{skill}***")
    if sr["as_consequent"].empty:
        st.caption("No rule with this skill on the *then* side passes the mining thresholds.")
    else:
        st.dataframe(sr["as_consequent"][cols], hide_index=True, width="stretch", column_config=cfg)

    left, right = st.columns(2)
    with left:
        st.subheader("Network connections")
        nb = dd.skill_neighbours(edges(), skill, top=10)
        if nb.empty:
            st.caption(f"No partner reaches {NET_MIN_CO_COUNT}+ co-occurrences with positive PMI.")
        else:
            st.dataframe(nb, hide_index=True, width="stretch",
                         column_config={"partner": "Connected skill", "co_count": "Co-occurrences",
                                        "pmi": st.column_config.NumberColumn("PMI", format="%.2f"),
                                        "ppmi_norm": st.column_config.ProgressColumn("PPMI (norm)", min_value=0, max_value=1, format="%.2f")})
    with right:
        st.subheader("Across posting clusters")
        dist = dd.skill_cluster_distribution(cluster_shares(), summaries(), skill)
        if dist.empty:
            st.caption("This skill does not appear in any clustered posting.")
        else:
            st.altair_chart(alt.Chart(dist).mark_bar().encode(
                x=alt.X("share:Q", title="Share of cluster's postings mentioning it",
                        axis=alt.Axis(format="%")),
                y=alt.Y("display_label:N", sort="-x", title=None, axis=LABEL_AXIS),
                tooltip=["display_label", "postings", alt.Tooltip("share:Q", format=".1%")]),
                width="stretch")

    st.subheader("Example postings")
    snapshot_caption()
    st.dataframe(dd.postings_with_skill(clean(), txn(), skill), hide_index=True,
                 width="stretch")


@guarded
def emerging_page():
    st.title("Emerging skills")
    status = dd.trend_status(clean())

    if not status["available"]:
        st.error("**Trend analysis unavailable — insufficient time coverage.**  \n" + status["reason"])
        st.markdown(f"""
An *emerging* skill is one whose share of postings rises over time from a low or medium base.
Measuring that needs postings sampled consistently across many months, so a slope can be fitted to
each skill's monthly share. The current data cannot support this:

- Primary dataset (LinkedIn): {status['date_min']:%d %b %Y} – {status['date_max']:%d %b %Y}.
- Validation dataset: a single-day snapshot (20 Dec 2023), collected differently — comparing the two
  would mix source differences with real change.

**What would enable it:** a legitimate multi-month (ideally multi-year) posting source with consistent
collection, run through the same skill extractor. Until then, SkillMine shows **no growth scores or
rankings** rather than ones the data cannot justify.
""")
    else:
        st.info(f"{status['n_months']} months of data detected, but the trend module "
                "(monthly share + linear slope) has not been implemented yet.")

    st.subheader("Current snapshot only")
    st.caption("How common each skill is **within this one window**. "
               "This is frequency, not growth — it says nothing about which skills are rising.")
    freq = skill_freq()
    st.dataframe(freq[["skill", "postings", "share_all"]], hide_index=True, width="stretch",
                 column_config={"skill": "Skill", "postings": "Postings",
                                "share_all": st.column_config.ProgressColumn(
                                    "Share of postings", min_value=0,
                                    max_value=float(freq["share_all"].max()), format="%.3f")})


