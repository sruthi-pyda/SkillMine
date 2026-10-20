"""
SkillMine — central configuration.
All tunable thresholds, file paths, and role-group patterns live here.
Import this module everywhere instead of scattering magic numbers through scripts.
"""

from pathlib import Path

# ── Project root (one level above src/) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data paths ───────────────────────────────────────────────────────────
DATA_DIR             = PROJECT_ROOT / "data"
PRIMARY_POSTINGS     = DATA_DIR / "linkedin_primary" / "postings.csv"
VALIDATION_POSTINGS  = DATA_DIR / "validation" / "postings_validation.csv"

# ── Phase 1 outputs ───────────────────────────────────────────────────────
OUTPUT_DIR           = PROJECT_ROOT / "output"
TABLES_DIR           = OUTPUT_DIR / "tables"
FIGURES_DIR          = OUTPUT_DIR / "figures"
NETWORK_DIR          = OUTPUT_DIR / "network"

CLEAN_CSV            = OUTPUT_DIR / "skillmine_clean.csv"
TRANSACTIONS_CSV     = OUTPUT_DIR / "skillmine_transactions.csv"

# ── Phase 2 — FP-Growth thresholds ───────────────────────────────────────
# Overall dataset
FP_MIN_SUPPORT_OVERALL   = 0.01   # 1 % of baskets (≈174 postings at 17 393 baskets)
FP_MIN_CONFIDENCE        = 0.50   # minimum rule confidence
FP_MIN_LIFT              = 1.20   # minimum lift to keep a rule

# Per role-group (smaller pools need a higher relative threshold)
FP_MIN_SUPPORT_GROUP     = 0.02   # 2 % within each role group
FP_MIN_CONFIDENCE_GROUP  = 0.50
FP_MIN_LIFT_GROUP        = 1.20

# Absolute basket floor: skip groups with fewer than this many skill-bearing postings
FP_GROUP_MIN_BASKETS     = 150

# Minimum absolute count: rules whose support_count is below this are dropped
# to prevent high-lift rules based on tiny cell counts from dominating.
# Applies to both overall and group runs.
FP_MIN_COUNT             = 20

# ── Role-group definitions ────────────────────────────────────────────────
# Patterns match against the lower-cased job title.
# A posting is assigned to the FIRST group whose pattern matches; unmatched → None.
ROLE_GROUPS = {
    "data_analyst":      r"data anal|business anal|research anal|marketing anal"
                         r"|financial anal|operations anal|bi anal|reporting anal",
    "software_engineer": r"software eng|software dev|\bswe\b|backend|back-end"
                         r"|back end|fullstack|full.stack|full-stack|full stack",
    # "infrastructure" alone was too broad (matched "infrastructure analyst",
    # "infrastructure project manager", "hardware infrastructure engineer").
    # Narrowed to require an engineering or architecture suffix, which keeps
    # legitimate DevOps/infra engineers while excluding non-DevOps roles.
    "cloud_devops":      r"cloud eng|devops|\bsre\b|platform eng"
                         r"|infrastructure eng|infrastructure arch"
                         r"|site reliability|solutions arch",
    "data_scientist_ml": r"data scien|machine learning eng|\bml eng|\bmlops"
                         r"|ai eng|applied scien|research scien",
    "data_engineer":     r"data eng|data architect|\betl\b|pipeline eng"
                         r"|analytics eng",
}

# ── Phase 3 — Skill co-occurrence network ────────────────────────────────
# Minimum co-occurrence count to create an edge.
# Pairs with fewer joint postings produce unreliable PMI estimates.
NET_MIN_CO_COUNT         = 20

# Only keep edges with strictly positive PMI (skills that co-occur more than
# expected by chance).  Edges with PMI ≤ 0 are discarded.
NET_MIN_PMI              = 0.0

# Minimum skill frequency to include a node (used for viz filtering only;
# the underlying graph retains all canonical skills).
NET_VIZ_MIN_FREQ         = 50

# Minimum ppmi_norm to include an edge in the standard viz HTML (0.05 keeps
# 829/908 edges after Phase 5 re-run — still moderately dense).
NET_VIZ_MIN_EDGE_WEIGHT  = 0.05

# Higher threshold for a presentation-quality version: cleaner and less
# cluttered (≈470/908 edges).  Both HTMLs are exported; the analytical
# CSVs always contain all edges with co_count ≥ NET_MIN_CO_COUNT and PMI > 0.
NET_VIZ_PRES_EDGE_WEIGHT = 0.20

# Louvain resolution parameter: 1.0 is the standard default.
# Increase > 1.0 for more/smaller communities; decrease < 1.0 for fewer/larger.
# Tested range 1.0–2.5 on actual data:
#   1.0 → 3 communities (too coarse: 22–27 skills each, mixes BI+data-eng, software+cloud)
#   1.5 → 6 communities (chosen: clean semantic separation into recognisable skill families)
#   2.0 → 13 communities (too granular: singletons appear)
# Resolution=1.5 selected for legibility and interpretability.
NET_LOUVAIN_RESOLUTION   = 1.5

# Minimum community size to include in summary output (tiny singletons omitted).
NET_MIN_COMMUNITY_SIZE   = 2

# Number of top job titles to show per community summary
NET_TITLES_PER_COMMUNITY = 10

# ── Phase 3 output paths ─────────────────────────────────────────────────
NODES_CSV      = OUTPUT_DIR / "network" / "skill_nodes.csv"
EDGES_CSV      = OUTPUT_DIR / "network" / "skill_edges.csv"
COMMUNITIES_CSV= OUTPUT_DIR / "network" / "skill_communities.csv"
NETWORK_HTML   = OUTPUT_DIR / "network" / "skill_network.html"

# ── Phase 4 — K-Means posting clusters ───────────────────────────────────
CLUSTERS_DIR             = OUTPUT_DIR / "clusters"
CLUSTER_ASSIGNMENTS_CSV  = CLUSTERS_DIR / "posting_clusters.csv"
CLUSTER_SUMMARIES_CSV    = CLUSTERS_DIR / "cluster_summaries.csv"
CLUSTER_SILHOUETTE_CSV   = CLUSTERS_DIR / "silhouette_scores.csv"

# Range of k values to evaluate; silhouette guides final choice
KMEANS_K_MIN    = 6
KMEANS_K_MAX    = 12
KMEANS_N_INIT   = 10    # KMeans restarts per k
KMEANS_MAX_ITER = 300

# Minimum cluster size to include in summary (tiny clusters omitted)
KMEANS_MIN_CLUSTER_SIZE  = 50

# Number of top skills / titles to show per cluster summary
KMEANS_TOP_SKILLS        = 10
KMEANS_TOP_TITLES        = 8

# ── Reproducibility ───────────────────────────────────────────────────────
RANDOM_SEED = 42
