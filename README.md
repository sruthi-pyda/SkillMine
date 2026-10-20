# SkillMine

**Mining Skill Associations and Emerging Career Trends from Job Postings**

VIT Data Mining — BCSE208P · October 2026
Team: Rampudi Jaiwanth Ram, Pyda Venkata Satya Sruthi, Hari Shree
Guide: Maheshwari B

---

## Project overview

SkillMine treats every job posting as a basket of skills and answers four questions:

1. **Which skills go together?** — FP-Growth association rules ranked by lift
2. **Which skill families form roles?** — Skill co-occurrence network (Louvain) + K-Means clustering
3. **Which skills are emerging?** — Monthly share trends *(blocked: single-month dataset — see Known Limitations)*
4. **Which current openings fit me?** — Live job board with skill-match scoring

---

## Setup

### 1. Data

Download both datasets from Kaggle and place them as shown:

| Dataset | Kaggle slug | Place at |
|---------|-------------|----------|
| LinkedIn Job Postings 2023–24 | `arshkon/linkedin-job-postings` | `data/linkedin_primary/postings.csv` (+ `jobs/`, `companies/`, `mappings/` subfolders) |
| Data Analyst Job Postings | `asaniczka/data-analyst-job-postings-...` | `data/validation/postings_validation.csv` |

### 2. R environment (Phase 1 — preprocessing & EDA)

```bash
# Required R packages (install once)
Rscript -e "install.packages(c('tidyverse','lubridate','janitor','skimr','stringr','scales'), repos='https://cran.r-project.org')"

# Run pipeline from project root
Rscript R/skillmine_review1.R
```

Outputs written to `output/`:
- `skillmine_clean.csv` — 22,372 cleaned tech/data postings
- `skillmine_transactions.csv` — long (job_id, skill) transaction table
- `before_after_stats.csv` — actual before/after cleaning statistics
- `missing_before.csv` — missing-value profile of raw data
- `figures/fig1–fig9` — EDA charts (PNG, 300 dpi)

### 3. Python environment (Phases 2–7)

```bash
python3 -m src.skillmine.association_rules   # Phase 2: FP-Growth
```

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Environment variables (Phase 7 — live job board)

```bash
cp .env.example .env
# Edit .env and fill in API keys
```

---

## Repository structure

```
data/          Raw datasets (not committed — download per Setup above)
R/             R preprocessing & EDA scripts
src/skillmine/ Python mining pipeline (Phases 2–7)
output/        Generated artifacts (not committed — reproducible from scripts)
tests/         Unit tests
```

---

## Known limitations

- **Single-month dataset:** 123,848 of 123,849 postings are from April 2024 (date range 2024-04-05 to 2024-04-20). Monthly trend analysis is not feasible on this snapshot. Phase 5 (emerging skills) is blocked until a multi-month source is obtained.
- **LinkedIn / US bias:** Data is US-heavy and from LinkedIn only.
- **Small skill dictionary:** Current extraction uses 38 seed skills. Mean 2.35 skills/posting; ~27% of postings match zero skills. Dictionary expansion is Phase 1 work in progress.
- **Salary data sparse:** 15,478 of 22,372 tech/data postings have no usable salary.

---

## Actual pipeline statistics (from last run)

| Metric | Value |
|--------|-------|
| Raw rows | 123,849 |
| Raw columns | 31 |
| Duplicate job IDs removed | 0 |
| Duplicate title+company+desc removed | 12,659 |
| Missing descriptions removed | 7 |
| Tech/data postings retained | 22,372 |
| Date range | 2024-04-05 to 2024-04-20 |
| Skills in dictionary | 38 |
| Total (job_id, skill) rows | 52,555 |
| Mean skills per posting | 2.35 |
| Postings with 0 skills | ~6,100 (est.) |
| Median annual salary | ~$114,400 |
