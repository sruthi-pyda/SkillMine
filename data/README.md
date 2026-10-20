# Data

## Primary dataset — LinkedIn Job Postings 2023–24
- **Source:** Kaggle · arshkon (`kaggle.com/datasets/arshkon/linkedin-job-postings`)
- **Place at:** `data/linkedin_primary/postings.csv` (and companion files under `jobs/`, `companies/`, `mappings/`)
- **Size:** ~493 MB, 123,849 rows

## Validation dataset — Data Analyst Job Postings
- **Source:** Kaggle · asaniczka
- **Place at:** `data/validation/postings_validation.csv`
- **Size:** ~57 MB, 492,220 rows

## Known data limitations
- Primary dataset is almost entirely April 2024 (123,848 of 123,849 rows).
  Monthly trend analysis is therefore **not feasible** on this snapshot alone.
- Data is US-heavy and LinkedIn-only.
- `skills_desc` column is >90% missing; skills are extracted from `description`.
