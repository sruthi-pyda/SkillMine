# SkillMine — Master Implementation Plan for Claude Code

> **Purpose:** This file is the implementation contract for completing SkillMine. Read it before changing code. Treat the project-review PDF as the high-level product specification and the Report 1 document as the detailed record of the current implementation, code, outputs, limitations, and next steps. The existing repository and real outputs are the source of truth for what is already implemented.

## 1. Project goal

**SkillMine: Mining Skill Associations and Emerging Career Trends from Job Postings**

SkillMine analyses job postings as baskets of skills. It should help users understand:

1. **Which skills go together?** Mine association rules with FP-Growth and rank useful rules by lift.
2. **Which skill families form roles?** Build a skill co-occurrence network and detect communities with Louvain; cluster postings with K-Means to identify role archetypes.
3. **Which skills are emerging?** Calculate monthly posting share per skill and a growth score based on a linear trend, identifying skills rising from a low or medium base share.
4. **Which current openings fit a user?** Provide a job board with recent postings, user-entered skills, match scores, matched/missing skills, filters, and links to the original employer posting.

The final system is a usable website/dashboard, not just notebooks or static plots.

## 2. Required final deliverables

Do not consider the project complete until the applicable items below exist, run, and are checked.

- [ ] Reproducible data cleaning and skill-extraction pipeline.
- [ ] Clean posting-level dataset and posting × skill transaction representation.
- [ ] Skill-extractor evaluation on approximately 100 manually checked postings, with precision reported and examples documented.
- [ ] FP-Growth association-rule tables, ranked by lift, overall and by role group.
- [ ] Skill co-occurrence network with edge weights based on co-occurrence/PMI and Louvain communities.
- [ ] K-Means posting clusters using TF-IDF representations, with selection of `k` guided by silhouette score (investigate a range such as 6–12 if data supports it), cluster descriptions, and job-title-based interpretation.
- [ ] Emerging-skills table/chart: top 15 skills, monthly share, growth score/slope, and base share.
- [ ] Skill explorer: user selects/types a skill and sees partner skills, its community/cluster context, and trend.
- [ ] Interactive skill-network graph coloured by skill family.
- [ ] Dashboard/website integrating the analysis outputs.
- [ ] Live job board: postings within the most recent 5–6 hours, hourly refresh, user skill input, match percentage, matched and missing skills, role/location/remote/experience filters, and an Apply link to the original posting.
- [ ] Final evaluation, documented limitations, setup/run instructions, and report/review-ready outputs.

## 3. Non-negotiable implementation rules

1. **Audit before coding.** Inspect the repository, current branch/status, existing files, notebooks, scripts, outputs, dependencies, and README. Do not assume this is a blank project.
2. **Preserve working work.** Reuse and improve existing implementation. Do not overwrite working scripts or regenerate existing outputs without understanding them.
3. **Follow the phases in order.** Complete and verify a phase before beginning the next. Do not start a dependent phase while its prerequisite data contract or tests are failing.
4. **No fake results.** Never fabricate job postings, dates, metrics, model scores, or successful API refreshes to make the dashboard look populated. Clearly label any demo/sample data and keep it separate from real results.
5. **Investigate data contradictions.** Report 1 states that all 22,372 retained tech/data postings fall in April 2024, so emerging-skill trends cannot be inferred from the current subset. Verify this against the actual data and code. Do not claim a trend from a single month. Find and use a legitimate multi-month source/refresh of the public dataset if available; otherwise show an honest “insufficient time coverage” state and document the limitation.
6. **Do not scrape LinkedIn pages.** Use only permitted public APIs and employer career-board feeds. Respect provider terms, rate limits, and attribution requirements.
7. **Keep secrets out of Git.** API keys belong in environment variables or a local `.env` excluded by `.gitignore`. Commit only `.env.example` with placeholder names.
8. **Avoid unnecessary rewrites.** Prefer small, reviewable changes, sensible modules, and tests. Do not change the whole stack without explaining why and obtaining approval if the change is material.
9. **Report actual status.** Distinguish implemented, partially implemented, blocked, and not implemented. Never mark a checkbox complete merely because a file exists.
10. **At each phase end, stop and report.** Summarise files changed, commands run, test/output evidence, issues, and the exact next phase. Do not silently continue through every phase in one giant unreviewable change.

## 4. Current implementation context to verify

Report 1 describes the following work as already undertaken:

- R-based preprocessing/EDA using packages including `tidyverse`, `lubridate`, `janitor`, `stringr`, `naniar`, `ggplot2`, and related utilities.
- Primary dataset: Kaggle LinkedIn Job Postings 2023–24, with fields such as `job_id`, `title`, `company_name`, `description`, `skills_desc`, `location`, experience/work type, salary fields, `listed_time`, `views`, and `applies`.
- Validation dataset: Data Analyst Job Postings from Kaggle, with ready-made `job_skills`.
- Current reported intermediate output: 123,849 raw rows, 31 columns; 22,372 tech/data postings after cleaning/filtering; all retained postings reportedly in April 2024.
- Existing output names described in the report:
  - `output/skillmine_clean.csv`
  - `output/skillmine_transactions.csv`
  - `output/before_after_stats.csv`
  - `output/missing_before.csv`
  - `output/figures/fig1_missing_values.png` through the EDA figure files
- The report’s code uses a seed skill dictionary of approximately 37 skills, even though the target design describes growing this to approximately 300. Treat expansion and validation as work to do, not as already completed.
- The report labels some before/after values as illustrative and explicitly says they should be replaced with actual generated output. Recompute them from the actual run before reusing them in the final report.
- The current validation snippet described in the report calculates dictionary coverage against the validation skill list; that is **not the same as extractor precision**. Implement a proper posting-level/manual validation and report precision honestly.
- The high-level review specifies Python libraries for the mining stages: `mlxtend` for FP-Growth, `NetworkX` and `python-louvain` for the skill network, `scikit-learn` for TF-IDF/K-Means/silhouette, and `pandas`/NumPy for trend calculations. The detailed report says preprocessing/EDA is in R and permits R Shiny or Streamlit for the website. Inspect the repository and retain the existing split where practical; do not port the R pipeline just for uniformity.
- Live job sources mentioned as examples in the report include Adzuna and public Greenhouse/Lever boards. These are options to investigate, not proof that credentials or access are available.

**Important:** These are facts reported in the documents, not a substitute for inspecting the current repository and rerunning the code.

## 5. Expected architecture and data contracts

Keep the pipeline understandable and reproducible:

`Raw historical postings → clean/filter → normalized text → skill extraction → posting-level skill sets / binary matrix → {association rules, skill network + Louvain, TF-IDF + K-Means, monthly trends} → dashboard`

The live feed is a separate ingestion path:

`Permitted job APIs/boards → normalize fields → shared skill extractor → recent-job store/cache → skill-match scoring → live job board`

Historical Kaggle postings and live jobs must remain distinguishable. Do not pretend the historical snapshot is live data.

### Shared posting schema

Map actual source columns to a consistent internal schema without losing useful original fields. At minimum, preserve where available:

- `job_id` or a stable source-specific ID
- `title`
- `company_name`
- `description` / cleaned description
- `listed_time` and parsed `listed_date` / `listed_month`
- `location`
- `experience_level`
- `work_type`
- `remote_allowed`
- `original_url` for live listings where available
- `skills` as a normalized list/set
- `n_skills`

For live postings, also store source/provider and `fetched_at`/`last_refreshed_at`. Use a stable ID or deduplication key to prevent duplicate listings across refreshes.

### Skill vocabulary and extraction

- Build a curated dictionary with canonical skill names and aliases/synonyms.
- Use robust regex/token boundaries where appropriate. Preserve technical names such as `C++`, `C#`, `Node.js`, `.NET`, `CI/CD`, and `scikit-learn`.
- Prevent false positives from ambiguous short tokens (for example, language names or abbreviations that appear in ordinary prose).
- Normalize synonyms to one canonical label, e.g. `sklearn` and `scikit learn` → `scikit-learn`.
- Each posting must have a deduplicated skill set. Keep the original posting identifier so every skill set can be traced back to its posting.
- Do not remove no-skill postings silently. Count them, inspect examples, and document their effect on analysis.
- The dictionary should be expanded toward the stated ~300 skills only when supported by validation data and sensible aliases; do not add arbitrary entries just to hit a number.

## 6. Phased implementation plan

### Phase 0 — Repository audit and execution plan

**Goal:** Establish the true state of the project before editing.

Tasks:
1. Inspect repository tree, Git status/branch, README, dependency files, scripts, notebooks, data/output directories, and any existing dashboard.
2. Identify which current files implement preprocessing, EDA, extraction, mining, visualizations, and UI.
3. Run existing safe tests/scripts where feasible and record errors. Do not run expensive or destructive jobs before checking their inputs.
4. Compare actual outputs to Report 1. Verify row counts, columns, duplicates, skill counts, date range, and figure/output paths.
5. Identify missing datasets or unavailable credentials without exposing secrets.
6. Create/update a concise implementation status checklist and dependency map.

**Exit criteria:**
- A written status report identifies what is complete, partial, broken, and missing.
- The actual project commands and data paths are known.
- No unrelated code has been changed.

**Stop after Phase 0 and report findings before proceeding.**

### Phase 1 — Stabilize preprocessing, data quality, and skill extraction

**Goal:** Produce a trustworthy, repeatable shared dataset for every downstream algorithm.

Tasks:
1. Fix reproducibility and path assumptions in the current R preprocessing script. Make output directories safely and document how to run it.
2. Verify timestamp units and parsed dates against source records. Investigate the apparent one-month date range rather than assuming it is correct.
3. Recompute all before/after statistics from code. Replace illustrative report values with actual values in generated outputs.
4. Confirm deduplication is applied in a defined order and counts are reported accurately.
5. Audit title filtering for false positives/negatives; report retained count and categories.
6. Expand/organize the canonical skill dictionary and aliases; add tests for tricky technical names and false positives.
7. Generate both:
   - posting-level clean data with one row per posting and a list/set of skills;
   - long transaction data with one row per `(job_id, skill)`;
   - a binary posting × skill matrix or sparse equivalent for algorithms that need it.
8. Add data validation checks: unique IDs, valid date values, canonical skill names, no duplicate `(job_id, skill)` pairs, consistent row counts, and handling of zero-skill postings.
9. Implement a proper extractor evaluation on about 100 manually reviewed postings. Record expected skills and extracted skills; calculate precision (and, where feasible, recall/F1). Explain the annotation method.
10. Use the ready-made validation dataset as a cross-check only after mapping its schema and canonical skill labels correctly. Do not call dictionary coverage “precision.”

**Expected artifacts:** cleaned data, transaction table, matrix/sparse matrix or a reproducible builder, corrected statistics, skill dictionary, validation sample/results, and tests.

**Exit criteria:** The pipeline runs from documented inputs to outputs; key invariants pass; actual data range and extraction quality are reported; downstream code can consume stable outputs.

### Phase 2 — FP-Growth association rules

**Goal:** Answer “Which skills go together?”

Tasks:
1. Build transactions from the Phase 1 posting-level skill sets, ensuring each skill appears at most once per posting.
2. Use `mlxtend` FP-Growth for frequent itemsets. Start around 1–2% minimum support as proposed, but tune based on actual transaction count and useful output.
3. Generate association rules and calculate/support the standard metrics: support, confidence, lift; include antecedent and consequent in readable form.
4. Initially retain rules with confidence ≥ 0.5 and lift > 1.2 as proposed, but make thresholds configurable and report if they need tuning.
5. Filter out trivial/self rules and low-value results; guard against empty results at chosen thresholds.
6. Run overall and by a meaningful role group derived from titles. Do not make groups too small to be statistically useful.
7. Rank by lift while also showing support/confidence, so high lift based on tiny counts is not misleading.
8. Export stable CSV outputs and create readable tables for the dashboard.
9. Manually sanity-check the strongest rules and document a few sensible and surprising examples without claiming causality.

**Expected artifacts:** frequent itemsets and association rules overall/per role, parameters/configuration, tests, and a concise interpretation.

**Exit criteria:** Results are reproducible, metrics are correctly defined, thresholds are recorded, and top rules pass manual sanity checks.

### Phase 3 — Skill co-occurrence network and Louvain communities

**Goal:** Find skill families from co-occurrence patterns.

Tasks:
1. Build a skill co-occurrence graph from the posting × skill data. Nodes are canonical skills; edges represent skills appearing in the same posting.
2. Follow the review design by weighting relationships using PMI/co-occurrence. Define the formula and treatment of zero/negative PMI explicitly. Avoid allowing very rare skills to dominate the graph.
3. Use `NetworkX` and a compatible Louvain implementation (`python-louvain` or a suitable installed alternative after checking dependencies).
4. Save node attributes including skill, community ID, and useful frequency measures; save edge weights.
5. Inspect graph connectivity and tune a documented minimum frequency/edge threshold for legibility, not merely to force a pretty graph.
6. Export graph data and create an interactive PyVis graph with colours by community/skill family.
7. Produce a community summary: member skills, frequent skills, example job titles, and a cautious human-readable label such as “Cloud/DevOps” only when supported by the members.

**Expected artifacts:** graph edge/node tables, community assignments, interactive graph HTML, and a summary table.

**Exit criteria:** Graph is generated from actual data, community assignments are reproducible enough for the workflow, and the visualization is usable at realistic scale.

### Phase 4 — K-Means posting clusters / role archetypes

**Goal:** Group postings with similar skill profiles.

Tasks:
1. Represent each posting’s extracted skill set using TF-IDF as specified in the review. Preserve posting IDs and title metadata for interpretation.
2. Use scikit-learn K-Means. Test a reasonable range of `k` (the review suggests 6–12) and calculate silhouette scores where computationally and mathematically appropriate.
3. Handle sparse matrices efficiently; avoid converting a large sparse matrix to dense without a clear memory reason.
4. Record random seeds and parameters for reproducibility.
5. Select `k` using silhouette results plus interpretability; do not claim the largest score is automatically the best if clusters are unusable.
6. Describe clusters with top weighted skills and representative job titles. Give each cluster a cautious, evidence-based role archetype name.
7. Export posting-to-cluster assignments, cluster summaries, and evaluation metrics.

**Expected artifacts:** cluster assignments, silhouette comparison, top skills/titles per cluster, and a visual or table summary.

**Exit criteria:** Clusters are reproducible, quality metrics are saved, and cluster labels are grounded in real posting content.

### Phase 5 — Monthly trends and emerging-skill detection

**Goal:** Identify skills with rising share over time without overstating evidence.

Tasks:
1. Confirm that the historical dataset contains enough distinct months. The report currently says all filtered rows are from April 2024; if that is true, this phase is blocked until legitimate multi-month data is obtained.
2. Investigate timestamp parsing, selected source files, and filters to determine whether the single-month result is a data issue or a genuine source limitation.
3. If needed, obtain a legitimate public historical snapshot covering multiple months or years. Document source and licence/terms; do not fabricate dates or duplicate postings into fake months.
4. Calculate each skill’s monthly share as postings containing that skill divided by all eligible postings in that month. Ensure numerator and denominator use the same posting population.
5. Calculate growth as the slope of a linear fit over monthly share, as proposed. Record the time interval and number of months; optionally report absolute/relative change only where well-defined.
6. Flag “emerging” skills only when they have a high positive slope and a low/medium base share, with configurable minimum frequency and adequate time coverage.
7. Clearly distinguish genuine growth from noisy results caused by sparse months, short time windows, or changing data coverage.
8. Produce the top 15 chart/table with skill, base share, slope/growth score, and relevant period.
9. If adequate data cannot be obtained, implement an honest UI state: trends unavailable/insufficient monthly coverage. Do not output a fake “top emerging skills” ranking.

**Expected artifacts:** monthly share table, growth-score table, chart, data coverage summary, and documented limitation or justified result.

**Exit criteria:** Every reported trend is based on at least multiple real months and its time coverage is visible; otherwise the system explicitly marks trend analysis as unavailable.

### Phase 6 — Integrate historical analysis dashboard and skill explorer

**Goal:** Make the mining results understandable and interactive.

Use the existing app framework if one already exists. The project report permits R Shiny or Streamlit; the review plan specifically proposes Streamlit. Do not introduce a second UI framework without a concrete reason.

Required pages/sections:
1. **Overview:** dataset coverage, posting count, time range, extracted skill count, and key caveats.
2. **Association rules:** overall/by-role tables, filters, support/confidence/lift, readable antecedent → consequent.
3. **Skill network:** interactive graph coloured by Louvain community, with useful hover details.
4. **Role clusters:** cluster summaries, top skills/titles, silhouette scores.
5. **Emerging skills:** top 15 skills and trend chart, or an explicit insufficient-data explanation.
6. **Skill explorer:** search/select a canonical skill and display its strongest association partners, community, role-cluster context where meaningful, and monthly trend when valid.
7. Clear loading/empty/error states. Missing output files should trigger useful instructions rather than cryptic stack traces.

Implementation requirements:
- Load precomputed artifacts; do not rerun expensive mining algorithms on every UI interaction.
- Use caching where appropriate and invalidate cache when artifacts change.
- Make file paths relative/configurable and support launching from the documented project root.
- Keep chart labels, units, time periods, and metric definitions clear.
- Avoid claims of causation or job-market certainty.

**Exit criteria:** The dashboard launches from documented commands and all analysis sections work with real generated outputs.

### Phase 7 — Live job board and skill-match ranking

**Goal:** Add recent job discovery as a separate, clearly labelled feature.

Tasks:
1. Inspect which permitted providers are realistically accessible. Examples in the report include Adzuna API and public Greenhouse/Lever boards; provider availability, API credentials, quotas, and terms must be verified before implementation.
2. Implement a provider adapter interface so a source can be added or disabled without rewriting the UI.
3. Do not scrape LinkedIn pages. Use only public APIs/feeds that permit programmatic access.
4. Fetch on an hourly schedule where deployment supports scheduling. Keep provider keys in environment variables.
5. Normalize live fields into the shared schema and use the same skill vocabulary/extractor as historical analysis.
6. Store/cache recent jobs in SQLite or the existing supported database. Deduplicate by provider ID or stable key.
7. Filter to postings whose provider-supplied posted timestamp is within the last 5–6 hours. If posted time is missing or unreliable, label the timestamp unknown and do not falsely claim it is recent.
8. Show the last successful refresh time, source/provider, and errors/stale-data state. Never display a fake “updated now” time.
9. Let users enter skills (and, if practical, upload a simple skills file). Normalize these using the same canonical dictionary.
10. Compute match score as `number of required posting skills the user has / number of extracted required posting skills × 100`. Explain the formula. Handle postings with no extracted skills as “match unavailable,” not 0% or 100%.
11. Show matched skills, missing skills, match percentage, and an Apply link to the original employer posting. SkillMine must not submit applications on behalf of users.
12. Add filters for role, location, remote status, and experience level.
13. Association rules may suggest a next skill to learn, but recommendations must be grounded in mined rules and clearly worded as suggestions, not guarantees.
14. If API access is unavailable, build and test the provider interface and show a clear unavailable state. Do not quietly replace live results with historical listings.

**Exit criteria:** The board distinguishes live data from historical analysis, respects source terms, ranks skill matches correctly, and exposes truthful freshness/error information.

### Phase 8 — End-to-end QA, evaluation, report, and delivery

**Goal:** Make the project reproducible and review-ready.

Tasks:
1. Run the full pipeline from documented raw inputs to outputs and dashboard.
2. Add/complete tests for data contracts, extraction aliases, transaction uniqueness, metric calculations, empty datasets, single-month trend handling, match scores, duplicate live jobs, and missing API credentials.
3. Confirm the association-rule metrics against small hand-checkable examples.
4. Report:
   - skill extraction precision (and recall/F1 if annotated data permits);
   - K-Means silhouette results and chosen `k`;
   - manual sanity-check results for top association rules;
   - time coverage and trend limitations;
   - live feed source, refresh behaviour, and access limitations.
5. Verify all figures/tables are generated from actual current outputs. Remove stale/illustrative statistics from final materials.
6. Write setup instructions, dependencies, data placement instructions, run commands, environment variables, and troubleshooting.
7. Document limitations: LinkedIn/US bias, incomplete skill dictionary, missing fields, short or narrow time coverage, provider API constraints, and noisy extraction.
8. Prepare final report/review materials aligned with the implemented system; do not describe unimplemented features as complete.

**Exit criteria:** Fresh setup instructions work, tests pass or documented blockers remain, all four core review deliverables are present, and claims match evidence.

## 7. Recommended repository organisation

Adapt to the current repository rather than forcing this exact structure. Avoid duplicate implementations.

```text
project-root/
  README.md
  CLAUDE.md                     # optional short pointer to this master plan
  requirements.txt              # if Python dependencies are used
  renv.lock / R package notes   # if R dependency management is present
  .env.example
  .gitignore
  data/
    README.md                   # explain where to obtain data; do not commit large/private data blindly
  R/
    preprocess.R
    eda.R
  src/
    skillmine/
      config.py
      data_io.py
      skill_extraction.py
      association_rules.py
      skill_network.py
      clustering.py
      trends.py
      live_jobs/
      matching.py
  app/                          # only if a separate app directory fits the existing project
  output/
    README.md
    tables/
    figures/
    network/
  tests/
```

The names above are suggestions. Keep current naming conventions if they already work.

## 8. Commands, configuration, and reproducibility

- Discover and document the actual commands after auditing the repository; do not invent commands that do not exist.
- Use a fixed random seed for clustering and any stochastic algorithm.
- Put tunable thresholds in one config file or documented constants, not scattered through scripts.
- Do not commit downloaded datasets or generated large artifacts unless the repository’s existing policy requires it. Document how to obtain/recreate them.
- Ensure generated artifacts record relevant parameters, run date, data source, and time coverage where practical.
- Keep raw data immutable; write cleaned data to separate outputs.
- If an API needs credentials, validate configuration and provide an actionable error without printing the secret.
- If a task cannot run because data or credentials are missing, report the blocker and add tests for the available behaviour.

## 9. Git and change-management rules

- Begin by checking `git status` and current branch. Do not discard or overwrite the user's uncommitted work.
- Make changes in small logical batches aligned to phases.
- Run relevant tests after each change.
- Do not commit automatically unless the user explicitly asks for commits. If asked to commit, inspect the diff, ensure secrets/data are not included, and use clear phase-based commit messages.
- At the end of each phase, provide:
  1. what changed;
  2. files added/modified;
  3. commands/tests run and their real results;
  4. outputs created;
  5. unresolved issues/blockers;
  6. whether exit criteria are satisfied;
  7. the next phase that should be started.
- Do not start Phase N+1 until Phase N exit criteria are satisfied, or explicitly mark a blocker and ask for approval to proceed with a documented workaround.

## 10. First instruction to Claude Code

Start with **Phase 0 only**.

1. Read this entire file.
2. Inspect the repository and current Git state.
3. Compare current code and outputs with the project specification and Report 1 context above.
4. Run only safe, relevant checks.
5. Return a concise but evidence-based audit: completed / partial / broken / missing, actual date coverage, actual output paths, and blockers.
6. Propose the smallest next set of changes for Phase 1.
7. **Stop and wait for approval before implementing Phase 1.**

The master goal is to complete SkillMine as specified—not to rewrite it from scratch, not to hide data limitations, and not to mark unfinished features as done.
