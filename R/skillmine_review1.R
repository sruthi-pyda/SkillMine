# =================================================================
# SkillMine - Report 1: Dataset, Preprocessing & EDA
# Team: Rampudi Jaiwanth Ram, Pyda Venkata Satya Sruthi, Hari Shree
# Data: Kaggle "LinkedIn Job Postings 2023-24" (arshkon) - primary
#       Kaggle "Data Analyst Job Postings" (asaniczka) - validation
# Put postings.csv (from linkedin_primary/) in data/
# Run from project root: Rscript R/skillmine_review1.R
# =================================================================

# ---- 0. Packages ---------------------------------------------------
pkgs <- c("tidyverse", "lubridate", "janitor", "skimr", "stringr", "scales")
new  <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) {
  options(repos = c(CRAN = "https://cran.r-project.org"))
  install.packages(new)
}
invisible(lapply(pkgs, library, character.only = TRUE))

dir.create("output",         showWarnings = FALSE)
dir.create("output/figures", showWarnings = FALSE)

# ---- 1. Load raw data ---------------------------------------------
raw <- read_csv("data/linkedin_primary/postings.csv",
                show_col_types = FALSE) |>
  clean_names()

cat("RAW: rows =", nrow(raw), " cols =", ncol(raw), "\n")
glimpse(raw)

# Before-statistics snapshot
before_stats <- tibble(
  stage            = "Before cleaning",
  rows             = nrow(raw),
  cols             = ncol(raw),
  duplicate_ids    = sum(duplicated(raw$job_id)),
  duplicate_text   = sum(duplicated(raw[, c("title", "company_name", "description")])),
  missing_desc     = sum(is.na(raw$description)),
  missing_salary   = sum(is.na(raw$max_salary) & is.na(raw$med_salary)),
  missing_skills   = sum(is.na(raw$skills_desc))
)
print(before_stats)

# Missing-value profile (before) — base-R version of miss_var_summary
miss_before <- tibble(
  variable = names(raw),
  n_miss   = map_int(raw, ~ sum(is.na(.x))),
  pct_miss = n_miss / nrow(raw) * 100
) |> arrange(desc(pct_miss))
write_csv(miss_before, "output/missing_before.csv")

# ---- 2. Data cleaning ---------------------------------------------
# 2.1 Keep only the columns the project needs
df <- raw |>
  select(job_id, company_name, title, description, skills_desc,
         location, formatted_experience_level, formatted_work_type,
         remote_allowed, max_salary, med_salary, min_salary,
         pay_period, listed_time, views, applies)

# 2.2 Remove duplicates (same id, then same title+company+description)
df <- df |>
  distinct(job_id, .keep_all = TRUE) |>
  distinct(title, company_name, description, .keep_all = TRUE)
cat("After de-duplication:", nrow(df), "\n")

# 2.3 Drop rows without a usable description (skills come from it)
df <- df |> filter(!is.na(description), str_length(description) >= 100)
cat("After removing empty/short descriptions:", nrow(df), "\n")

# 2.4 Convert listed_time (epoch milliseconds) to date and month
df <- df |>
  mutate(listed_date  = as_date(as_datetime(listed_time / 1000)),
         listed_month = floor_date(listed_date, "month"))

# 2.5 Normalise text: lower case, strip HTML, URLs, extra spaces
clean_text <- function(x) {
  x |>
    str_to_lower() |>
    str_replace_all("<[^>]+>", " ") |>
    str_replace_all("http\\S+|www\\.\\S+", " ") |>
    str_replace_all("[^a-z0-9+#./\\- ]", " ") |>
    str_squish()
}

df <- df |>
  mutate(title_clean = clean_text(title),
         desc_clean  = clean_text(description))

# 2.6 Filter to tech and data roles using title keywords
tech_pattern <- paste0(
  "data|analyst|scientist|engineer|developer|software|machine learning|",
  "\\bml\\b|\\bai\\b|cloud|devops|python|java|web|full stack|backend|",
  "frontend|database|\\bbi\\b|business intelligence|cyber|security|",
  "\\bit\\b|programmer|architect|sre|qa|test automation"
)
df_tech <- df |> filter(str_detect(title_clean, tech_pattern))
cat("Tech/data subset:", nrow(df_tech), "\n")

# 2.7 Standardise categorical fields
df_tech <- df_tech |>
  mutate(
    experience_level = replace_na(formatted_experience_level, "Not stated"),
    work_type        = replace_na(formatted_work_type, "Not stated"),
    remote_allowed   = if_else(remote_allowed %in% 1, "Remote", "On-site/Hybrid"),
    state            = str_trim(str_extract(location, "[A-Z]{2}$")),
    state            = replace_na(state, "Other/Unknown")
  )

# 2.8 Salary: convert all pay periods to an annual figure, cap outliers
to_annual <- function(v, period) {
  case_when(
    period == "HOURLY"   ~ v * 2080,
    period == "WEEKLY"   ~ v * 52,
    period == "BIWEEKLY" ~ v * 26,
    period == "MONTHLY"  ~ v * 12,
    TRUE                 ~ v
  )
}
df_tech <- df_tech |>
  mutate(salary_raw    = coalesce(med_salary, (min_salary + max_salary) / 2, max_salary),
         salary_annual = to_annual(salary_raw, pay_period))

q <- quantile(df_tech$salary_annual, c(0.01, 0.99), na.rm = TRUE)
df_tech <- df_tech |>
  mutate(salary_annual = if_else(salary_annual < q[1] | salary_annual > q[2],
                                 NA_real_, salary_annual))

# ---- 3. Skill extraction (dictionary + regex) ---------------------
skill_dict <- tribble(
  ~skill,            ~pattern,
  "python",          "\\bpython\\b",
  "sql",             "\\bsql\\b|\\bmysql\\b|\\bpostgres(ql)?\\b|\\bbt-sql\\b",
  "r",               "\\br\\b(?=[ ,/)])|\\br programming\\b|\\brstudio\\b",
  "java",            "\\bjava\\b(?!\\.?script)",
  "javascript",      "\\bjavascript\\b|\\bjs\\b",
  "typescript",      "\\btypescript\\b",
  "c#",              "c#|\\.net\\b",
  "c++",             "c\\+\\+",
  "excel",           "\\bexcel\\b",
  "tableau",         "\\btableau\\b",
  "power bi",        "\\bpower ?bi\\b",
  "aws",             "\\baws\\b|amazon web services",
  "azure",           "\\bazure\\b",
  "gcp",             "\\bgcp\\b|google cloud",
  "docker",          "\\bdocker\\b",
  "kubernetes",      "\\bkubernetes\\b|\\bk8s\\b",
  "spark",           "\\bspark\\b|\\bpyspark\\b",
  "hadoop",          "\\bhadoop\\b",
  "airflow",         "\\bairflow\\b",
  "snowflake",       "\\bsnowflake\\b",
  "databricks",      "\\bdatabricks\\b",
  "machine learning","machine learning|\\bml\\b",
  "deep learning",   "deep learning",
  "tensorflow",      "\\btensorflow\\b",
  "pytorch",         "\\bpytorch\\b",
  "scikit-learn",    "scikit[- ]learn|\\bsklearn\\b",
  "nlp",             "\\bnlp\\b|natural language processing",
  "generative ai",   "generative ai|\\bgenai\\b|\\bllms?\\b|large language model",
  "statistics",      "\\bstatistic(s|al)\\b",
  "git",             "\\bgit\\b|github|gitlab",
  "linux",           "\\blinux\\b|\\bunix\\b",
  "react",           "\\breact(\\b|\\.js)?\\b",
  "node.js",         "node\\.?js",
  "ci/cd",           "ci/cd|continuous integration",
  "terraform",       "\\bterraform\\b",
  "rest api",        "rest(ful)? api",
  "agile",           "\\bagile\\b|\\bscrum\\b",
  "communication",   "communication skills?",
  # ── Expansion: validated against asaniczka dataset (top unextracted skills) ──
  "sap",             "\\bsap\\b",
  "jira",            "\\bjira\\b",
  "sas",             "\\bsas\\b",
  "salesforce",      "\\bsalesforce\\b",
  "looker",          "\\blooker\\b",
  "data visualization","data visuali[sz]ation",
  "ms office",       "microsoft office|\\bms office\\b",
  "powerpoint",      "\\bpowerpoint\\b|\\bppt\\b",
  "data modeling",   "data model(l?ing)?",
  "etl",             "\\betl\\b|extract.*transform.*load",
  "data warehousing","data warehou(s|z)ing|\\bdwh\\b",
  "kafka",           "\\bkafka\\b",
  "mongodb",         "\\bmongodb\\b|\\bmongo\\b",
  "redis",           "\\bredis\\b",
  "scala",           "\\bscala\\b",
  "bash",            "\\bbash\\b|shell scripting|\\bshell script",
  "flask",           "\\bflask\\b",
  "django",          "\\bdjango\\b",
  "angular",         "\\bangular\\b",
  "vue",             "\\bvue\\.?js\\b",
  "selenium",        "\\bselenium\\b",
  "dbt",             "\\bdbt\\b",
  "matlab",          "\\bmatlab\\b",
  # ── Phase 5 additions: evidence-based dictionary expansion ───────────
  # go/golang: 174 occurrences in tech postings; \bgolang\b is unambiguous.
  # Bare \bgo\b rejected — 1,586 broad matches risk "go to/go ahead" FPs.
  "go",              "\\bgolang\\b",
  # erp: 702 occurrences; enterprise resource planning — low false-positive risk.
  "erp",             "\\berp\\b",
  # swift: 161 occurrences; iOS/Apple context validated by title inspection
  # (iOS Developer, iOS Software Engineer, Mobile SDK Engineer).
  "swift",           "\\bswift\\b",
  # api: 1,513 occurrences; technical acronym with \b boundary — safe addition.
  # Note: \brest api\b already in dictionary; this captures standalone "api" use.
  "api",             "\\bapi\\b"
)

extract_skills <- function(text) {
  hits <- map_lgl(skill_dict$pattern, ~ str_detect(text, regex(.x)))
  paste(skill_dict$skill[hits], collapse = ";")
}

df_tech <- df_tech |>
  mutate(skills   = map_chr(desc_clean, extract_skills),
         n_skills = if_else(skills == "", 0L, str_count(skills, ";") + 1L))

# Long (posting x skill) table - the transaction format used in Steps 3-5
skills_long <- df_tech |>
  select(job_id, listed_month, skills) |>
  separate_rows(skills, sep = ";") |>
  filter(skills != "") |>
  rename(skill = skills)

# ---- 4. After-statistics and save cleaned data --------------------
after_stats <- tibble(
  stage            = "After cleaning",
  rows             = nrow(df_tech),
  cols             = ncol(df_tech),
  duplicate_ids    = sum(duplicated(df_tech$job_id)),
  duplicate_text   = sum(duplicated(df_tech[, c("title", "company_name", "description")])),
  missing_desc     = sum(is.na(df_tech$description)),
  missing_salary   = sum(is.na(df_tech$salary_annual)),
  missing_skills   = sum(df_tech$n_skills == 0)
)

before_after <- bind_rows(before_stats, after_stats)
print(before_after)
write_csv(before_after, "output/before_after_stats.csv")

summary(df_tech$n_skills)
summary(df_tech$salary_annual)

write_csv(df_tech |> select(-description, -desc_clean),
          "output/skillmine_clean.csv")
write_csv(skills_long, "output/skillmine_transactions.csv")

cat("Clean postings written:     output/skillmine_clean.csv\n")
cat("Transaction table written:  output/skillmine_transactions.csv\n")
cat("Unique skills in dict:     ", nrow(skill_dict), "\n")
cat("Total (job_id, skill) rows:", nrow(skills_long), "\n")
cat("Date range: ",
    format(min(df_tech$listed_date, na.rm=TRUE)), "to",
    format(max(df_tech$listed_date, na.rm=TRUE)), "\n")

# ---- 5. EDA & visualisations --------------------------------------
theme_set(theme_minimal(base_size = 12))
save_fig <- function(p, name) {
  ggsave(file.path("output/figures", name), p, width = 8, height = 5, dpi = 300)
}

# Fig 1: Missing values before cleaning
p1 <- miss_before |> slice_max(pct_miss, n = 15) |>
  ggplot(aes(reorder(variable, pct_miss), pct_miss)) +
  geom_col(fill = "#C0622B") + coord_flip() +
  labs(title = "Missing values per column (raw data)",
       x = "Column", y = "Missing (%)")
save_fig(p1, "fig1_missing_values.png")

# Fig 2: Top 20 skills by share of postings
top_skills <- skills_long |> count(skill, sort = TRUE) |>
  mutate(share = n / nrow(df_tech)) |> slice_head(n = 20)
p2 <- ggplot(top_skills, aes(reorder(skill, share), share)) +
  geom_col(fill = "#1E2A44") + coord_flip() +
  scale_y_continuous(labels = percent) +
  labs(title = "Top 20 skills in tech/data job postings",
       x = "Skill", y = "Share of postings")
save_fig(p2, "fig2_top_skills.png")

# Fig 3: Number of skills per posting
p3 <- ggplot(df_tech, aes(n_skills)) +
  geom_histogram(binwidth = 1, fill = "#2E6E8E", colour = "white") +
  labs(title = "Distribution of skills extracted per posting",
       x = "Number of skills", y = "Number of postings")
save_fig(p3, "fig3_skills_per_posting.png")

# Fig 4: Postings per month
p4 <- df_tech |> count(listed_month) |>
  ggplot(aes(listed_month, n)) + geom_line(colour = "#C0622B", linewidth = 1) +
  geom_point() +
  labs(title = "Tech/data postings listed per month",
       x = "Month", y = "Number of postings")
save_fig(p4, "fig4_postings_per_month.png")

# Fig 5: Experience level
p5 <- df_tech |> count(experience_level) |>
  ggplot(aes(reorder(experience_level, n), n)) +
  geom_col(fill = "#2E6E8E") + coord_flip() +
  labs(title = "Postings by experience level",
       x = "Experience level", y = "Number of postings")
save_fig(p5, "fig5_experience_level.png")

# Fig 6: Salary distribution by experience level
p6 <- df_tech |> filter(!is.na(salary_annual)) |>
  ggplot(aes(experience_level, salary_annual)) +
  geom_boxplot(fill = "#E6D9C6") + coord_flip() +
  scale_y_continuous(labels = dollar) +
  labs(title = "Annual salary by experience level",
       x = "Experience level", y = "Annual salary (USD)")
save_fig(p6, "fig6_salary_by_level.png")

# Fig 7: Top 10 states
p7 <- df_tech |> count(state, sort = TRUE) |> slice_head(n = 10) |>
  ggplot(aes(reorder(state, n), n)) +
  geom_col(fill = "#1E2A44") + coord_flip() +
  labs(title = "Top 10 locations (US state) for tech/data postings",
       x = "State", y = "Number of postings")
save_fig(p7, "fig7_top_states.png")

# Fig 8: Co-occurrence heat map of the top 15 skills
top15 <- top_skills$skill[1:15]
m <- skills_long |> filter(skill %in% top15) |>
  mutate(v = 1L) |> distinct() |>
  pivot_wider(names_from = skill, values_from = v, values_fill = 0) |>
  select(all_of(top15)) |> as.matrix()
co <- cor(m)
co_long <- as_tibble(co, rownames = "skill_x") |>
  pivot_longer(-skill_x, names_to = "skill_y", values_to = "phi") |>
  filter(skill_x != skill_y)
p8 <- ggplot(co_long, aes(skill_x, skill_y, fill = phi)) +
  geom_tile() +
  scale_fill_gradient2(low = "#6688BB", mid = "white", high = "#C0622B",
                       midpoint = 0, name = "Corr") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Co-occurrence (phi correlation) of top 15 skills",
       x = NULL, y = NULL)
save_fig(p8, "fig8_cooccurrence.png")

# Fig 9: Monthly share of postings for selected skills
watch_skills <- c("python", "sql", "aws", "generative ai", "kubernetes", "power bi")
p9_data <- skills_long |>
  filter(skill %in% watch_skills) |>
  count(listed_month, skill) |>
  left_join(df_tech |> count(listed_month, name = "total"), by = "listed_month") |>
  mutate(share = n / total)
p9 <- ggplot(p9_data, aes(listed_month, share, colour = skill)) +
  geom_line(linewidth = 1) + geom_point() +
  scale_y_continuous(labels = percent) +
  labs(title = "Monthly share of postings for selected skills",
       x = "Month", y = "Share of postings", colour = "Skill")
save_fig(p9, "fig9_monthly_skill_share.png")

cat("\nAll figures saved to output/figures/\n")
cat("NOTE: Only April 2024 data — trend analysis requires multiple months.\n")
