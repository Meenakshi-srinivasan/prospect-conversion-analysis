# Prospect Conversion Forecasting

A data-driven lead prioritization framework that identifies free portals most likely to convert into paying customers within a 30-day horizon. The system combines **behavioral scoring** (LightGBM / XGBoost with Optuna tuning), **exploratory analysis**, and **GenAI-based segmentation** to produce weekly ranked prospect lists and tailored outreach playbooks.

---

## Objective

- **Predict** 30-day conversion probability from product usage, firmographics, and recency.
- **Rank** prospects by likelihood to convert so revenue teams can prioritize outreach.
- **Segment** top prospects with an LLM into behavior patterns, lifecycle stages, and playbook focus.
- **Output** a weekly, sales-ready list with conversion scores, segments, and suggested messaging.

The solution integrates:

1. **ICP qualification** — firmographics (industry, size, Alexa rank).
2. **Behavioral intent modeling** — 30-day usage signals and engagement depth.
3. **Momentum detection** — recency and usage acceleration.
4. **GenAI segmentation** — structured playbooks and personalized outreach from behavioral signals.

---

## Key Assumptions

- **noncustomers.csv** represents active free-tier portals that have not yet converted. Absence from the customer dataset is treated as non-paying status.
- Snapshots are **pre-conversion** only (rows on or after a company’s close date are excluded).
- The target **LABEL_CONVERT_30D** is 1 if the company closes within 30 days after the snapshot date, 0 otherwise.

---

## Data Sources

| Source | Purpose |
|--------|---------|
| **customers.csv** | Paid customers: `id`, `CLOSEDATE`, `MRR`, firmographics (ALEXA_RANK, EMPLOYEE_RANGE, INDUSTRY). |
| **noncustomers.csv** | Prospects: `id`, firmographics (no CLOSEDATE). |
| **usage_actions.csv** | Daily usage: `id`, `WHEN_TIMESTAMP`, action/user counts (e.g. ACTIONS_CRM_DEALS, ACTIONS_EMAIL, USERS_*). |

Firmographics and usage column names are configured in the notebook (e.g. `FIRMOGRAPHIC_COLS`, `USAGE_ACTION_COLS`, `USAGE_USER_COLS`).

---

## Project Structure

```
Prospect Conversion/
├── Prospect Conversion Forecasting.ipynb   # Main pipeline (data → modeling → evaluation → segmentation)
├── README.md
├── requirements.txt
├── .env                                    # Not in repo; add OPENAI_API_KEY
├── prompts/
│   ├── system.txt                          # LLM system prompt for segmentation
│   └── task.txt                            # Task template (company + usage → taxonomy)
├── utilities/
│   ├── helper_functions_for_forecasting.py  # Data/forecasting helpers (midpoints, log1p, rolling, etc.)
│   └── segmentation_helper_functions.py     # Segmentation helpers (_num, _int, _str, build_prompt, parse_and_validate)
└── results/                                # Output CSVs (created when notebook runs)      
    ├── sunday_ranked_prospects_xgb.csv     # XGBoost-ranked list 
    └── sunday_ranked_prospects_enriched_taxonomy.csv  # Top N with GenAI segmentation 
```

---

## Setup

1. **Python**  
   Python 3.9+ recommended (e.g. 3.10–3.13).

2. **Virtual environment (optional but recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **OpenAI API key (for GenAI segmentation)**  
   - Set `OPENAI_API_KEY=...` (one line, no spaces around `=`).

---

## How to Run

Open **Prospect Conversion Forecasting.ipynb** and run cells in order:

| Section | What it does |
|--------|----------------|
| **Import Packages** | Installs/imports pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, xgboost, optuna, openai, etc. |
| **Data Gathering** | Loads customers, noncustomers, usage CSVs; defines config (windows, columns). |
| **Data Cleaning, Processing and Transformation** | Builds firmographics, daily usage, Sunday snapshots, rolling 7/14/30-day features, recency, and LABEL_CONVERT_30D. |
| **Exploratory Data Analysis** | Overview, target distribution, firmographics, numeric feature histograms, correlation heatmap. |
| **Modeling** | Train/val/test split (time-based), derived features, preprocessing, LightGBM with early stopping. |
| **Evaluation** | Test ROC-AUC / PR-AUC; score latest Sunday; save `sunday_ranked_prospects.csv`. |
| **Alternative models: XGBoost** | Optuna hyperparameter tuning for XGBoost; compare LightGBM vs XGBoost (and optional ensemble); save XGB/ensemble CSVs. |
| **Playbook Segmentation** | For top N prospects: build prompt from row, call LLM, parse/validate JSON; save enriched taxonomy CSV. |

- **Minimum path:** Run through **Evaluation** to get the main ranked list.
- **XGBoost + tuning:** Run the **Alternative models** cell (requires Modeling + Evaluation first).
- **GenAI segmentation:** Run the **Playbook Segmentation** cell (requires `OPENAI_API_KEY` in `.env` and a ranked dataframe in memory).

---

## Outputs

- **sunday_ranked_prospects.csv** — All free portals for the latest Sunday, ranked by `P_CONVERT_30D` (LightGBM), with firmographics and usage columns.
- **sunday_ranked_prospects_xgb.csv** — Same idea, ranked by XGBoost (if that cell was run).
- **sunday_ranked_prospects_ensemble.csv** — Ranked by 0.5×LightGBM + 0.5×XGBoost (if that cell was run).
- **sunday_ranked_prospects_enriched_taxonomy.csv** — Top N rows with added columns: `behavior_pattern`, `likely_stage`, `playbook_focus`, `urgency`, `subject_line`, `opening_line` (from LLM).

Metrics printed in the notebook: **Test ROC-AUC**, **Test PR-AUC**, and (if Optuna is run) **best validation ROC-AUC** and best hyperparameters.

---

## Technology Stack

- **Data & ML:** pandas, numpy, scikit-learn, LightGBM, XGBoost, Optuna  
- **Viz:** matplotlib, seaborn  
- **GenAI:** OpenAI API (e.g. gpt-4o-mini), prompts in `prompts/`  
- **Env:** python-dotenv, `.env` for `OPENAI_API_KEY`

---

## License

Use and adapt as needed for your organization.
