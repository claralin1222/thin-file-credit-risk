# Credit Risk Modeling for Thin-File Borrowers

**Portfolio project: cost-sensitive threshold selection, fair lending, and Reg B-compliant explainability for a non-prime consumer lender.**

Built by [Clara Lin](https://linkedin.com/in/lin-sufang-clara) — MSBA (UIC, Dec 2025), Data Scientist targeting credit risk and fraud analytics roles.

---

## What this project is

A production-shaped credit risk model built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (~307k loan applications, 8% default rate). The scope goes beyond training a model to AUC: it works through the decisions a credit risk team actually has to make once a model exists.

Three questions drive the project, each mapped to a decision-maker in a real lending organization:

1. **Where does the approve/decline threshold go?** (Chief Credit Officer) — Cost-sensitive threshold analysis under asymmetric FN vs FP costs.
2. **Does the model treat protected classes fairly?** (Fair Lending Compliance) — Four-fifths rule, error rate parity, and an ablation study of the demographically-loaded EXT_SOURCE_1 feature.
3. **How do we explain a decline to the applicant?** (Regulatory / Adverse Action) — SHAP-based principal reasons for denial, translated into plain English.

## Why this framing

Most credit risk portfolio projects stop at "I trained a model and got AUC X." That's not what the job is. The job is deciding what threshold to deploy, defending the model to a regulator, and giving applicants the specific reasons they were denied. This project engages with each of those substantively.

The framing draws on my summer at the UChicago Crime Lab, where I validated a pretrial risk assessment pipeline across 1.8M+ case records -- a domain where the same asymmetric-cost, protected-class, individual-explainability questions apply, just with different stakeholders.

## Key findings

- **LightGBM beats the WoE-encoded logistic regression baseline by ~1.5 AUC points** (0.7626 vs 0.7462), meaningful but not transformative. Both models are well-calibrated after minimal work.
- **The cost-optimal threshold varies from 0.09 to 0.33** across defensible FN:FP ratios (2:1 to 10:1). The threshold isn't a modeling decision -- it's a business decision.
- **The chosen operating point (FN:FP = 5:1, τ = 0.168, 87.7% approval) fails four-fifths on age.** The 20-25 band has an approval ratio of 0.667 vs the 62+ reference, well below the 0.80 floor. Gender passes comfortably.
- **The age disparity is structural, not driven by any single feature.** An ablation of EXT_SOURCE_1 (the external score with heavy demographic structure identified in EDA) improved the 20-25 ratio only from 0.667 to 0.704 at a 0.006 AUC cost. Meaningful but not sufficient.
- **Segment modeling for the thin-file population didn't help.** A LightGBM trained only on the 26% of applicants with ≤1 bureau records performed identically to the pooled model with `is_thin_file` as a feature. The pooled model is already segment-aware through LightGBM's native NaN handling.
- **SHAP-based adverse action notices work.** For a random declined applicant, the top-3 SHAP contributions map to specific reasons ("EXT_SOURCE_2 is low", "employment tenure is short", "loan amount vs. price of goods financed is high") that meet Reg B's "specific principal reasons" requirement.

## Methods

- **Data**: Home Credit's `application_train.csv` joined with aggregated `bureau.csv` features. Thin-file segment defined as `bureau_count ≤ 1` (~26% of book, 1.24x default lift).
- **Baseline**: L2-regularized logistic regression on WoE-encoded features (via `optbinning`). Interpretable as a points-based scorecard.
- **Main model**: LightGBM with 8 monotonic constraints on domain-defensible features, isotonic-regression-calibrated post-hoc.
- **Threshold analysis**: Explicit cost matrix (FN cost = principal × LGD × timing factor; FP cost = foregone profit). Cost curves computed across FN:FP ratios from 2:1 to 10:1.
- **Fair lending**: Four-fifths rule and equalized-odds analysis by gender and age. Sensitivity analysis at both primary (5:1) and stricter (8:1) thresholds. EXT_SOURCE_1 ablation to isolate feature-specific contribution to disparity.
- **Explainability**: TreeSHAP for per-applicant attributions; feature-to-plain-English translator produces notice-ready reasons.

## What this project doesn't do

- No Reject Inference (would require multi-week effort and more data).
- No fairness-aware post-processing (mentioned as a future step for closing the four-fifths gap on age).
- No race / ethnicity / national origin analysis -- not available in Home Credit data. A US production fair lending audit would include these.
- No hyperparameter tuning to chase max AUC. The project optimizes for framing and credit-risk methodology, not last-mile performance.

## Repo structure
thin-file-credit-risk/
├── notebooks/
│   ├── 01_eda.ipynb                        # Stakeholder-framed exploratory analysis
│   ├── 02_baseline_and_model.ipynb         # WoE+LR baseline and LightGBM challenger
│   ├── 03_calibration_and_segments.ipynb   # Calibration, monotonic constraints, segment model evaluation
│   ├── 04_threshold_and_costs.ipynb        # Cost-sensitive threshold analysis
│   ├── 05_fair_lending.ipynb               # Disparity analysis + EXT_SOURCE_1 ablation
│   └── 06_shap_adverse_action.ipynb        # SHAP-based Reg B principal reasons
├── src/
│   ├── data.py                             # Loading, joining, cleaning
│   ├── features.py                         # Feature preparation with EXT_SOURCE_1 toggle
│   └── splits.py                           # Stratified train/val/test with cached indices
├── data/                                   # Gitignored; raw data downloaded via Kaggle API
├── requirements.txt
├── requirements-dev.txt                    # Dev tools (nbstripout)
└── pyproject.toml                          # Editable package install

## Reproducing

Requires Python 3.12+ and a Kaggle account with API credentials configured.

```bash
git clone https://github.com/YOUR-USERNAME/thin-file-credit-risk
cd thin-file-credit-risk

# Set up environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Download data (requires Kaggle API credentials configured)
mkdir -p data/raw
cd data/raw
kaggle competitions download -c home-credit-default-risk
unzip -q home-credit-default-risk.zip
cd ../..

# Run notebooks in order
jupyter lab notebooks/
```

Notebook outputs are stripped by [nbstripout](https://github.com/kynan/nbstripout) during development; the shipped notebooks include outputs so results are visible without re-running.

## Blog post

*(link to blog post once published)*

## Contact

- LinkedIn: [Clara Lin](https://linkedin.com/in/lin-sufang-clara)
- Email: clarasflin@gmail.com

---

*Data source: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk), a Kaggle competition from Home Credit Group. This is a portfolio project, not affiliated with or endorsed by Home Credit Group.*