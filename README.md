# Credit Risk Modeling for Thin-File Borrowers: Profitability, Fair Lending, and Explainability

## Project positioning

A mid-sized consumer lender serving near-prime and thin-file borrowers needs a default risk model to support approve/decline and pricing decisions. The model must satisfy three constraints simultaneously: expected loss minimization at the portfolio level, fair lending compliance with documented disparate impact testing, and individual-level explainability for adverse action notices. This project works through the threshold selection decision as a negotiation among three stakeholders — Chief Credit Officer, Fair Lending Compliance, and Head of Growth — and demonstrates how the same model can yield different business outcomes depending on how that negotiation resolves.

## Project Structure

```
thin-file-credit-risk/
├── README.md                      
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_and_model.ipynb
│   ├── 03_calibration_and_segments.ipynb
│   ├── 04_threshold_and_costs.ipynb
│   ├── 05_fair_lending.ipynb
│   └── 06_shap_adverse_action.ipynb
├── src/
│   ├── data.py                    # loading, joining, cleaning
│   ├── features.py                # WoE, feature engineering
│   ├── models.py                  # train/predict wrappers
│   ├── evaluation.py              # cost curves, calibration, fairness metrics
│   └── explain.py                 # SHAP → adverse action reasons
├── reports/
│   └── figures/                   
└── requirements.txt