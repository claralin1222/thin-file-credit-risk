"""
Feature preparation for the Home Credit modeling pipeline.

This module takes the raw joined+cleaned dataframe and produces a modeling-ready
feature set: ~35 base columns plus ~6 derived features. The output preserves NaN
where present (LightGBM handles it natively; WoE encoding will bin it as its
own category).

Features excluded from modeling (per EDA findings):
- 47 housing/apartment columns (high missingness, weak signal)
- 20 FLAG_DOCUMENT_* columns (weak individual signal)
- Social circle features (ethically questionable, weak signal)

Usage:
    from src.features import prepare_features
    
    df_features = prepare_features(df, include_ext_source_1=True)
"""

import numpy as np
import pandas as pd


# Feature groups — defined here so they're explicit and easy to audit

LOAN_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
]

DEMOGRAPHIC_FEATURES = [
    "CODE_GENDER",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

EXT_SOURCE_FEATURES = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

REGION_FEATURES = [
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "REGION_POPULATION_RELATIVE",
]

BUREAU_FEATURES = [
    "bureau_count",
    "bureau_active_count",
    "bureau_closed_count",
    "bureau_count_recent",
    "bureau_credit_sum_mean",
    "bureau_credit_sum_max",
    "bureau_credit_debt_mean",
    "bureau_overdue_max",
    "bureau_days_credit_mean",
    "bureau_days_credit_min",
    "is_thin_file",
]

INQUIRY_FEATURES = [
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# All base features combined
BASE_FEATURES = (
    LOAN_FEATURES
    + DEMOGRAPHIC_FEATURES
    + EXT_SOURCE_FEATURES
    + REGION_FEATURES
    + BUREAU_FEATURES
    + INQUIRY_FEATURES
)

# Identifiers (kept in output but not used as model features)
ID_COLUMNS = ["SK_ID_CURR", "TARGET"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio and transformed features that carry signal in credit risk modeling.
    
    Returns a copy of df with new columns added.
    """
    df = df.copy()

    # Loan structure ratios
    df["loan_to_income"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["payment_to_income"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["financing_premium"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # Age and tenure in years (more interpretable than negative days)
    df["age_years"] = -df["DAYS_BIRTH"] / 365
    df["employment_years"] = -df["DAYS_EMPLOYED"] / 365

    # Log-transformed income to handle the extreme outliers found in EDA
    df["income_log"] = np.log1p(df["AMT_INCOME_TOTAL"])

    return df


DERIVED_FEATURES = [
    "loan_to_income",
    "payment_to_income",
    "financing_premium",
    "age_years",
    "employment_years",
    "income_log",
]


def prepare_features(
    df: pd.DataFrame,
    include_ext_source_1: bool = True,
) -> pd.DataFrame:
    """
    Produce the modeling-ready feature set from the joined/cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_joined() and basic_clean() — the joined application+bureau data.
    include_ext_source_1 : bool, default True
        Whether to include EXT_SOURCE_1 as a feature. EXT_SOURCE_1 is known from EDA
        to embed heavy demographic structure (gender, age); excluding it allows
        comparison of model performance with and without this demographically-loaded
        feature.

    Returns
    -------
    pd.DataFrame
        ID columns + ~40 modeling features. Same row count as input. NaN preserved.
    """
    # Add derived features
    df = add_derived_features(df)

    # Build the feature column list
    feature_cols = BASE_FEATURES + DERIVED_FEATURES
    if not include_ext_source_1:
        feature_cols = [c for c in feature_cols if c != "EXT_SOURCE_1"]

    # Verify all expected columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Return ID columns + feature columns
    return df[ID_COLUMNS + feature_cols].copy()


def get_feature_columns(include_ext_source_1: bool = True) -> list[str]:
    """Return the list of feature columns (excludes IDs and TARGET)."""
    feature_cols = BASE_FEATURES + DERIVED_FEATURES
    if not include_ext_source_1:
        feature_cols = [c for c in feature_cols if c != "EXT_SOURCE_1"]
    return feature_cols