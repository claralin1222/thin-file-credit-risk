"""
Data loading and joining for Home Credit Default Risk.

The project uses two tables: application_train (one row per application)
and bureau (one row per prior credit reported to credit bureaus).
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_applications(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the main application table."""
    df = pd.read_csv(data_dir / "application_train.csv")
    return df


def load_bureau(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the bureau table (prior credits at other institutions)."""
    df = pd.read_csv(data_dir / "bureau.csv")
    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the bureau table to one row per SK_ID_CURR.

    The bureau table has multiple rows per applicant (one per prior credit).
    We aggregate to numeric features the application-level model can use.

    Key features for the thin-file framing:
    - bureau_count: total number of prior credits
    - bureau_active_count: number currently active
    - bureau_count_recent: prior credits in last 2 years
    These let us define thin-file at the model level rather than guessing.
    """
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=(
            "CREDIT_ACTIVE",
            lambda x: (x == "Active").sum(),
        ),
        bureau_closed_count=(
            "CREDIT_ACTIVE",
            lambda x: (x == "Closed").sum(),
        ),
        bureau_credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_credit_sum_max=("AMT_CREDIT_SUM", "max"),
        bureau_credit_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_overdue_max=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_days_credit_mean=("DAYS_CREDIT", "mean"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),  # most recent
    )

    # Recent activity: prior credits opened in last 2 years (730 days)
    recent = (
        bureau[bureau["DAYS_CREDIT"] >= -730]
        .groupby("SK_ID_CURR")
        .size()
        .rename("bureau_count_recent")
    )
    agg = agg.join(recent, how="left")
    agg["bureau_count_recent"] = agg["bureau_count_recent"].fillna(0)

    return agg.reset_index()


def load_joined(
    data_dir: Path = DATA_DIR,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load applications joined with aggregated bureau features.

    Applicants with no bureau records get NaN for bureau aggregates,
    which we fill with 0 for counts and leave NaN for amount features
    (NaN is informative — it means thin-file).
    """
    cache_path = PROCESSED_DIR / "applications_with_bureau.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    apps = load_applications(data_dir)
    bureau = load_bureau(data_dir)
    bureau_agg = aggregate_bureau(bureau)

    joined = apps.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Count features: NaN means no bureau record, which is 0
    count_cols = [
        "bureau_count",
        "bureau_active_count",
        "bureau_closed_count",
        "bureau_count_recent",
    ]
    joined[count_cols] = joined[count_cols].fillna(0)

    # Add a thin-file flag (applicants with zero or one prior bureau record, essentially no track record) for downstream segmentation
    joined["is_thin_file"] = (joined["bureau_count"] <= 1).astype(int)

    # Defragment after multiple in-place column modifications
    joined = joined.copy()

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(cache_path, index=False)

    return joined


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply known data quirks documented in the Home Credit dataset.

    - DAYS_EMPLOYED has a sentinel value of 365243 (~1000 years) for
      retired/unemployed applicants. Replace with NaN.
    - CODE_GENDER has a small number of 'XNA' values; treat as NaN.
    """
    df = df.copy()
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan
    return df