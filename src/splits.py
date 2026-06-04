"""
Train/validation/test split for the Home Credit modeling project.

Strategy:
- Stratified on TARGET to preserve the 8% default rate across all splits
- 70/15/15 split with fixed random_state for reproducibility
- Indices are cached to disk so the same split is used across all notebooks

Usage:
    from src.splits import get_splits

    train_idx, val_idx, test_idx = get_splits(df)
    df_train = df.loc[train_idx]
    df_val   = df.loc[val_idx]
    df_test  = df.loc[test_idx]
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"

RANDOM_STATE = 42


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create stratified train/val/test indices and return as a DataFrame with one column.
    
    The single column 'split' has values 'train', 'val', or 'test'.
    Indexed by the original df's index (which is SK_ID_CURR-equivalent positional).
    """
    # First split off the test set
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["TARGET"],
        random_state=random_state,
    )

    # Now split the remaining into train and val
    # Adjust val_size to be relative to the remaining (train + val), not the full df
    val_size_adjusted = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        stratify=df.loc[train_val_idx, "TARGET"],
        random_state=random_state,
    )

    # Build the split DataFrame
    splits = pd.Series(index=df.index, dtype="object", name="split")
    splits.loc[train_idx] = "train"
    splits.loc[val_idx] = "val"
    splits.loc[test_idx] = "test"

    return splits.to_frame()


def get_splits(df: pd.DataFrame, cache: bool = True) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Get train/val/test indices, computing and caching if needed.

    Returns three pandas Index objects: (train_idx, val_idx, test_idx).
    """
    if cache and SPLITS_PATH.exists():
        splits = pd.read_parquet(SPLITS_PATH)
    else:
        splits = make_splits(df)
        if cache:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            splits.to_parquet(SPLITS_PATH)

    train_idx = splits.index[splits["split"] == "train"]
    val_idx = splits.index[splits["split"] == "val"]
    test_idx = splits.index[splits["split"] == "test"]

    return train_idx, val_idx, test_idx


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    """Print a sanity-check summary of the splits."""
    train_idx, val_idx, test_idx = get_splits(df)
    report = pd.DataFrame({
        "n_rows": [len(train_idx), len(val_idx), len(test_idx)],
        "pct": [len(train_idx) / len(df), len(val_idx) / len(df), len(test_idx) / len(df)],
        "default_rate": [
            df.loc[train_idx, "TARGET"].mean(),
            df.loc[val_idx, "TARGET"].mean(),
            df.loc[test_idx, "TARGET"].mean(),
        ],
    }, index=["train", "val", "test"]).round(4)
    return report