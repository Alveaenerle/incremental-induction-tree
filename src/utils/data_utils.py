import pandas as pd
import numpy as np


def discretize_dataset(df: pd.DataFrame, continuous_columns: list, bins=5) -> np.ndarray:
    df_clean = df.copy()
    for col in continuous_columns:
        col_bins = min(bins, df_clean[col].nunique())
        df_clean[col] = pd.qcut(df_clean[col], q=col_bins, labels=False)
    return df_clean.to_numpy()
