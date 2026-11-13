"""Order rows for exact comparisons without converting their values."""
import pandas as pd


def canonical_row_order(frame):
    # astype(str) preserves missing scalars in pandas 3. Tagged scalar keys
    # distinguish them from literal strings and avoid delimiter collisions.
    def key(value):
        return ("missing", "") if pd.isna(value) else (type(value).__name__, str(value))
    order = sorted(range(len(frame)), key=lambda i: tuple(
        key(value) for value in frame.iloc[i]))
    return frame.iloc[order].reset_index(drop=True)
