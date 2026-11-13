"""Order rows for exact comparisons without converting their values."""
import pandas as pd


def canonical_row_order(frame):
    # Do not use astype(str): pandas 3 keeps missing scalars as NaN there, and
    # in pandas 2 it turns them into the string "nan". Tagged scalar keys keep
    # missing values distinct from literal strings and avoid delimiter collisions.
    def key(value):
        return ("missing", "") if pd.isna(value) else (type(value).__name__, str(value))
    order = sorted(range(len(frame)), key=lambda i: tuple(
        key(value) for value in frame.iloc[i]))
    return frame.iloc[order].reset_index(drop=True)
