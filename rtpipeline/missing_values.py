"""Pandas-version-neutral missing-value handling for row-derived metadata.

Under pandas 2.x a parquet round trip of a column that mixes present values
with absent ones comes back as ``None``; under pandas 3.x the same round
trip comes back as float ``nan`` (and nullable dtypes may yield ``pd.NA``
or ``pd.NaT``). ``float('nan')`` is truthy, so the common ``row.get(key)
or default`` idiom silently turns a missing reason/status/code into the
string ``'nan'`` instead of applying the default.

The helpers here centralize the check: treat ``None``, float ``nan``,
``pd.NA`` and ``pd.NaT`` as missing, and pass every other value through
unchanged. Call sites that read ledger, disposition, reason-code, status
or provenance fields out of DataFrame-derived dict rows (``to_dict``,
``iterrows``, ``itertuples``, ``read_parquet`` results) must use
:func:`value_or` / :func:`text_or` instead of a bare ``or`` default or a
bare ``str(...)``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


def is_missing_value(value: Any) -> bool:
    """Return True for None / NaN / NA / NaT; False for containers and arrays."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, complex):
        return math.isnan(value.real) or math.isnan(value.imag)
    if isinstance(value, (dict, list, set, tuple)):
        return False
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return False
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False
    if isinstance(value, np.generic):
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def value_or(value: Any, default: Any = "") -> Any:
    """Return ``default`` when ``value`` is missing, else ``value`` unchanged."""
    return default if is_missing_value(value) else value


def text_or(mapping: Mapping[str, Any], key: str, default: str = "") -> str:
    """Read ``mapping[key]`` as text, returning ``default`` for missing values.

    Unlike ``str(mapping.get(key) or default)``, a pandas-3 ``nan``
    round-trip of an absent cell yields ``default`` rather than ``'nan'``.
    """
    try:
        value = mapping.get(key, default)
    except AttributeError:
        value = default
    if is_missing_value(value):
        return default
    text = str(value)
    if text == "nan":
        # Defensive: a value that was already stringified from NaN upstream.
        return default
    return text
