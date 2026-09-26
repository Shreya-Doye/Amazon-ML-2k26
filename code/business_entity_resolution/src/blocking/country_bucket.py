"""
Country Bucketing Module for Entity Resolution.
Partitions records into open-set country buckets without hardcoding specific country names.
"""

from typing import Dict, List, Optional
import pandas as pd


def country_key(country_str: str) -> str:
    """
    Normalize country label to open-set canonical key.
    Handles extra whitespace and case variations consistently.
    """
    if not isinstance(country_str, str):
        return "unknown"
    key = country_str.strip().casefold()
    return key if key else "unknown"


def partition_by_country(df: pd.DataFrame, country_col: str = "country") -> Dict[str, pd.DataFrame]:
    """
    Partition a DataFrame into a dictionary of DataFrames keyed by normalized country.
    """
    if country_col not in df.columns:
        raise ValueError(f"Column '{country_col}' not found in DataFrame.")

    df_copy = df.copy()
    df_copy["_norm_country"] = df_copy[country_col].apply(country_key)
    
    partitions = {}
    for c_key, group in df_copy.groupby("_norm_country"):
        partitions[c_key] = group.drop(columns=["_norm_country"])
        
    return partitions
