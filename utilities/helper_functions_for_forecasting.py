from typing import Union

import pandas as pd
import numpy as np

class HelperFunctions:
    @staticmethod
    def parse_employee_range_to_midpoint(val: Union[str, float, int]) -> float:
        """
        Converts EMPLOYEE_RANGE like "51-200" -> 125.5, "1000+" -> 1000, etc.
        If it can't parse, returns NaN.
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s:
            return np.nan

        # Handle "1000+" style
        if s.endswith("+"):
            num = s[:-1].replace(",", "").strip()
            try:
                return float(num)
            except Exception:
                return np.nan

        # Handle "51-200" or "51 - 200" style
        if "-" in s:
            parts = s.replace(",", "").split("-", 1)
            if len(parts) == 2:
                try:
                    lo = float(parts[0].strip())
                    hi = float(parts[1].strip())
                    return (lo + hi) / 2.0
                except Exception:
                    pass

        # Handle "2 to 5", "26 to 50" style
        if " to " in s.lower():
            parts = s.replace(",", "").split(" to ", 1)
            if len(parts) == 2:
                try:
                    lo = float(parts[0].strip())
                    hi = float(parts[1].strip())
                    return (lo + hi) / 2.0
                except Exception:
                    pass

        # Handle single number
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return np.nan

    @staticmethod
    def make_sunday_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Build a Sunday date index between start and end (inclusive).
        """
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        # W-SUN produces Sundays
        return pd.date_range(start=start, end=end, freq="W-SUN")

    @staticmethod
    def safe_log1p(x: pd.Series) -> pd.Series:
        return np.log1p(np.clip(x.astype(float), a_min=0, a_max=None))

    @staticmethod
    def complete_daily_index(df: pd.DataFrame, agg_cols: list) -> pd.DataFrame:
        cid = df["id"].iloc[0]
        dmin = df["DATE"].min()
        dmax = df["DATE"].max()
        full = pd.DataFrame({"DATE": pd.date_range(dmin, dmax, freq="D")})
        full["id"] = cid
        out = full.merge(df, on=["id", "DATE"], how="left")
        out[agg_cols] = out[agg_cols].fillna(0.0)
        return out

    @staticmethod
    def days_since_last_activity(company_id: Union[int, str], snapshot_date: pd.Timestamp, last_active: dict) -> float:
        dates = last_active.get(company_id, [])
        # find the most recent activity <= snapshot_date
        # dates are Timestamps; list may be long, but this is fine for small/medium datasets
        prior = [d for d in dates if d <= snapshot_date]
        if not prior:
            return np.nan
        return float((snapshot_date - max(prior)).days)

# Instantiate the class, as requested.
helper_functions_obj = HelperFunctions()
