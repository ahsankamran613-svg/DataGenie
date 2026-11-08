"""Programmatic Pandas 3.x Data Cleaning Pipeline for DataGenie.

Transforms messy user-uploaded or built-in demo CSVs into SQL-ready DataFrames while
recording a transparent Before/After CleaningReport for Tab 2 ("Data Health & Auto-EDA").
Fully compatible with Pandas 3.x Copy-on-Write and StringDtype (`str`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd


@dataclass
class CleaningAction:
    """Single logged data-cleaning step displayed in the UI action log."""

    icon: str
    category: str
    message: str
    affected_count: int


@dataclass
class CleaningReport:
    """Structured Before/After report produced by `clean_dataframe`."""

    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_before: int
    missing_after: int
    columns_standardized: int
    column_mapping: dict[str, str] = field(default_factory=dict)
    numeric_columns_converted: list[str] = field(default_factory=list)
    date_columns_normalized: list[str] = field(default_factory=list)
    categorical_columns_cleaned: list[str] = field(default_factory=list)
    actions: list[CleaningAction] = field(default_factory=list)


def to_snake_case(name: str) -> str:
    """Convert messy column headers (e.g. 'Revenue ($)', 'Discount (%)') to SQL-safe snake_case."""
    cleaned = str(name).strip()
    # Replace common unit symbols with descriptive suffixes before stripping punctuation
    cleaned = re.sub(r"\(\s*%\s*\)", "_pct", cleaned)
    cleaned = re.sub(r"%", "_pct", cleaned)
    cleaned = re.sub(r"\(\s*\$\s*\)", "", cleaned)
    cleaned = re.sub(r"\$", "", cleaned)
    # Handle camelCase -> snake_case
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
    # Replace non-alphanumeric sequences with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    if not cleaned:
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def _is_text_series(series: pd.Series) -> bool:
    """Pandas 2.x (`object`) and Pandas 3.x (`str` / `StringDtype`) safe text column check."""
    return bool(
        pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype)
    )


def _looks_like_id_column(col_name: str, series: pd.Series) -> bool:
    """Prevent ID/code columns (e.g., ORD-10001, SKU-200) from being parsed as dates or floats."""
    if col_name == "id" or col_name.endswith("_id") or col_name.endswith("_code"):
        return True
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return False
    # Check if values look like alphanumeric codes e.g. ORD-10001
    code_match_ratio = non_null.str.match(r"^[A-Za-z]{2,6}-\d+$").mean()
    return float(code_match_ratio) >= 0.7


def _try_convert_dirty_numeric(series: pd.Series) -> pd.Series | None:
    """Strip currency/percentage/comma symbols from text series and convert to float if >=80% valid."""
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    # Must contain digits and mostly numeric/currency characters
    stripped = (
        series.astype("string")
        .str.strip()
        .str.replace(r"^[\$€£¥]\s*", "", regex=True)
        .str.replace(r"\s*[%]$", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # accounting negatives ($120.00) -> -120.00
    )

    converted = pd.to_numeric(stripped, errors="coerce")
    valid_ratio = float(converted.notna().sum()) / float(len(non_null))
    if valid_ratio >= 0.80:
        return converted.astype("float64")
    return None


def _try_parse_mixed_dates(col_name: str, series: pd.Series) -> tuple[pd.Series, int] | None:
    """Detect and normalize mixed date strings into ISO `YYYY-MM-DD` strings."""
    date_name_hint = any(
        token in col_name for token in ("date", "dt", "time", "day", "month", "created", "updated")
    )
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    # Quick pattern check so arbitrary text columns aren't tested unnecessarily
    date_pattern_ratio = float(
        non_null.str.match(
            r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
        ).mean()
    )
    if not date_name_hint and date_pattern_ratio < 0.75:
        return None
    if date_pattern_ratio < 0.50 and not date_name_hint:
        return None

    s_clean = series.astype("string").str.strip()
    iso_year_first_mask = s_clean.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").fillna(False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if iso_year_first_mask.any():
        parsed_year_first = pd.to_datetime(
            s_clean.loc[iso_year_first_mask].str.replace("/", "-", regex=False),
            format="mixed",
            dayfirst=False,
            errors="coerce",
        )
        parsed.loc[iso_year_first_mask] = parsed_year_first

    other_mask = (~iso_year_first_mask) & s_clean.notna() & (s_clean != "")
    if other_mask.any():
        parsed_other = pd.to_datetime(
            s_clean.loc[other_mask],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        parsed.loc[other_mask] = parsed_other

    valid_ratio = float(parsed.notna().sum()) / float(len(non_null))
    if valid_ratio < 0.80:
        return None

    normalized = parsed.dt.strftime("%Y-%m-%d")
    changed_count = int((normalized.notna() & (normalized != s_clean)).sum())
    return normalized, changed_count


def _standardize_categorical_text(col_name: str, series: pd.Series) -> tuple[pd.Series, int]:
    """Trim extra whitespace and unify inconsistent casing on low/medium cardinality text columns."""
    s_str = series.astype("string")
    trimmed = s_str.str.strip().str.replace(r"\s+", " ", regex=True)

    if _looks_like_id_column(col_name, trimmed):
        changed = int((trimmed.notna() & (trimmed != s_str)).sum())
        return trimmed, changed

    non_null = trimmed.dropna()
    if non_null.empty:
        return trimmed, 0

    # Group values that differ only by case and pick Clean Title Case (preserving '&', acronyms <=3 chars)
    lower_keys = non_null.str.lower()
    unique_lower = lower_keys.nunique()
    unique_exact = non_null.nunique()

    # Apply casing normalization when categories collide across cases or cardinality is categorical
    if unique_lower < unique_exact or unique_lower <= 50:
        canonical_map: dict[str, str] = {}
        for val in non_null.unique():
            val_str = str(val)
            key = val_str.lower()
            if key not in canonical_map:
                words = val_str.split(" ")
                titled_words = [
                    w.upper() if (len(w) <= 3 and w.isalpha() and w.isupper()) else (w if w == "&" else w.capitalize())
                    for w in words
                ]
                canonical_map[key] = " ".join(titled_words)

        standardized = trimmed.map(
            lambda x: canonical_map.get(str(x).lower(), str(x)) if pd.notna(x) else pd.NA
        ).astype("string")
    else:
        standardized = trimmed

    changed = int((standardized.notna() & (standardized != s_str)).sum())
    return standardized, changed


def clean_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full Pandas 3.x automated cleaning pipeline and return `(clean_df, report)`."""
    df = raw_df.copy()

    rows_before = int(len(df))
    missing_before = int(df.isna().sum().sum())
    actions: list[CleaningAction] = []

    # 1. Standardize column headers to unique SQL-safe snake_case
    column_mapping: dict[str, str] = {}
    seen_names: dict[str, int] = {}
    new_columns: list[str] = []
    columns_standardized = 0

    for raw_col in df.columns:
        base_snake = to_snake_case(str(raw_col))
        if base_snake in seen_names:
            seen_names[base_snake] += 1
            final_col = f"{base_snake}_{seen_names[base_snake]}"
        else:
            seen_names[base_snake] = 0
            final_col = base_snake

        column_mapping[str(raw_col)] = final_col
        new_columns.append(final_col)
        if str(raw_col) != final_col:
            columns_standardized += 1

    df.columns = new_columns
    if columns_standardized > 0:
        actions.append(
            CleaningAction(
                icon="🏷️",
                category="schema",
                message=f"Standardized {columns_standardized} column headers to SQL-safe snake_case",
                affected_count=columns_standardized,
            )
        )

    # 2. Remove exact duplicate rows BEFORE type conversion & after?
    # Note: removing exact raw duplicates first catches exact row copies immediately,
    # and we also check post-normalization duplicates just in case!
    dup_before_norm = int(df.duplicated().sum())
    if dup_before_norm > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    numeric_converted: list[str] = []
    date_normalized: list[str] = []
    categorical_cleaned: list[str] = []

    # 3. Inspect text columns for dirty dates, dirty currency/percent numbers, or messy categories
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
            date_normalized.append(col)
            continue

        if not _is_text_series(df[col]):
            continue

        # Replace blank/whitespace-only strings and 'null'/'N/A' tokens with NA
        cleaned_tokens = (
            df[col]
            .astype("string")
            .replace(r"^\s*(?:nan|null|none|n/a|na|-|--)?\s*$", pd.NA, regex=True)
        )
        df[col] = cleaned_tokens

        if _looks_like_id_column(col, df[col]):
            std_series, changed = _standardize_categorical_text(col, df[col])
            df[col] = std_series
            continue

        # 3a. Try date parsing
        date_result = _try_parse_mixed_dates(col, df[col])
        if date_result is not None:
            norm_dates, changed_dates = date_result
            df[col] = norm_dates
            date_normalized.append(col)
            actions.append(
                CleaningAction(
                    icon="📅",
                    category="dates",
                    message=f"Normalized mixed date formats in `{col}` to ISO YYYY-MM-DD",
                    affected_count=max(changed_dates, int(norm_dates.notna().sum())),
                )
            )
            continue

        # 3b. Try numeric conversion (currency $, commas, % signs)
        num_series = _try_convert_dirty_numeric(df[col])
        if num_series is not None:
            affected = int(num_series.notna().sum())
            df[col] = num_series
            numeric_converted.append(col)
            actions.append(
                CleaningAction(
                    icon="💲",
                    category="numeric",
                    message=f"Converted `{col}` from formatted currency/percentage text to float",
                    affected_count=affected,
                )
            )
            continue

        # 3c. Clean categorical whitespace & casing (" north " / "NORTH" -> "North")
        std_cat, changed_cat = _standardize_categorical_text(col, df[col])
        df[col] = std_cat
        if changed_cat > 0:
            categorical_cleaned.append(col)
            actions.append(
                CleaningAction(
                    icon="🔤",
                    category="categorical",
                    message=f"Trimmed whitespace & unified category casing in `{col}`",
                    affected_count=changed_cat,
                )
            )

    # Check if normalizing text/dates exposed any additional duplicate rows
    dup_after_norm = int(df.duplicated().sum())
    if dup_after_norm > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    total_duplicates_removed = dup_before_norm + dup_after_norm
    if total_duplicates_removed > 0:
        actions.insert(
            0,
            CleaningAction(
                icon="🧽",
                category="duplicates",
                message=f"Removed {total_duplicates_removed} duplicate rows",
                affected_count=total_duplicates_removed,
            ),
        )

    # 4. Impute missing values (median for numeric, 'Unknown' for categorical/date text)
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = float(df[col].median()) if df[col].notna().any() else 0.0
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(round(median_val, 2))
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Imputed {null_count} missing values in `{col}` with median ({median_val:.2f})",
                    affected_count=null_count,
                )
            )
        elif col in date_normalized:
            mode_series = df[col].mode(dropna=True)
            fill_date = str(mode_series.iloc[0]) if not mode_series.empty else "1970-01-01"
            df[col] = df[col].fillna(fill_date)
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Filled {null_count} missing dates in `{col}` with modal date ({fill_date})",
                    affected_count=null_count,
                )
            )
        else:
            df[col] = df[col].fillna("Unknown")
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Filled {null_count} missing categorical values in `{col}` with 'Unknown'",
                    affected_count=null_count,
                )
            )

    rows_after = int(len(df))
    missing_after = int(df.isna().sum().sum())

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=rows_after,
        duplicates_removed=total_duplicates_removed,
        missing_before=missing_before,
        missing_after=missing_after,
        columns_standardized=columns_standardized,
        column_mapping=column_mapping,
        numeric_columns_converted=numeric_converted,
        date_columns_normalized=date_normalized,
        categorical_columns_cleaned=categorical_cleaned,
        actions=actions,
    )
    return df, report
