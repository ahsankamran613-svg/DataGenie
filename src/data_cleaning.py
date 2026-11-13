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

from src.stats_analyzer import is_binary_bool_indicator, is_id_or_code_column, tokenize_col_name


MISSING_TOKEN_PATTERN = r"(?i)^\s*(?:nan|null|none|n/a|na|-|--)?\s*$"
TIME_ONLY_PATTERN = r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[AaPp][Mm])?$"
MONTH_BUCKET_PATTERN = r"^\d{4}[-/]\d{1,2}$"
DATE_LIKE_PATTERN = (
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[T\s]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[AaPp][Mm])?)?"
    r"|\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|[A-Za-z]{3,9}[-\s]\d{1,2},?[-\s]\d{4}"
    r"|\d{1,2}[-\s][A-Za-z]{3,9},?[-\s]\d{4})$"
)
DATE_NAME_HINT_TOKENS: frozenset[str] = frozenset(
    {"date", "dt", "time", "day", "month", "created", "updated"}
)
MAX_TIME_ONLY_RATIO = 0.30
MIN_MONTH_BUCKET_RATIO = 0.50
MIN_DATE_HINT_RATIO = 0.50
MIN_DATE_NO_HINT_RATIO = 0.75
MIN_DATE_PARSED_RATIO = 0.80
MIN_NUMERIC_PARSED_RATIO = 0.80
ISO_DATE_FORMAT = "%Y-%m-%d"
ISO_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


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
    """Prevent ID/code columns (e.g., ORD-10001, SKU-200, zip_code) from being parsed as dates or floats."""
    if is_id_or_code_column(col_name):
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
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None

    # Unwrap accounting negatives ($120.00) -> -$120.00 before stripping leading currency symbols
    stripped = (
        series.astype("string")
        .str.strip()
        .str.replace(r"^([+-]?)\s*[\$€£¥]\s*", r"\1", regex=True)
        .str.replace(r"^\(\s*(.*?)\s*\)$", r"-\1", regex=True)
        .str.replace(r"^([+-]?)\s*[\$€£¥]\s*", r"\1", regex=True)
        .str.replace(r"\s*[%]$", "", regex=True)
        .str.replace(",", "", regex=False)
    )

    converted = pd.to_numeric(stripped, errors="coerce")
    valid_ratio = float(converted.notna().sum()) / float(len(non_null))
    if valid_ratio >= MIN_NUMERIC_PARSED_RATIO:
        return converted.astype("float64")
    return None


def _infer_dayfirst(values: pd.Series) -> bool:
    """Infer day-first vs month-first across slash/dash numeric dates in a column."""
    dd_first_evidence = 0
    mm_first_evidence = 0
    dash_count = 0
    slash_count = 0

    extracted = values.astype(str).str.extract(
        r"^(\d{1,2})([-/.])(\d{1,2})\2(\d{2,4})(?:\s.*)?$"
    )
    for _, row in extracted.dropna().iterrows():
        p1 = int(row[0])
        sep = str(row[1])
        p2 = int(row[2])
        if sep == "-":
            dash_count += 1
        elif sep == "/":
            slash_count += 1
        if p1 > 12 and 1 <= p2 <= 12:
            dd_first_evidence += 1
        elif p2 > 12 and 1 <= p1 <= 12:
            mm_first_evidence += 1

    if dd_first_evidence > mm_first_evidence:
        return True
    if mm_first_evidence > dd_first_evidence:
        return False
    return dash_count >= slash_count


def _parse_datetime_series(s_vals: pd.Series, *, dayfirst: bool) -> pd.Series:
    """Parse date/datetime strings safely, normalizing timezone-aware values to naive UTC."""
    try:
        parsed = pd.to_datetime(
            s_vals,
            format="mixed",
            dayfirst=dayfirst,
            errors="coerce",
        )
    except ValueError:
        parsed = pd.to_datetime(
            s_vals,
            format="mixed",
            dayfirst=dayfirst,
            utc=True,
            errors="coerce",
        )
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed.astype("datetime64[us]")


def _format_parsed_datetimes(parsed: pd.Series) -> pd.Series:
    """Format parsed datetimes as ISO date or ISO datetime when sub-day time components exist."""
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    non_nat = parsed.dropna()
    has_subday_time = bool(
        ((non_nat.dt.hour != 0) | (non_nat.dt.minute != 0) | (non_nat.dt.second != 0)).any()
    )
    fmt = ISO_DATETIME_FORMAT if has_subday_time else ISO_DATE_FORMAT
    return parsed.dt.strftime(fmt)


def _try_parse_mixed_dates(col_name: str, series: pd.Series) -> tuple[pd.Series, int] | None:
    """Detect and normalize mixed date/datetime strings into ISO strings."""
    col_tokens = set(tokenize_col_name(col_name))
    date_name_hint = bool(col_tokens & DATE_NAME_HINT_TOKENS)
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None

    # Do not convert time-only columns (e.g. "09:15") into today's date
    time_only_ratio = float(non_null.str.match(TIME_ONLY_PATTERN).mean())
    if time_only_ratio >= MAX_TIME_ONLY_RATIO:
        return None

    # Do not expand YYYY-MM month buckets into YYYY-MM-01 dates
    month_bucket_ratio = float(non_null.str.match(MONTH_BUCKET_PATTERN).mean())
    if month_bucket_ratio >= MIN_MONTH_BUCKET_RATIO:
        return None

    # Quick pattern check so arbitrary text columns aren't tested unnecessarily
    date_pattern_ratio = float(non_null.str.match(DATE_LIKE_PATTERN).mean())
    min_required = MIN_DATE_HINT_RATIO if date_name_hint else MIN_DATE_NO_HINT_RATIO
    if date_pattern_ratio < min_required:
        return None

    s_clean = series.astype("string").str.strip()
    iso_year_first_mask = s_clean.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}").fillna(False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[us]")

    if iso_year_first_mask.any():
        iso_vals = s_clean.loc[iso_year_first_mask].str.replace(
            r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})",
            r"\1-\2-\3",
            regex=True,
        )
        parsed.loc[iso_year_first_mask] = _parse_datetime_series(iso_vals, dayfirst=False)

    non_date_mask = (
        s_clean.str.match(TIME_ONLY_PATTERN).fillna(False)
        | s_clean.str.match(MONTH_BUCKET_PATTERN).fillna(False)
    )
    other_mask = (~iso_year_first_mask) & (~non_date_mask) & s_clean.notna() & (s_clean != "")
    if other_mask.any():
        inferred_dayfirst = _infer_dayfirst(s_clean.loc[other_mask])
        parsed.loc[other_mask] = _parse_datetime_series(
            s_clean.loc[other_mask],
            dayfirst=inferred_dayfirst,
        )

    valid_ratio = float(parsed.notna().sum()) / float(len(non_null))
    if valid_ratio < MIN_DATE_PARSED_RATIO:
        return None

    normalized = _format_parsed_datetimes(parsed)
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
            df[col] = _format_parsed_datetimes(df[col])
            date_normalized.append(col)
            continue

        if not _is_text_series(df[col]):
            continue

        # Replace blank/whitespace-only strings and case-insensitive 'null'/'N/A'/'None' tokens with NA
        cleaned_tokens = (
            df[col]
            .astype("string")
            .replace(MISSING_TOKEN_PATTERN, pd.NA, regex=True)
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
            has_time_fmt = bool(norm_dates.dropna().astype(str).str.contains(":").any())
            iso_fmt_label = "ISO YYYY-MM-DD HH:MM:SS" if has_time_fmt else "ISO YYYY-MM-DD"
            actions.append(
                CleaningAction(
                    icon="📅",
                    category="dates",
                    message=f"Normalized mixed date formats in `{col}` to {iso_fmt_label}",
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

        if pd.api.types.is_bool_dtype(df[col]):
            mode_series = df[col].mode(dropna=True)
            fill_bool = bool(mode_series.iloc[0]) if not mode_series.empty else False
            df[col] = df[col].fillna(fill_bool)
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Filled {null_count} missing boolean values in `{col}` with mode ({fill_bool})",
                    affected_count=null_count,
                )
            )
        elif is_binary_bool_indicator(col, df[col]):
            mode_series = df[col].mode(dropna=True)
            fill_flag = int(mode_series.iloc[0]) if not mode_series.empty else 0
            df[col] = df[col].fillna(fill_flag)
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Filled {null_count} missing boolean indicator values in `{col}` with mode ({fill_flag})",
                    affected_count=null_count,
                )
            )
        elif _looks_like_id_column(col, df[col]):
            if _is_text_series(df[col]):
                df[col] = df[col].fillna("Unknown")
                actions.append(
                    CleaningAction(
                        icon="🩹",
                        category="imputation",
                        message=f"Filled {null_count} missing identifier values in `{col}` with 'Unknown'",
                        affected_count=null_count,
                    )
                )
            else:
                actions.append(
                    CleaningAction(
                        icon="🆔",
                        category="imputation",
                        message=f"Preserved {null_count} missing values in identifier column `{col}` as NULL (skipped median imputation)",
                        affected_count=null_count,
                    )
                )
        elif pd.api.types.is_numeric_dtype(df[col]):
            median_val = float(df[col].median()) if df[col].notna().any() else 0.0
            if np.isnan(median_val):
                median_val = 0.0
            if pd.api.types.is_integer_dtype(df[col]):
                fill_val: int | float = int(round(median_val))
                fill_label = str(fill_val)
            else:
                fill_val = round(median_val, 2)
                fill_label = f"{fill_val:.2f}"
            df[col] = df[col].fillna(fill_val)
            actions.append(
                CleaningAction(
                    icon="🩹",
                    category="imputation",
                    message=f"Imputed {null_count} missing values in `{col}` with median ({fill_label})",
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
