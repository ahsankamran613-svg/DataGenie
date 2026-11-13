"""Deterministic Python Statistical Analyzer & Crash-Free Plotly Chart Builder.

Ensures numerical accuracy (totals, averages, contributor shares, period deltas,
and Z-score outliers are computed in Python rather than hallucinated by an LLM)
and renders clean, theme-adaptive (light/dark) interactive Plotly charts from a Pydantic `ChartSpec`.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Literal
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field


# Dark Theme Palette: Oceanic Midnight Navy + Luminous Coral + Electric Sky Blue
CHART_PALETTE = [
    "#FF6B6B",  # Luminous Coral
    "#38BDF8",  # Electric Sky Blue
    "#34D399",  # Fresh Emerald
    "#FBBF24",  # Amber Sun
    "#A78BFA",  # Iris Violet
    "#F472B6",  # Punch Pink
]

COLOR_PRIMARY = "#FF6B6B"
COLOR_SECONDARY = "#38BDF8"
COLOR_GRID = "rgba(148, 163, 184, 0.18)"
COLOR_MUTED = "#94A3B8"

# Column-classification, unit-formatting, and visualization configuration constants
TIME_DIMENSION_WORDS: frozenset[str] = frozenset(
    {"date", "month", "year", "week", "quarter", "period", "yr", "qtr", "fiscal"}
)
DURATION_METRIC_WORDS: frozenset[str] = frozenset(
    {
        "delay",
        "delays",
        "duration",
        "tenure",
        "age",
        "lead",
        "latency",
        "days",
        "months",
        "years",
        "weeks",
        "hours",
        "minutes",
        "seconds",
        "warranty",
        "experience",
        "grace",
        "trial",
        "elapsed",
        "wait",
        "cycle",
        "turnaround",
        "processing",
        "delivery",
        "shipping",
        "payback",
        "stay",
    }
)
ID_CODE_EXACT_NAMES: frozenset[str] = frozenset(
    {"id", "code", "zip", "zipcode", "postcode", "postal_code", "zip_code", "sku", "upc", "isbn"}
)
ID_CODE_SUFFIX_TOKENS: frozenset[str] = frozenset(
    {"id", "code", "zipcode", "postcode", "sku", "upc", "isbn", "key"}
)
ID_CODE_PREFIX_TOKENS: frozenset[str] = frozenset({"zip", "zipcode", "postcode", "postal"})
BOOL_INDICATOR_PREFIX_TOKENS: frozenset[str] = frozenset({"is", "has", "was", "can", "should"})
BOOL_INDICATOR_SUFFIX_TOKENS: frozenset[str] = frozenset({"flag", "bool"})
PERCENT_UNIT_TOKENS: frozenset[str] = frozenset(
    {"pct", "percent", "percentage", "margin", "rate", "share", "ratio", "ctr", "cvr", "roi", "roas"}
)
EXPLICIT_PERCENT_HEAD_TOKENS: frozenset[str] = frozenset(
    {"pct", "percent", "percentage", "share"}
)
STRICT_COUNT_TOKENS: frozenset[str] = frozenset(
    {
        "count",
        "cnt",
        "num",
        "qty",
        "quantity",
        "records",
        "rows",
        "rating",
        "score",
        "rank",
        "index",
        "days",
        "hours",
        "minutes",
        "seconds",
        "months",
        "years",
        "weeks",
    }
)
ENTITY_COUNT_TOKENS: frozenset[str] = frozenset(
    {
        "units",
        "orders",
        "items",
        "transactions",
        "txns",
        "sessions",
        "visits",
        "clicks",
        "users",
        "customers",
    }
)
COUNT_UNIT_TOKENS: frozenset[str] = STRICT_COUNT_TOKENS | ENTITY_COUNT_TOKENS
COUNT_MARKER_TOKENS: frozenset[str] = frozenset({"count", "cnt", "num", "qty", "quantity", "rank", "score", "index"})
CURRENCY_UNIT_TOKENS: frozenset[str] = frozenset(
    {
        "revenue",
        "profit",
        "sales",
        "amount",
        "amt",
        "price",
        "cost",
        "spend",
        "budget",
        "income",
        "earnings",
        "salary",
        "wage",
        "fee",
        "fees",
        "subtotal",
        "usd",
        "dollar",
        "dollars",
        "arr",
        "mrr",
        "gmv",
        "ltv",
        "cac",
        "arpu",
        "aov",
    }
)
ADDITIVE_CURRENCY_HEAD_TOKENS: frozenset[str] = frozenset(
    {
        "revenue",
        "profit",
        "sales",
        "amount",
        "amt",
        "spend",
        "budget",
        "income",
        "earnings",
        "salary",
        "wage",
        "fee",
        "fees",
        "subtotal",
        "usd",
        "dollar",
        "dollars",
        "gmv",
    }
)
NON_ADDITIVE_TOKENS: frozenset[str] = PERCENT_UNIT_TOKENS | frozenset(
    {
        "avg",
        "average",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "stddev",
        "variance",
        "aov",
        "arpu",
        "cac",
        "delay",
        "delays",
        "duration",
        "latency",
        "tenure",
        "age",
        "warranty",
        "rating",
        "score",
        "index",
        "days",
        "per",
    }
)
AGGREGATE_STAT_TOKENS: frozenset[str] = frozenset(
    {"avg", "average", "mean", "median", "min", "max", "std", "stddev", "variance", "per"}
)
PREFERRED_ADDITIVE_METRIC_TOKENS: tuple[str, ...] = (
    "revenue",
    "sales",
    "profit",
    "amount",
    "units",
    "quantity",
    "orders",
    "total",
    "count",
)
PREFERRED_ANY_METRIC_TOKENS: tuple[str, ...] = (
    "revenue",
    "sales",
    "profit",
    "margin",
    "amount",
    "units",
    "total",
    "count",
)

MAX_NUMERIC_TIME_CARDINALITY = 60
MAX_PIE_SLICE_COUNT = 8
MAX_EXPLICIT_PIE_SLICE_COUNT = 12
MAX_SCATTER_COLOR_CARDINALITY = 12
DEFAULT_COMPACT_K_THRESHOLD = 1_000.0
KPI_COMPACT_K_THRESHOLD = 10_000.0
HEATMAP_COLOR_SCALE = "YlOrRd"
UNSUPPORTED_TRENDLINES: frozenset[str] = frozenset({"ols", "lowess"})

_CHART_VIOLIN_RE = re.compile(r"\bviolins?\b", re.IGNORECASE)
_CHART_BOX_RE = re.compile(
    r"\b(?:box\s*[-\s]?\s*(?:plots?|charts?)|boxplots?|box(?:\s*[-\s]?\s*(?:&|and)\s*[-\s]?\s*|\s*-\s*)whiskers?|whiskers?\s*plots?)\b",
    re.IGNORECASE,
)
_CHART_HEATMAP_RE = re.compile(r"\b(?:heatmaps?|heat\s*[-\s]?\s*maps?)\b", re.IGNORECASE)
_CHART_HISTOGRAM_RE = re.compile(r"\bhistograms?\b", re.IGNORECASE)
_CHART_TREEMAP_RE = re.compile(r"\b(?:treemaps?|tree\s*[-\s]?\s*maps?)\b", re.IGNORECASE)
_CHART_FUNNEL_RE = re.compile(r"\bfunnels?\b", re.IGNORECASE)
_CHART_SCATTER_RE = re.compile(
    r"\b(?:scatter|scatterplots?|scatter\s*[-\s]?\s*(?:plots?|charts?|graphs?))\b",
    re.IGNORECASE,
)
_CHART_DONUT_RE = re.compile(r"\b(?:donuts?|doughnuts?)\b", re.IGNORECASE)
_CHART_PIE_RE = re.compile(r"\bpie\b", re.IGNORECASE)
_CHART_SHARE_RE = re.compile(r"\bshare\b", re.IGNORECASE)
_CHART_AREA_RE = re.compile(
    r"\b(?:stacked\s+area|area\s*[-\s]?\s*(?:charts?|plots?|graphs?)|as\s+an?\s+area)\b",
    re.IGNORECASE,
)
_CHART_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:"
    r"do(?:es)?\s+not|don'?t|doesn'?t|"
    r"should\s+not|shouldn'?t|"
    r"would\s+not|wouldn'?t|"
    r"could\s+not|couldn'?t|"
    r"can(?:\s+not|'?t)|cannot|"
    r"will\s+not|won'?t|"
    r"instead\s+of|rather\s+than|other\s+than|apart\s+from|"
    r"without|avoid(?:ing|ed|s)?|exclud(?:e|ing|ed|es)|"
    r"skip(?:ping|ped|s)?|omit(?:ting|ted|s)?|"
    r"except|besides|neither|nor|never|not|no|non"
    r")"
    r"(?:\s+(?:us(?:e|ing)|show(?:ing)?|plot(?:ting)?|draw(?:ing)?|mak(?:e|ing)|"
    r"build(?:ing)?|render(?:ing)?|generat(?:e|ing)|creat(?:e|ing)|"
    r"display(?:ing)?|add(?:ing)?|includ(?:e|ing)|embed(?:ding)?))?"
    r"(?:\s+(?:a|an|the|any))?"
    r"(?:\s+(?:embedded|inline|stacked|grouped|simple|basic|standard|extra|another|style|styled))*"
    r"(?:\s+[a-z]+(?:[-\s]+[a-z]+){0,2}\s+(?:or|nor)(?:\s+(?:a|an|the|any))?"
    r"(?:\s+(?:embedded|inline|stacked|grouped|simple|basic|standard|extra|another|style|styled))*)?"
    r"(?:\s+|-)$",
    re.IGNORECASE,
)


def _has_chart_intent(question: str, pattern: re.Pattern[str]) -> bool:
    """Return True if `pattern` matches `question` outside a negated phrase."""
    if not question:
        return False
    for match in pattern.finditer(question):
        if not _CHART_NEGATION_PREFIX_RE.search(question[: match.start()]):
            return True
    return False


def tokenize_col_name(col_name: str) -> list[str]:
    """Split snake_case, kebab-case, and camelCase column names into lowercase word tokens."""
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(col_name))
    return [tok for tok in re.split(r"[^a-z0-9]+", snake.lower()) if tok]


def is_id_or_code_column(col_name: str) -> bool:
    """Return True if `col_name` denotes an identifier, key, code, or postal/zip column."""
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(col_name).strip().lower()).strip("_")
    if normalized in ID_CODE_EXACT_NAMES:
        return True
    tokens = tokenize_col_name(col_name)
    if not tokens:
        return False
    token_set = set(tokens)
    if tokens[-1] in ID_CODE_SUFFIX_TOKENS:
        return True
    if tokens[0] == "id" and not (token_set & {"count", "cnt", "num", "total"}):
        return True
    if tokens[0] in ID_CODE_PREFIX_TOKENS and not (
        token_set & (CURRENCY_UNIT_TOKENS | PERCENT_UNIT_TOKENS | {"count", "cnt", "num", "total", "units", "avg"})
    ):
        return True
    return False


def is_binary_bool_indicator(col_name: str, series: pd.Series) -> bool:
    """Return True for numeric `{0, 1}` boolean flag columns (e.g., SQLite-roundtripped `is_*`)."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False
    tokens = tokenize_col_name(col_name)
    if not tokens:
        return False
    is_bool_named = (
        tokens[0] in BOOL_INDICATOR_PREFIX_TOKENS or tokens[-1] in BOOL_INDICATOR_SUFFIX_TOKENS
    )
    if not is_bool_named:
        return False
    token_set = set(tokens)
    if token_set & (
        PERCENT_UNIT_TOKENS
        | CURRENCY_UNIT_TOKENS
        | {"count", "cnt", "num", "sum", "total", "avg", "mean", "median", "min", "max", "score", "days", "hours"}
    ):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    try:
        return set(non_null.unique()).issubset({0, 1, 0.0, 1.0})
    except TypeError:
        return False


def is_quantitative_numeric_series(col_name: str, series: pd.Series) -> bool:
    """Return True if `series` is a quantitative numeric measure (excluding bools and IDs/codes)."""
    if pd.api.types.is_bool_dtype(series):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if is_id_or_code_column(col_name):
        return False
    if is_binary_bool_indicator(col_name, series):
        return False
    return True


def get_quantitative_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return DataFrame column names that represent quantitative numeric measures."""
    return [str(col) for col in df.columns if is_quantitative_numeric_series(str(col), df[col])]


def is_non_additive_metric(col_name: str) -> bool:
    """Return True for rates, percentages, margins, averages, ratios, and delays that should not be summed."""
    tokens = tokenize_col_name(col_name)
    if not tokens:
        return False
    token_set = set(tokens)
    if token_set & AGGREGATE_STAT_TOKENS:
        return True
    if tokens[-1] in {"count", "cnt", "num"} or tokens[0] in {"count", "cnt", "num"}:
        return False
    if "unit" in token_set and bool(token_set & {"price", "cost"}):
        return True
    if tokens[-1] in ENTITY_COUNT_TOKENS and tokens[0] not in EXPLICIT_PERCENT_HEAD_TOKENS:
        return False
    if tokens[-1] in ADDITIVE_CURRENCY_HEAD_TOKENS and not (
        token_set & (EXPLICIT_PERCENT_HEAD_TOKENS | {"unit"})
    ):
        return False
    if token_set & NON_ADDITIVE_TOKENS:
        if tokens[0] in {"total", "sum"} and not (
            token_set
            & (PERCENT_UNIT_TOKENS | {"delay", "delays", "duration", "latency"})
        ):
            return False
        return True
    return False


def classify_metric_unit(col_name: str) -> Literal["percent", "currency", "none"]:
    """Classify a metric column into mutually exclusive units (`percent`, `currency`, or `none`)."""
    tokens = tokenize_col_name(col_name)
    if not tokens:
        return "none"
    if "per" in tokens:
        prefix_tokens = tokens[: tokens.index("per")]
        unit_tokens = prefix_tokens or tokens
    else:
        unit_tokens = tokens
    token_set = set(unit_tokens)
    last_tok = unit_tokens[-1]

    if last_tok in COUNT_MARKER_TOKENS or unit_tokens[0] in {"count", "cnt", "num"}:
        return "none"
    if last_tok in CURRENCY_UNIT_TOKENS and not (token_set & EXPLICIT_PERCENT_HEAD_TOKENS):
        return "currency"
    if last_tok in PERCENT_UNIT_TOKENS:
        return "percent"
    if last_tok in ENTITY_COUNT_TOKENS:
        if token_set & EXPLICIT_PERCENT_HEAD_TOKENS:
            return "percent"
        return "none"
    if token_set & PERCENT_UNIT_TOKENS:
        return "percent"
    if token_set & COUNT_UNIT_TOKENS:
        return "none"
    if token_set & CURRENCY_UNIT_TOKENS:
        return "currency"
    return "none"


def get_metric_unit_affixes(col_name: str) -> tuple[str, str]:
    """Return mutually exclusive `(prefix, suffix)` strings for chart labels and tooltips."""
    unit = classify_metric_unit(col_name)
    if unit == "percent":
        return "", "%"
    if unit == "currency":
        return "$", ""
    return "", ""


def format_metric_value(
    val: float,
    col_name: str,
    *,
    compact_k_threshold: float = DEFAULT_COMPACT_K_THRESHOLD,
) -> str:
    """Format numerical values cleanly for direct chart labels and KPI tiles (e.g. $145.0K, -$12.5K, 38.5%)."""
    if pd.isna(val):
        return ""
    unit = classify_metric_unit(col_name)
    if unit == "percent":
        return f"{float(val):.1f}%"

    prefix = "$" if unit == "currency" else ""
    sign = "-" if val < 0 else ""
    abs_v = abs(float(val))

    if abs_v >= 1_000_000:
        return f"{sign}{prefix}{abs_v / 1_000_000:.2f}M"
    if abs_v >= compact_k_threshold:
        return f"{sign}{prefix}{abs_v / 1_000:.1f}K"
    if abs_v.is_integer():
        return f"{sign}{prefix}{int(abs_v):,}"
    return f"{sign}{prefix}{abs_v:,.1f}"


def _format_data_value(val: float, col_name: str) -> str:
    """Format numerical values cleanly for direct chart labels (e.g. $145K, 38.5%)."""
    return format_metric_value(val, col_name, compact_k_threshold=DEFAULT_COMPACT_K_THRESHOLD)


class ChartSpec(BaseModel):
    """Structured chart specification produced by the Stats & Chart node."""

    chart_type: Literal[
        "bar",
        "line",
        "area",
        "pie",
        "scatter",
        "box",
        "violin",
        "histogram",
        "treemap",
        "heatmap",
        "funnel",
        "none",
    ] = Field(
        default="bar",
        description="Best chart type for the SQL result table ('none' if single scalar value).",
    )
    x_col: str = Field(
        default="",
        description="Exact column name from the SQL result DataFrame for the X-axis or category.",
    )
    y_col: str = Field(
        default="",
        description="Exact numeric column name from the SQL result DataFrame for the Y-axis or value.",
    )
    color_col: str | None = Field(
        default=None,
        description="Optional secondary grouping/series column name from the SQL result DataFrame.",
    )
    title: str = Field(
        default="Analysis Breakdown",
        description="Concise, business-friendly chart title.",
    )
    sort_descending: bool = Field(
        default=False,
        description="Set True for ranked categorical bar charts; False for chronological time series.",
    )


def _human_label(col_name: str) -> str:
    """Turn `snake_case` SQL column names into clean axis labels."""
    return str(col_name).replace("_", " ").strip().title()


def _is_calendar_time_name(col_name: str) -> bool:
    """Return True if a column name refers to a calendar bucket rather than a duration metric."""
    words = set(tokenize_col_name(col_name))
    if not words or (words & DURATION_METRIC_WORDS):
        return False
    return bool(words & TIME_DIMENSION_WORDS)


def _select_primary_metric(numeric_cols: list[str]) -> str:
    """Select the primary numeric measure, preferring additive business metrics over ratios/delays."""
    additive_cols = [c for c in numeric_cols if not is_non_additive_metric(c)]

    for pref in PREFERRED_ADDITIVE_METRIC_TOKENS:
        matched = next((c for c in additive_cols if pref in tokenize_col_name(c)), None)
        if matched is not None:
            return matched

    if additive_cols:
        return additive_cols[0]

    for pref in PREFERRED_ANY_METRIC_TOKENS:
        matched = next((c for c in numeric_cols if pref in tokenize_col_name(c)), None)
        if matched is not None:
            return matched

    return numeric_cols[0]


def _identify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], str | None]:
    """Classify DataFrame columns into `(numeric_cols, dimension_cols, time_col)`."""
    numeric_cols: list[str] = []
    dimension_cols: list[str] = []
    time_col: str | None = None

    for col in df.columns:
        s = df[col]
        col_str = str(col)
        is_time_named = _is_calendar_time_name(col_str)

        if pd.api.types.is_datetime64_any_dtype(s):
            dimension_cols.append(col)
            if time_col is None:
                time_col = col
        elif pd.api.types.is_bool_dtype(s):
            dimension_cols.append(col)
        elif is_id_or_code_column(col_str):
            dimension_cols.append(col)
        elif is_binary_bool_indicator(col_str, s):
            dimension_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            # Only treat integer calendar columns (e.g. year=2026, month=1..12) as time dimensions;
            # float columns and duration/delay/count metrics must remain numeric measures.
            is_integer_calendar = (
                pd.api.types.is_integer_dtype(s)
                and is_time_named
                and s.nunique(dropna=True) <= MAX_NUMERIC_TIME_CARDINALITY
                and len(df) > 1
            )
            if is_integer_calendar:
                dimension_cols.append(col)
                if time_col is None:
                    time_col = col
            else:
                numeric_cols.append(col)
        else:
            dimension_cols.append(col)
            if time_col is None and is_time_named:
                time_col = col

    return numeric_cols, dimension_cols, time_col


def run_statistical_checks(df: pd.DataFrame) -> dict[str, Any]:
    """Compute deterministic Python statistics over a SQL result DataFrame."""
    if df is None or df.empty:
        return {
            "row_count": 0,
            "primary_metric": None,
            "summary_lines": ["Query returned 0 rows."],
            "summary_text": "Query returned 0 rows.",
            "metrics": {},
            "outliers": [],
        }

    numeric_cols, dimension_cols, time_col = _identify_columns(df)
    row_count = int(len(df))
    summary_lines: list[str] = [f"Rows analyzed: {row_count}"]

    if not numeric_cols:
        text = "\n".join(f"- {line}" for line in summary_lines)
        return {
            "row_count": row_count,
            "primary_metric": None,
            "summary_lines": summary_lines,
            "summary_text": text,
            "metrics": {},
            "outliers": [],
        }

    primary_metric = _select_primary_metric(numeric_cols)
    primary_is_non_additive = is_non_additive_metric(primary_metric)

    s_num = pd.to_numeric(df[primary_metric], errors="coerce").dropna()
    if s_num.empty:
        return {
            "row_count": row_count,
            "primary_metric": primary_metric,
            "summary_lines": summary_lines,
            "summary_text": f"- Rows analyzed: {row_count}",
            "metrics": {},
            "outliers": [],
        }

    total_val = float(s_num.sum())
    mean_val = float(s_num.mean())
    median_val = float(s_num.median())
    min_val = float(s_num.min())
    max_val = float(s_num.max())
    std_val = float(s_num.std(ddof=0)) if len(s_num) > 1 else 0.0

    metrics_info: dict[str, Any] = {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "std": round(std_val, 2),
        "is_non_additive": primary_is_non_additive,
    }
    if not primary_is_non_additive:
        metrics_info["total"] = round(total_val, 2)

    metric_label = _human_label(primary_metric)
    if row_count == 1:
        single_val = mean_val if primary_is_non_additive else total_val
        summary_lines.append(f"{metric_label}: {single_val:,.2f}")
    elif primary_is_non_additive:
        summary_lines.append(
            f"{metric_label} — Mean: {mean_val:,.2f} | Median: {median_val:,.2f} "
            f"(Min: {min_val:,.2f}, Max: {max_val:,.2f})"
        )
    else:
        summary_lines.append(
            f"{metric_label} — Total: {total_val:,.2f} | Mean: {mean_val:,.2f} | Median: {median_val:,.2f}"
        )

    label_col = dimension_cols[0] if dimension_cols else None

    # Top & Bottom contributors when multiple rows exist
    if row_count >= 2:
        valid_df = df.dropna(subset=[primary_metric]).copy()
        if not valid_df.empty:
            top_row = valid_df.loc[valid_df[primary_metric].idxmax()]
            bot_row = valid_df.loc[valid_df[primary_metric].idxmin()]
            top_val = float(top_row[primary_metric])
            bot_val = float(bot_row[primary_metric])

            if label_col is not None:
                dim_label_cols = dimension_cols[:2]
                top_name = " · ".join(str(top_row[c]) for c in dim_label_cols)
                bot_name = " · ".join(str(bot_row[c]) for c in dim_label_cols)
            else:
                top_name = f"Row {top_row.name}"
                bot_name = f"Row {bot_row.name}"

            if not primary_is_non_additive and total_val > 0 and float(valid_df[primary_metric].min()) >= 0:
                top_share = (top_val / total_val) * 100.0
                bot_share = (bot_val / total_val) * 100.0
                summary_lines.append(
                    f"Top contributor: {top_name} ({top_val:,.2f}, {top_share:.1f}% of total)"
                )
                summary_lines.append(
                    f"Lowest contributor: {bot_name} ({bot_val:,.2f}, {bot_share:.1f}% of total)"
                )
                metrics_info["top_contributor"] = {
                    "label": top_name,
                    "value": round(top_val, 2),
                    "share_pct": round(top_share, 1),
                }
                metrics_info["bottom_contributor"] = {
                    "label": bot_name,
                    "value": round(bot_val, 2),
                    "share_pct": round(bot_share, 1),
                }
            else:
                summary_lines.append(f"Highest {metric_label}: {top_name} ({top_val:,.2f})")
                summary_lines.append(f"Lowest {metric_label}: {bot_name} ({bot_val:,.2f})")

    # Period-over-Period delta when a time column is present
    if time_col is not None and row_count >= 2:
        period_agg = "mean" if primary_is_non_additive else "sum"
        time_grouped = (
            df.dropna(subset=[time_col, primary_metric])
            .groupby(time_col, as_index=False)[primary_metric]
            .agg(period_agg)
            .sort_values(time_col)
        )
        if len(time_grouped) >= 2:
            prev_period = str(time_grouped.iloc[-2][time_col])
            curr_period = str(time_grouped.iloc[-1][time_col])
            prev_val = float(time_grouped.iloc[-2][primary_metric])
            curr_val = float(time_grouped.iloc[-1][primary_metric])
            if prev_val != 0:
                pct_delta = ((curr_val - prev_val) / abs(prev_val)) * 100.0
                metrics_info["latest_period_delta_pct"] = round(pct_delta, 2)
                summary_lines.append(
                    f"Latest period change ({prev_period} → {curr_period}): "
                    f"{prev_val:,.2f} → {curr_val:,.2f} ({pct_delta:+.1f}%)"
                )

    # Secondary numeric columns quick averages/sums (helpful on multi-metric root-cause queries)
    secondary_cols = [c for c in numeric_cols if c != primary_metric][:3]
    for sec_col in secondary_cols:
        sec_s = pd.to_numeric(df[sec_col], errors="coerce").dropna()
        if sec_s.empty:
            continue
        sec_label = _human_label(sec_col)
        if is_non_additive_metric(sec_col):
            summary_lines.append(
                f"{sec_label} — Mean: {float(sec_s.mean()):,.2f} (Min: {float(sec_s.min()):,.2f}, Max: {float(sec_s.max()):,.2f})"
            )
        else:
            summary_lines.append(
                f"{sec_label} — Total: {float(sec_s.sum()):,.2f} | Mean: {float(sec_s.mean()):,.2f}"
            )

    # Z-score outlier check (|z| > 2.0) when >= 4 rows
    outliers: list[dict[str, Any]] = []
    if len(s_num) >= 4 and std_val > 0:
        z_scores = (df[primary_metric].astype(float) - mean_val) / std_val
        outlier_mask = z_scores.abs() > 2.0
        for idx in df.index[outlier_mask.fillna(False)]:
            row_label = (
                str(df.loc[idx, label_col]) if label_col is not None else f"Row {idx}"
            )
            val = float(df.loc[idx, primary_metric])
            z_val = float(z_scores.loc[idx])
            outliers.append(
                {"label": row_label, "value": round(val, 2), "z_score": round(z_val, 2)}
            )
        if outliers:
            formatted = ", ".join(
                f"{o['label']} ({o['value']:,.2f}, z={o['z_score']:+.2f})" for o in outliers[:3]
            )
            summary_lines.append(f"Statistical outliers (|z| > 2.0): {formatted}")

    return {
        "row_count": row_count,
        "primary_metric": primary_metric,
        "summary_lines": summary_lines,
        "summary_text": "\n".join(f"- {line}" for line in summary_lines),
        "metrics": metrics_info,
        "outliers": outliers,
    }


ALLOWED_SANDBOX_MODULES = {
    "plotly",
    "plotly.express",
    "px",
    "plotly.graph_objects",
    "go",
    "pandas",
    "pd",
    "numpy",
    "np",
    "scipy",
}

FORBIDDEN_SANDBOX_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "input",
    "exit",
    "quit",
    "breakpoint",
    "help",
    "memoryview",
}

def _safe_sandbox_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted import hook allowing only approved data visualization modules."""
    root_pkg = name.split(".")[0]
    if root_pkg not in ALLOWED_SANDBOX_MODULES:
        raise ImportError(f"Import of module '{name}' is forbidden in sandbox.")
    return __import__(name, *args, **kwargs)


SAFE_SANDBOX_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "range": range,
    "isinstance": isinstance,
    "print": lambda *args, **kwargs: None,
    "__import__": _safe_sandbox_import,
    "True": True,
    "False": False,
    "None": None,
}


def _extract_raw_python_code(code_str: str) -> str:
    """Strip markdown code fences and extraneous text from LLM response."""
    cleaned = code_str.strip()
    if "```python" in cleaned:
        cleaned = cleaned.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
    return cleaned.strip()


def is_plotly_code_safe(code_str: str) -> tuple[bool, str]:
    """Inspect AST statically to ensure LLM-generated code cannot escape sandbox or access OS."""
    raw_code = _extract_raw_python_code(code_str)
    if not raw_code:
        return False, "Empty code snippet."
    try:
        tree = ast.parse(raw_code)
    except SyntaxError as exc:
        return False, f"SyntaxError in generated code: {exc}"

    for node in ast.walk(tree):
        # 1. Imports check: only approved data/chart packages allowed
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in ALLOWED_SANDBOX_MODULES:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in ALLOWED_SANDBOX_MODULES:
                    return False, f"Forbidden import from: {node.module}"

        # 2. Block system / meta / eval calls
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_SANDBOX_NAMES:
                return False, f"Forbidden identifier: {node.id}"

        # 3. Block private / dunder attribute access to prevent sandbox escapes
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"Forbidden attribute access: {node.attr}"

        # 4. Block Plotly trendlines that require uninstalled statsmodels
        elif isinstance(node, ast.keyword) and node.arg == "trendline":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if node.value.value.strip().lower() in UNSUPPORTED_TRENDLINES:
                    return (
                        False,
                        f"Unsupported Plotly trendline '{node.value.value}' requires uninstalled 'statsmodels'.",
                    )

    return True, "Safe"


def execute_sandboxed_plotly_code(
    code_str: str,
    df: pd.DataFrame,
    default_title: str = "",
) -> go.Figure | None:
    """Safely executes validated AI code in a sandboxed namespace and returns a Plotly Figure."""
    if df is None or df.empty or not code_str:
        return None

    raw_code = _extract_raw_python_code(code_str)
    is_safe, _ = is_plotly_code_safe(raw_code)
    if not is_safe:
        return None

    safe_globals = {
        "__builtins__": SAFE_SANDBOX_BUILTINS,
        "df": df.copy(),
        "px": px,
        "go": go,
        "pd": pd,
        "np": np,
        "CHART_PALETTE": CHART_PALETTE,
        "COLOR_PRIMARY": COLOR_PRIMARY,
        "COLOR_SECONDARY": COLOR_SECONDARY,
        "apply_neutral_layout": apply_neutral_layout,
        "_human_label": _human_label,
        "_format_data_value": _format_data_value,
    }
    local_vars: dict[str, Any] = {}

    try:
        exec(raw_code, safe_globals, local_vars)
    except Exception:
        return None

    candidate_fig = local_vars.get("fig") or safe_globals.get("fig")
    if not isinstance(candidate_fig, go.Figure):
        for v in list(local_vars.values()) + list(safe_globals.values()):
            if isinstance(v, go.Figure):
                candidate_fig = v
                break

    if not isinstance(candidate_fig, go.Figure):
        return None

    # For box plots: auto-correct scale compression if heavily skewed
    try:
        is_box = any(getattr(trace, "type", "") == "box" for trace in getattr(candidate_fig, "data", []))
        if is_box:
            all_y: list[float] = []
            for tr in candidate_fig.data:
                y_arr = getattr(tr, "y", None)
                if y_arr is not None:
                    all_y.extend([float(v) for v in y_arr if pd.notna(v)])
            if len(all_y) >= 4:
                s_y = pd.Series(all_y)
                q75 = float(s_y.quantile(0.75))
                max_v = float(s_y.max())
                min_v = float(s_y.min())
                if q75 > 0 and max_v >= 5 * q75 and min_v > 0:
                    current_y_type = getattr(getattr(candidate_fig.layout, "yaxis", None), "type", "linear")
                    if current_y_type != "log":
                        candidate_fig.update_yaxes(type="log")
                    if not candidate_fig.layout.updatemenus:
                        candidate_fig.update_layout(
                            updatemenus=[
                                dict(
                                    type="buttons",
                                    direction="right",
                                    x=1.0,
                                    y=1.14,
                                    xanchor="right",
                                    yanchor="top",
                                    bgcolor="rgba(255, 255, 255, 0.08)",
                                    bordercolor="rgba(148, 163, 184, 0.25)",
                                    font=dict(size=11),
                                    buttons=[
                                        dict(label="Log Scale (Un-squished)", method="relayout", args=[{"yaxis.type": "log"}]),
                                        dict(label="Linear Scale", method="relayout", args=[{"yaxis.type": "linear"}]),
                                    ],
                                )
                            ]
                        )
    except Exception:
        pass

    try:
        paper_bg = getattr(candidate_fig.layout, "paper_bgcolor", None)
        if not paper_bg or paper_bg != "rgba(0,0,0,0)":
            title_text = ""
            if candidate_fig.layout.title and candidate_fig.layout.title.text:
                title_text = candidate_fig.layout.title.text
            apply_neutral_layout(candidate_fig, title=title_text or default_title)
    except Exception:
        pass

    return candidate_fig


def _is_numeric_non_bool(series: pd.Series) -> bool:
    """Return True if `series` has a numeric dtype and is not boolean."""
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def infer_default_chart_spec(df: pd.DataFrame, question: str = "") -> ChartSpec:
    """Deterministic fallback ChartSpec builder when LLM output is omitted or mismatched."""
    if df is None or len(df) < 2:
        return ChartSpec(chart_type="none", title="Summary Result")

    numeric_cols, dimension_cols, time_col = _identify_columns(df)
    if not numeric_cols:
        return ChartSpec(chart_type="none", title="Query Result")

    y_col = _select_primary_metric(numeric_cols)

    if _has_chart_intent(question, _CHART_VIOLIN_RE):
        x_col = dimension_cols[0] if dimension_cols else ""
        return ChartSpec(
            chart_type="violin",
            x_col=x_col,
            y_col=y_col,
            color_col=dimension_cols[1] if len(dimension_cols) >= 2 else None,
            title=f"{_human_label(y_col)} Violin Distribution" + (f" by {_human_label(x_col)}" if x_col else ""),
        )

    if _has_chart_intent(question, _CHART_BOX_RE):
        x_col = dimension_cols[0] if dimension_cols else ""
        return ChartSpec(
            chart_type="box",
            x_col=x_col,
            y_col=y_col,
            color_col=dimension_cols[1] if len(dimension_cols) >= 2 else None,
            title=f"{_human_label(y_col)} Distribution" + (f" by {_human_label(x_col)}" if x_col else ""),
        )

    if _has_chart_intent(question, _CHART_HEATMAP_RE):
        if len(dimension_cols) >= 2:
            return ChartSpec(
                chart_type="heatmap",
                x_col=dimension_cols[0],
                y_col=y_col,
                color_col=dimension_cols[1],
                title=f"{_human_label(y_col)} Heatmap ({_human_label(dimension_cols[0])} × {_human_label(dimension_cols[1])})",
            )

    if _has_chart_intent(question, _CHART_HISTOGRAM_RE):
        return ChartSpec(
            chart_type="histogram",
            x_col=y_col,
            y_col=y_col,
            title=f"{_human_label(y_col)} Frequency Distribution",
        )

    if _has_chart_intent(question, _CHART_TREEMAP_RE) and dimension_cols:
        x_col = dimension_cols[0]
        return ChartSpec(
            chart_type="treemap",
            x_col=x_col,
            y_col=y_col,
            color_col=dimension_cols[1] if len(dimension_cols) >= 2 else None,
            title=f"{_human_label(y_col)} Treemap by {_human_label(x_col)}",
        )

    if _has_chart_intent(question, _CHART_FUNNEL_RE) and dimension_cols:
        x_col = dimension_cols[0]
        return ChartSpec(
            chart_type="funnel",
            x_col=x_col,
            y_col=y_col,
            color_col=dimension_cols[1] if len(dimension_cols) >= 2 else None,
            title=f"{_human_label(y_col)} Funnel by {_human_label(x_col)}",
        )

    if _has_chart_intent(question, _CHART_SCATTER_RE) and len(numeric_cols) >= 2:
        scatter_y = y_col
        other_nums = [c for c in numeric_cols if c != scatter_y]
        scatter_x = other_nums[0]
        non_id_dims = [c for c in dimension_cols if not is_id_or_code_column(c)]
        non_time_dims = [c for c in non_id_dims if c != time_col]
        color_candidates = non_time_dims or non_id_dims
        scatter_color = next(
            (
                c
                for c in color_candidates
                if 2 <= int(df[c].nunique(dropna=True)) <= MAX_SCATTER_COLOR_CARDINALITY
            ),
            None,
        )
        return ChartSpec(
            chart_type="scatter",
            x_col=scatter_x,
            y_col=scatter_y,
            color_col=scatter_color,
            title=f"{_human_label(scatter_y)} vs {_human_label(scatter_x)}",
        )

    if (_has_chart_intent(question, _CHART_DONUT_RE) or _has_chart_intent(question, _CHART_PIE_RE)) and dimension_cols:
        pie_dims = [c for c in dimension_cols if c != time_col] or dimension_cols
        x_col = pie_dims[0]
        color_col = pie_dims[1] if len(pie_dims) >= 2 else None
        y_min = pd.to_numeric(df[y_col], errors="coerce").min()
        if (
            color_col is None
            and int(df[x_col].nunique(dropna=True)) <= MAX_EXPLICIT_PIE_SLICE_COUNT
            and pd.notna(y_min)
            and float(y_min) >= 0
        ):
            return ChartSpec(
                chart_type="pie",
                x_col=x_col,
                y_col=y_col,
                title=f"{_human_label(y_col)} Share by {_human_label(x_col)}",
            )

    if time_col is not None:
        other_dims = [c for c in dimension_cols if c != time_col]
        color_col = other_dims[0] if other_dims else None
        # Use grouped bar if only 2 time periods (e.g. Sep vs Oct comparison), line if >=3 periods
        unique_periods = int(df[time_col].nunique(dropna=True))
        if unique_periods <= 2 and color_col is not None:
            return ChartSpec(
                chart_type="bar",
                x_col=color_col,
                y_col=y_col,
                color_col=time_col,
                title=f"{_human_label(y_col)} by {_human_label(color_col)} ({_human_label(time_col)})",
                sort_descending=False,
            )
        chart_kind: Literal["line", "area", "bar"] = (
            "area" if _has_chart_intent(question, _CHART_AREA_RE) else "line"
        )
        return ChartSpec(
            chart_type=chart_kind,
            x_col=time_col,
            y_col=y_col,
            color_col=color_col,
            title=f"{_human_label(y_col)} Over {_human_label(time_col)}",
            sort_descending=False,
        )

    if dimension_cols:
        x_col = dimension_cols[0]
        color_col = dimension_cols[1] if len(dimension_cols) >= 2 else None
        if _has_chart_intent(question, _CHART_SHARE_RE):
            y_min = pd.to_numeric(df[y_col], errors="coerce").min()
            if (
                color_col is None
                and len(df) <= MAX_PIE_SLICE_COUNT
                and pd.notna(y_min)
                and float(y_min) >= 0
            ):
                return ChartSpec(
                    chart_type="pie",
                    x_col=x_col,
                    y_col=y_col,
                    title=f"{_human_label(y_col)} Share by {_human_label(x_col)}",
                )
        return ChartSpec(
            chart_type="bar",
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            title=f"{_human_label(y_col)} by {_human_label(x_col)}",
            sort_descending=color_col is None,
        )

    if len(numeric_cols) >= 2:
        scatter_y = y_col
        other_nums = [c for c in numeric_cols if c != scatter_y]
        scatter_x = other_nums[0]
        return ChartSpec(
            chart_type="scatter",
            x_col=scatter_x,
            y_col=scatter_y,
            title=f"{_human_label(scatter_y)} vs {_human_label(scatter_x)}",
        )

    return ChartSpec(
        chart_type="histogram",
        x_col=y_col,
        y_col=y_col,
        title=f"{_human_label(y_col)} Frequency Distribution",
    )


def _resolve_heatmap_columns(df: pd.DataFrame, spec: ChartSpec) -> tuple[str, str, str] | None:
    """Resolve `(x_col, y_col, value_col)` for a 2D heatmap from a ChartSpec or DataFrame."""
    cols = set(df.columns)
    numeric_cols, dimension_cols, _ = _identify_columns(df)
    if not numeric_cols:
        return None
    numeric_set = set(numeric_cols)

    # Honor explicit spec columns if they supply two dimensions and one numeric value column
    if (
        spec.x_col in cols
        and spec.y_col in cols
        and spec.color_col in cols
        and spec.x_col != spec.y_col
        and spec.color_col not in (spec.x_col, spec.y_col)
    ):
        if spec.y_col in numeric_set and spec.x_col not in numeric_set:
            return spec.x_col, spec.color_col, spec.y_col
        if spec.color_col in numeric_set and spec.y_col not in numeric_set:
            return spec.x_col, spec.y_col, spec.color_col

    if len(dimension_cols) >= 2:
        val_col = spec.y_col if spec.y_col in numeric_set else numeric_cols[0]
        return dimension_cols[0], dimension_cols[1], val_col

    return None


def _is_chart_spec_valid(df: pd.DataFrame, spec: ChartSpec) -> tuple[bool, tuple[str, str, str] | None]:
    """Validate `spec` against `df` columns and resolve heatmap axes when applicable."""
    if spec.chart_type == "none":
        return False, None
    cols = set(df.columns)
    if spec.chart_type == "heatmap":
        heat_cols = _resolve_heatmap_columns(df, spec)
        return (heat_cols is not None), heat_cols
    if spec.chart_type in ("box", "violin"):
        ok = bool(spec.y_col and spec.y_col in cols and _is_numeric_non_bool(df[spec.y_col]))
        return ok, None
    if spec.chart_type == "histogram":
        ok = bool(spec.x_col and spec.x_col in cols and _is_numeric_non_bool(df[spec.x_col]))
        return ok, None
    ok = bool(
        spec.x_col
        and spec.x_col in cols
        and spec.y_col
        and spec.y_col in cols
        and _is_numeric_non_bool(df[spec.y_col])
    )
    return ok, None


def build_plotly_chart(
    df: pd.DataFrame,
    chart_spec: ChartSpec | dict[str, Any] | None = None,
    question: str = "",
) -> go.Figure | None:
    """Render a clean, theme-adaptive Plotly figure from `df` and `chart_spec` with safe fallbacks."""
    if df is None or len(df) < 2:
        return None

    spec: ChartSpec
    if isinstance(chart_spec, ChartSpec):
        spec = chart_spec
    elif isinstance(chart_spec, dict):
        try:
            spec = ChartSpec(**chart_spec)
        except Exception:
            spec = infer_default_chart_spec(df, question=question)
    else:
        spec = infer_default_chart_spec(df, question=question)

    if spec.chart_type == "none":
        return None

    # Validate columns against actual DataFrame; fallback seamlessly if mismatched
    cols = set(df.columns)
    is_valid, heat_cols = _is_chart_spec_valid(df, spec)
    if not is_valid:
        spec = infer_default_chart_spec(df, question=question)
        is_valid, heat_cols = _is_chart_spec_valid(df, spec)
        if not is_valid:
            return None

    color_col = spec.color_col if (spec.color_col and spec.color_col in cols and spec.color_col != spec.x_col) else None

    plot_df = df.copy()
    if spec.sort_descending and spec.chart_type == "bar" and color_col is None:
        plot_df = plot_df.sort_values(by=spec.y_col, ascending=False)
    elif spec.chart_type in ("line", "area"):
        plot_df = plot_df.sort_values(by=spec.x_col, ascending=True)

    labels = {
        spec.x_col: _human_label(spec.x_col),
        spec.y_col: _human_label(spec.y_col),
    }
    if color_col:
        labels[color_col] = _human_label(color_col)

    try:
        y_prefix, y_suffix = get_metric_unit_affixes(spec.y_col)

        if spec.chart_type == "bar":
            show_data_labels = len(plot_df) <= 16 and color_col is None
            direct_labels = (
                [_format_data_value(v, spec.y_col) for v in plot_df[spec.y_col]]
                if show_data_labels
                else None
            )

            fig = px.bar(
                plot_df,
                x=spec.x_col,
                y=spec.y_col,
                color=color_col,
                barmode="group" if color_col else "relative",
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
                text=direct_labels,
            )
            # Modern tactile rounded corners and direct labels
            fig.update_traces(
                marker_cornerradius=8,
                marker_line_width=0,
                textposition="outside" if show_data_labels else "none",
                hovertemplate=f"<b>%{{x}}</b><br>{_human_label(spec.y_col)}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
            )

        elif spec.chart_type == "line":
            fig = px.line(
                plot_df,
                x=spec.x_col,
                y=spec.y_col,
                color=color_col,
                markers=True,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig.update_traces(
                line={"width": 3.2, "shape": "spline", "smoothing": 0.85},
                marker={"size": 8, "line": {"width": 2, "color": "#FFFFFF"}},
                hovertemplate=f"<b>%{{x}}</b><br>{_human_label(spec.y_col)}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
            )
            # Add subtle translucent area glow when single series
            if color_col is None:
                fig.update_traces(fill="tozeroy", fillcolor="rgba(255, 90, 95, 0.08)")

                # Automatic peak callout annotation if >= 3 data points
                if len(plot_df) >= 3 and not plot_df[spec.y_col].dropna().empty:
                    try:
                        max_idx = plot_df[spec.y_col].idxmax()
                        peak_x = plot_df.loc[max_idx, spec.x_col]
                        peak_y = float(plot_df.loc[max_idx, spec.y_col])
                        fig.add_annotation(
                            x=peak_x,
                            y=peak_y,
                            text=f"Peak: {_format_data_value(peak_y, spec.y_col)}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor=COLOR_PRIMARY,
                            ax=0,
                            ay=-34,
                            bordercolor=COLOR_PRIMARY,
                            borderwidth=1,
                            borderpad=4,
                            bgcolor="rgba(255, 90, 95, 0.12)",
                            font=dict(size=11, family="Inter, sans-serif"),
                        )
                    except Exception:
                        pass

        elif spec.chart_type == "area":
            fig = px.area(
                plot_df,
                x=spec.x_col,
                y=spec.y_col,
                color=color_col,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig.update_traces(
                line={"width": 2.8, "shape": "spline", "smoothing": 0.85},
                hovertemplate=f"<b>%{{x}}</b><br>{_human_label(spec.y_col)}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
            )

        elif spec.chart_type == "pie":
            fig = px.pie(
                plot_df,
                names=spec.x_col,
                values=spec.y_col,
                hole=0.56,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker=dict(line=dict(color="rgba(255,255,255,0.85)", width=2)),
                domain=dict(y=[0.02, 0.88]),
                hovertemplate=f"<b>%{{label}}</b><br>Share: <b>%{{percent}}</b><br>{_human_label(spec.y_col)}: {y_prefix}%{{value:,.2f}}{y_suffix}<extra></extra>",
            )

        elif spec.chart_type == "scatter":
            fig = px.scatter(
                plot_df,
                x=spec.x_col,
                y=spec.y_col,
                color=color_col,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig.update_traces(
                marker={"size": 10, "opacity": 0.88, "line": {"width": 1.5, "color": "#FFFFFF"}},
                hovertemplate=f"<b>%{{x:,.2f}}</b> × <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
            )

        elif spec.chart_type == "box":
            box_x = spec.x_col if (spec.x_col and spec.x_col in cols) else None
            s_val = pd.to_numeric(plot_df[spec.y_col], errors="coerce").dropna()
            q75 = float(s_val.quantile(0.75)) if len(s_val) >= 4 else 0.0
            max_val = float(s_val.max()) if not s_val.empty else 0.0
            min_val = float(s_val.min()) if not s_val.empty else 0.0
            is_heavily_skewed = (q75 > 0 and max_val >= 5 * q75 and min_val > 0)

            fig = px.box(
                plot_df,
                x=box_x,
                y=spec.y_col,
                color=color_col or box_x,
                points="outliers",
                log_y=is_heavily_skewed,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
            fig.update_traces(
                marker=dict(size=5, opacity=0.8),
                hovertemplate=f"<b>%{{x}}</b><br>{_human_label(spec.y_col)}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
            )
            if is_heavily_skewed:
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=1.0,
                            y=1.14,
                            xanchor="right",
                            yanchor="top",
                            bgcolor="rgba(255, 255, 255, 0.08)",
                            bordercolor="rgba(148, 163, 184, 0.25)",
                            font=dict(size=11),
                            buttons=[
                                dict(label="Log Scale (Un-squished)", method="relayout", args=[{"yaxis.type": "log"}]),
                                dict(label="Linear Scale", method="relayout", args=[{"yaxis.type": "linear"}]),
                            ],
                        )
                    ]
                )

        elif spec.chart_type == "violin":
            violin_x = spec.x_col if (spec.x_col and spec.x_col in cols) else None
            fig = px.violin(
                plot_df,
                x=violin_x,
                y=spec.y_col,
                color=color_col or violin_x,
                box=True,
                points="outliers",
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )

        elif spec.chart_type == "histogram":
            fig = px.histogram(
                plot_df,
                x=spec.x_col,
                color=color_col,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )

        elif spec.chart_type == "treemap":
            path_cols = [c for c in [color_col, spec.x_col] if c and c in plot_df.columns]
            if not path_cols:
                return None
            treemap_df = plot_df.dropna(subset=path_cols)
            treemap_vals = pd.to_numeric(treemap_df[spec.y_col], errors="coerce")
            treemap_df = treemap_df[treemap_vals.fillna(0) > 0]
            if treemap_df.empty:
                return None
            fig = px.treemap(
                treemap_df,
                path=path_cols,
                values=spec.y_col,
                color_discrete_sequence=CHART_PALETTE,
            )

        elif spec.chart_type == "heatmap" and heat_cols is not None:
            heat_x, heat_y, heat_z = heat_cols
            pivot_df = plot_df.pivot_table(
                index=heat_y,
                columns=heat_x,
                values=heat_z,
                aggfunc="mean",
            )
            if pivot_df.empty:
                return None
            fig = px.imshow(
                pivot_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=HEATMAP_COLOR_SCALE,
                labels={
                    "x": _human_label(heat_x),
                    "y": _human_label(heat_y),
                    "color": _human_label(heat_z),
                },
            )

        elif spec.chart_type == "funnel":
            if not spec.x_col or spec.x_col not in plot_df.columns:
                return None
            fig = px.funnel(
                plot_df,
                x=spec.y_col,
                y=spec.x_col,
                color=color_col,
                labels=labels,
                color_discrete_sequence=CHART_PALETTE,
            )
        else:
            return None
    except Exception:
        return None

    has_legend = (color_col is not None) or (spec.chart_type == "pie")
    top_margin = 88 if has_legend else 68

    apply_neutral_layout(
        fig,
        title=spec.title or f"{_human_label(spec.y_col)} by {_human_label(spec.x_col)}",
        height=410,
        top_margin=top_margin,
    )
    if spec.chart_type == "bar":
        fig.update_layout(bargap=0.32, bargroupgap=0.08)
        fig.update_xaxes(showgrid=False)
    return fig


def apply_neutral_layout(fig: go.Figure, title: str = "", height: int = 410, top_margin: int = 85) -> go.Figure:
    """Apply a modern, theme-adaptive layout with clean gridlines and high-contrast typography."""
    fig.update_layout(
        font={"family": "Inter, sans-serif"},
        title={
            "text": title,
            "x": 0.0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"b": 20, "t": 0, "l": 0, "r": 0},
            "font": {"size": 15, "family": "Plus Jakarta Sans, Inter, sans-serif"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_PALETTE,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.05,
            "xanchor": "left",
            "x": 0.0,
            "title_text": "",
            "font": {"size": 12},
        },
        margin={"l": 12, "r": 12, "t": top_margin, "b": 16},
        height=height,
        hovermode="closest",
    )
    fig.update_xaxes(
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont={"size": 11},
    )
    fig.update_yaxes(
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_GRID,
        tickfont={"size": 11},
    )
    return fig
