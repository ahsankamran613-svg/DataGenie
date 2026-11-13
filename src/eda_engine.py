"""Automated Exploratory Data Analysis (EDA) & Time-Series Anomaly Detection Engine.

Provides:
- Dataset overview & column health profiles
- Numeric Pearson correlation matrix & dark-themed Plotly heatmap (`design.md` palette)
- `detect_timeseries_anomalies`: Scikit-learn `IsolationForest` + rolling Z-score
  time-series anomaly detection with red circular markers on a theme-adaptive Plotly chart.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

from src.stats_analyzer import (
    apply_neutral_layout,
    get_metric_unit_affixes,
    get_quantitative_numeric_columns,
    is_binary_bool_indicator,
    is_id_or_code_column,
    is_non_additive_metric,
    is_quantitative_numeric_series,
    tokenize_col_name,
)


# Design tokens for dark executive theme
COLOR_PRIMARY = "#FF6B6B"
COLOR_ACCENT = "#38BDF8"
COLOR_DANGER = "#FF6B6B"
COLOR_NEUTRAL = "rgba(148, 163, 184, 0.15)"
COLOR_MUTED = "#94A3B8"
COLOR_GRID = "rgba(148, 163, 184, 0.18)"


def _is_date_like_series(col_name: str, series: pd.Series) -> bool:
    """Detect whether a column holds ISO date strings or datetime values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype)):
        return False
    sample = series.dropna().head(25).astype(str)
    if sample.empty:
        return False
    iso_ratio = float(sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean())
    return iso_ratio >= 0.80 or ("date" in col_name.lower() and iso_ratio >= 0.50)


def find_default_timeseries_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Automatically choose the best `(date_col, metric_col)` pair for anomaly detection."""
    date_col: str | None = None
    for col in df.columns:
        if _is_date_like_series(col, df[col]):
            date_col = col
            break

    numeric_cols = get_quantitative_numeric_columns(df)
    additive_cols = [c for c in numeric_cols if not is_non_additive_metric(c)]
    candidate_cols = additive_cols or numeric_cols
    metric_col: str | None = None
    preferred_names = ("revenue", "sales", "amount", "profit", "total", "units_sold")
    for pref in preferred_names:
        for col in candidate_cols:
            if col.lower() == pref or pref in tokenize_col_name(col):
                metric_col = col
                break
        if metric_col is not None:
            break

    if metric_col is None and candidate_cols:
        metric_col = candidate_cols[0]

    return date_col, metric_col


def compute_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Compute high-level summary statistics for a DataFrame."""
    numeric_cols = get_quantitative_numeric_columns(df)
    date_cols = [c for c in df.columns if _is_date_like_series(c, df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]
    total_cells = max(1, int(df.shape[0] * df.shape[1]))
    missing_cells = int(df.isna().sum().sum())

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "categorical_columns": categorical_cols,
        "missing_cells": missing_cells,
        "missing_pct": round((missing_cells / total_cells) * 100.0, 2),
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024.0, 1),
    }


def compute_column_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column profiling summary table for inspection."""
    rows: list[dict[str, Any]] = []
    total = max(1, len(df))

    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        uniques = int(s.nunique(dropna=True))

        if is_quantitative_numeric_series(col, s) and s.notna().any():
            col_type = "NUMERIC"
            summary = f"min={s.min():,.2f} · mean={s.mean():,.2f} · max={s.max():,.2f}"
        elif _is_date_like_series(col, s) and s.notna().any():
            col_type = "DATE"
            summary = f"{s.min()} → {s.max()}"
        elif pd.api.types.is_bool_dtype(s) or is_binary_bool_indicator(col, s):
            col_type = "BOOL"
            top_vals = s.dropna().astype(str).value_counts().head(3).index.tolist()
            summary = ", ".join(top_vals) if top_vals else "—"
        elif is_id_or_code_column(col):
            col_type = "ID"
            top_vals = s.dropna().astype(str).value_counts().head(3).index.tolist()
            summary = ", ".join(top_vals) if top_vals else "—"
        else:
            col_type = "TEXT"
            top_vals = s.dropna().astype(str).value_counts().head(3).index.tolist()
            summary = ", ".join(top_vals) if top_vals else "—"

        rows.append(
            {
                "column": col,
                "type": col_type,
                "non_null": int(len(df) - nulls),
                "missing_pct": round((nulls / total) * 100.0, 1),
                "unique": uniques,
                "profile": summary,
            }
        )

    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix across quantitative numeric columns."""
    numeric_cols = [
        c
        for c in get_quantitative_numeric_columns(df)
        if df[c].nunique(dropna=True) > 1
    ]
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(method="pearson").round(2)


def extract_key_correlations(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract notable positive and negative linear correlations for executive highlight cards."""
    corr = compute_correlation_matrix(df)
    if corr.empty or len(corr.columns) < 2:
        return []

    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = float(corr.iloc[i, j])
            if not pd.isna(r):
                pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "r": round(r, 2),
                    "col1_label": c1.replace("_", " ").title(),
                    "col2_label": c2.replace("_", " ").title(),
                })

    if not pairs:
        return []

    # Sort by absolute strength
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return pairs[:4]


def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """Render a punchy, theme-adaptive Pearson correlation matrix with high-contrast cells."""
    corr = compute_correlation_matrix(df)
    if corr.empty:
        return None

    labels = [c.replace("_", " ").title() for c in corr.columns]
    z_vals = corr.to_numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=labels,
            y=labels,
            text=z_vals,
            texttemplate="%{z:+.2f}",
            textfont={"color": "#FFFFFF", "size": 12, "family": "Inter, sans-serif"},
            zmin=-1.0,
            zmax=1.0,
            xgap=5,
            ygap=5,
            colorscale=[
                [0.0, COLOR_ACCENT],
                [0.5, COLOR_NEUTRAL],
                [1.0, COLOR_PRIMARY],
            ],
            colorbar={
                "title": {"text": "r", "font": {"size": 12}},
                "tickfont": {"size": 11},
                "outlinewidth": 0,
                "thickness": 12,
            },
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Pearson r = %{z:+.2f}<extra></extra>",
        )
    )

    apply_neutral_layout(fig, title="Feature Correlation Matrix (Pearson r)", height=410, top_margin=68)
    fig.update_xaxes(showgrid=False, tickangle=-15)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def detect_timeseries_anomalies(
    df: pd.DataFrame,
    date_col: str | None = None,
    metric_col: str | None = None,
    contamination: float = 0.04,
) -> tuple[pd.DataFrame, go.Figure]:
    """Aggregate time-series metrics and detect anomalies via IsolationForest + rolling Z-score.

    Returns:
        `(ts_df, fig)` where `ts_df` contains daily values, `rolling_mean`, `rolling_zscore`,
        `anomaly_score`, and boolean `is_anomaly`, and `fig` is a styled, theme-adaptive Plotly figure.
    """
    auto_date, auto_metric = find_default_timeseries_columns(df)
    date_col = date_col or auto_date
    metric_col = metric_col or auto_metric

    if not date_col or not metric_col or date_col not in df.columns or metric_col not in df.columns:
        raise ValueError("Valid date and numeric metric columns are required for anomaly detection.")
    if pd.api.types.is_bool_dtype(df[metric_col]) or is_binary_bool_indicator(metric_col, df[metric_col]):
        raise ValueError(f"Boolean column `{metric_col}` cannot be analyzed as a quantitative time-series metric.")

    working = df[[date_col, metric_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], format="mixed", errors="coerce")
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[date_col, metric_col])

    if working.empty:
        raise ValueError(f"No valid time-series observations found for `{date_col}` and `{metric_col}`.")

    is_non_add = is_non_additive_metric(metric_col)
    agg_fn = "mean" if is_non_add else "sum"
    agg_prefix = "Daily Avg" if is_non_add else "Daily"

    ts_df = (
        working.groupby(date_col, as_index=False)[metric_col]
        .agg(agg_fn)
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    values = ts_df[metric_col].astype(float)
    rolling_mean = values.rolling(window=14, min_periods=3, center=False).mean().bfill()
    rolling_std = values.rolling(window=14, min_periods=3, center=False).std().bfill()
    global_std = float(values.std()) if float(values.std()) > 0 else 1.0
    safe_std = rolling_std.replace(0.0, np.nan).fillna(global_std)

    rolling_z = ((values - rolling_mean) / safe_std).fillna(0.0)
    pct_delta = values.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    ts_df["rolling_mean"] = rolling_mean.round(2)
    ts_df["rolling_zscore"] = rolling_z.round(2)

    if len(ts_df) >= 10:
        features = np.column_stack(
            [
                values.to_numpy(),
                rolling_z.to_numpy(),
                pct_delta.to_numpy(),
            ]
        )
        effective_contam = float(np.clip(contamination, 0.01, 0.20))
        iso = IsolationForest(
            contamination=effective_contam,
            random_state=42,
            n_estimators=150,
        )
        preds = iso.fit_predict(features)
        scores = iso.decision_function(features)
        ts_df["anomaly_score"] = np.round(scores, 4)
        # Flag extreme statistical spikes/drops alongside strong IsolationForest outliers
        ts_df["is_anomaly"] = ((preds == -1) & (rolling_z.abs() >= 2.0)) | (rolling_z.abs() >= 3.0)
        # Ensure at least the top IsolationForest outliers are surfaced if none crossed |z|>=2.0
        if not bool(ts_df["is_anomaly"].any()):
            ts_df["is_anomaly"] = preds == -1
    else:
        ts_df["anomaly_score"] = 0.0
        ts_df["is_anomaly"] = rolling_z.abs() >= 2.2

    metric_label = metric_col.replace("_", " ").title()
    y_prefix, y_suffix = get_metric_unit_affixes(metric_col)
    anomalies = ts_df[ts_df["is_anomaly"]].copy()

    # Calculate statistical confidence corridor (rolling_mean +/- 2 sigma)
    upper_corridor = (rolling_mean + 2.0 * safe_std).clip(lower=0.0)
    lower_corridor = (rolling_mean - 2.0 * safe_std).clip(lower=0.0)

    fig = go.Figure()

    # 1. Expected normal corridor (shaded area)
    fig.add_trace(
        go.Scatter(
            x=ts_df[date_col],
            y=upper_corridor,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df[date_col],
            y=lower_corridor,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(2, 132, 199, 0.08)",
            name="Normal range (±2σ)",
            hoverinfo="skip",
        )
    )

    # 2. Main daily metric trend line
    fig.add_trace(
        go.Scatter(
            x=ts_df[date_col],
            y=ts_df[metric_col],
            mode="lines",
            name=f"{agg_prefix} {metric_label}",
            line={"color": "#0284C7", "width": 3.0, "shape": "spline", "smoothing": 0.85},
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.05)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>"
            + f"{metric_label}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
        )
    )

    # 3. 14-day rolling baseline
    fig.add_trace(
        go.Scatter(
            x=ts_df[date_col],
            y=ts_df["rolling_mean"],
            mode="lines",
            name="14-Day Baseline",
            line={"color": COLOR_MUTED, "width": 1.6, "dash": "dash"},
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>14d Baseline: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><extra></extra>",
        )
    )

    # 4. Outlier markers with glowing halo effect
    if not anomalies.empty:
        # Outer luminous halo
        fig.add_trace(
            go.Scatter(
                x=anomalies[date_col],
                y=anomalies[metric_col],
                mode="markers",
                marker={
                    "color": "rgba(255, 90, 95, 0.28)",
                    "size": 20,
                    "symbol": "circle",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
        # Inner sharp alert core
        fig.add_trace(
            go.Scatter(
                x=anomalies[date_col],
                y=anomalies[metric_col],
                mode="markers",
                name="Flagged Anomaly",
                customdata=anomalies["rolling_zscore"],
                marker={
                    "color": COLOR_DANGER,
                    "size": 9,
                    "symbol": "circle",
                    "line": {"color": "#FFFFFF", "width": 2},
                },
                hovertemplate="<b>Anomaly · %{x|%Y-%m-%d}</b><br>"
                + f"{metric_label}: <b>{y_prefix}%{{y:,.2f}}{y_suffix}</b><br>Deviation: <b>%{{customdata:+.2f}}σ</b><extra></extra>",
            )
        )

        # Annotate the single most extreme deviation
        try:
            worst_idx = anomalies["rolling_zscore"].abs().idxmax()
            worst_row = anomalies.loc[worst_idx]
            fig.add_annotation(
                x=worst_row[date_col],
                y=float(worst_row[metric_col]),
                text=f"Severe Outlier: {float(worst_row['rolling_zscore']):+.1f}σ",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=COLOR_DANGER,
                ax=0,
                ay=-36,
                bordercolor=COLOR_DANGER,
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(255, 90, 95, 0.14)",
                font=dict(size=11, family="Inter, sans-serif"),
            )
        except Exception:
            pass

    apply_neutral_layout(
        fig,
        title=f"{agg_prefix} {metric_label} · {len(anomalies)} Outlier Day(s) Flagged",
        height=400,
        top_margin=88,
    )
    fig.update_xaxes(title="", showgrid=False)
    fig.update_yaxes(title=metric_label)
    fig.update_layout(hovermode="closest")

    ts_df[date_col] = ts_df[date_col].dt.strftime("%Y-%m-%d")
    return ts_df, fig
