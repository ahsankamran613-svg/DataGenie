"""DataGenie — Autonomous BI & Data Analyst Web Application.

A clean, dashboard-style Streamlit interface with native light & dark themes:
- Header + KPI strip: dataset identity, model status, and month-over-month headline tiles.
- Tab 1 (Ask): Plain-English business Q&A with executive insights, interactive Plotly
  charts, an agent pipeline strip, and a tucked-away "Under the hood" SQL & stats trace.
- Tab 2 (Data health): Automated CSV cleaning summary, IsolationForest time-series
  anomaly detection, numeric correlation analysis, and column profiles.

Themes are configured in `.streamlit/config.toml` (`[theme.light]` / `[theme.dark]`).
Viewers switch between them via the ⋮ menu → Settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from src.agent_graph import (
    SUPPORTED_GROQ_MODELS,
    AnalystResult,
    fetch_available_groq_models,
    run_analyst_query,
)
from src.data_cleaning import CleaningReport, clean_dataframe
from src.db_manager import DatabaseManager
from src.demo_data import load_or_create_demo_csv
from src.eda_engine import (
    build_correlation_heatmap,
    compute_column_profiles,
    detect_timeseries_anomalies,
    extract_key_correlations,
    find_default_timeseries_columns,
)
from src.stats_analyzer import get_quantitative_numeric_columns
from src.ui import (
    APP_CSS,
    brand_html,
    compute_headline_kpis,
    correlation_insights_html,
    date_range_label,
    esc,
    get_app_css,
    hero_html,
    pipeline_html,
    section_header_html,
)
import importlib
import src.ui
importlib.reload(src.ui)


load_dotenv()

st.set_page_config(
    page_title="DataGenie · AI Business Analyst",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(src.ui.get_app_css(), unsafe_allow_html=True)


DEMO_STARTER_QUESTIONS = [
    (":material/troubleshoot:", "Why did revenue drop in the North region last month?"),
    (":material/pie_chart:", "Show units sold share by product category."),
    (":material/show_chart:", "Show monthly revenue trend by region."),
]

GENERIC_STARTER_QUESTIONS = [
    (":material/summarize:", "Give me a high-level summary of the main metrics in this dataset."),
    (":material/leaderboard:", "What are the top performing categories or segments?"),
    (":material/show_chart:", "Show how the primary metric trends over time."),
]

SOURCE_DEMO = "Demo dataset"
SOURCE_CSV = "Upload CSV"
SOURCE_PG = "PostgreSQL / Supabase"


def _get_secret_or_env(key: str) -> str:
    """Safely read a configuration value from Streamlit secrets or environment variables."""
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    val = os.getenv(key, "").strip()
    if val and not val.startswith("gsk_your_"):
        return val
    return ""


@st.cache_data(ttl=600, show_spinner=False)
def _cached_model_list(api_key: str) -> list[str]:
    """Cache the live Groq model lookup for 10 minutes so reruns stay snappy."""
    return fetch_available_groq_models(api_key)


@st.cache_data(show_spinner=False)
def _cached_headline_kpis(df: pd.DataFrame | None) -> list[Any]:
    return compute_headline_kpis(df)


@st.cache_data(show_spinner=False)
def _cached_anomaly_detection(df: pd.DataFrame, date_col: str, metric_col: str) -> tuple[pd.DataFrame, Any]:
    return detect_timeseries_anomalies(df, date_col=date_col, metric_col=metric_col)


@st.cache_data(show_spinner=False)
def _cached_correlation_analysis(df: pd.DataFrame) -> tuple[Any, list[dict[str, Any]]]:
    fig = build_correlation_heatmap(df)
    pairs = extract_key_correlations(df)
    return fig, pairs


@st.cache_data(show_spinner=False)
def _cached_column_profiles(df: pd.DataFrame) -> pd.DataFrame:
    return compute_column_profiles(df)


def _reset_active_db(new_db: DatabaseManager) -> None:
    """Safely close any previous session DatabaseManager and store the new one."""
    old_db: DatabaseManager | None = st.session_state.get("db_manager")
    if old_db is not None and old_db is not new_db:
        old_db.close()
    st.session_state["db_manager"] = new_db


def _load_demo_dataset() -> None:
    """Load and clean the built-in Retail Regional Sales demo dataset."""
    raw_df = load_or_create_demo_csv()
    clean_df, report = clean_dataframe(raw_df)
    db = DatabaseManager.from_dataframe(
        clean_df,
        table_name="sales_data",
        source_label="Demo Retail Sales",
    )
    _reset_active_db(db)
    st.session_state["active_source"] = "demo"
    st.session_state["raw_df"] = raw_df
    st.session_state["clean_df"] = clean_df
    st.session_state["cleaning_report"] = report
    st.session_state["active_dataset_name"] = "Retail & E-Commerce Regional Sales"
    st.session_state["messages"] = []


def _load_uploaded_csv(uploaded_file: Any) -> None:
    """Read, auto-clean, and index a user-uploaded CSV into an isolated SQLite DB."""
    raw_df = pd.read_csv(uploaded_file)
    clean_df, report = clean_dataframe(raw_df)
    stem = Path(uploaded_file.name).stem
    db = DatabaseManager.from_dataframe(
        clean_df,
        table_name=stem or "uploaded_data",
        source_label=f"CSV · {uploaded_file.name}",
    )
    _reset_active_db(db)
    st.session_state["active_source"] = "csv"
    st.session_state["uploaded_signature"] = f"{uploaded_file.name}:{uploaded_file.size}"
    st.session_state["raw_df"] = raw_df
    st.session_state["clean_df"] = clean_df
    st.session_state["cleaning_report"] = report
    st.session_state["active_dataset_name"] = uploaded_file.name
    st.session_state["messages"] = []


def _connect_external_database(db_url: str) -> None:
    """Connect to a live PostgreSQL / Supabase database in read-only mode."""
    db = DatabaseManager.from_connection_url(
        db_url,
        source_label="PostgreSQL / Supabase",
    )
    tables = db.get_table_names()
    sample_df = pd.DataFrame()
    if tables:
        try:
            sample_df = db.run_safe_query(f'SELECT * FROM "{tables[0]}" LIMIT 500')
        except Exception:
            sample_df = pd.DataFrame()

    _reset_active_db(db)
    st.session_state["active_source"] = "postgres"
    st.session_state["raw_df"] = None
    st.session_state["clean_df"] = sample_df if not sample_df.empty else None
    st.session_state["cleaning_report"] = None
    st.session_state["active_dataset_name"] = f"Live database · {len(tables)} tables"
    st.session_state["messages"] = []


# Initialize session state with the Demo Dataset ready out of the box
if "db_manager" not in st.session_state:
    _load_demo_dataset()
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str]:
    """Render the left sidebar and return `(resolved_api_key, selected_model)`."""
    with st.sidebar:
        st.markdown(brand_html(), unsafe_allow_html=True)
        st.space("small")

        # 1. Data source
        source_choice = st.radio(
            "Data source",
            options=[SOURCE_DEMO, SOURCE_CSV, SOURCE_PG],
            captions=[
                "650 retail orders, May–Oct 2026",
                "Messy files welcome, we clean them",
                "Live connection, strictly read-only",
            ],
            index=0,
        )

        if source_choice == SOURCE_DEMO:
            if st.session_state.get("active_source") != "demo":
                _load_demo_dataset()

        elif source_choice == SOURCE_CSV:
            uploaded = st.file_uploader(
                "Upload a CSV file",
                type=["csv"],
                help="Duplicates, currency symbols ($), mixed dates, and missing values are cleaned automatically.",
            )
            if uploaded is not None:
                sig = f"{uploaded.name}:{uploaded.size}"
                if st.session_state.get("uploaded_signature") != sig:
                    with st.spinner("Cleaning and indexing CSV..."):
                        try:
                            _load_uploaded_csv(uploaded)
                            st.toast(
                                f"Loaded {uploaded.name} · {len(st.session_state['clean_df']):,} rows",
                                icon=":material/check_circle:",
                            )
                        except Exception as exc:
                            st.error(f"Could not read CSV: {exc}")

        elif source_choice == SOURCE_PG:
            default_db_url = _get_secret_or_env("DATABASE_URL")
            db_uri = st.text_input(
                "Connection URI",
                value=default_db_url,
                type="password",
                placeholder="postgresql://user:pass@host:5432/postgres",
                help="Supabase: Project Settings → Database → Connection string (URI). Queries are strictly read-only.",
            )
            st.caption(":material/lock: `DROP`, `DELETE`, and `UPDATE` are blocked.")
            if st.button("Connect", type="primary", width="stretch", icon=":material/power:"):
                if not db_uri.strip():
                    st.warning("Please enter a PostgreSQL connection string.")
                else:
                    with st.spinner("Connecting to database..."):
                        try:
                            _connect_external_database(db_uri)
                            st.toast(
                                f"Connected · {st.session_state['active_dataset_name']}",
                                icon=":material/check_circle:",
                            )
                        except Exception as exc:
                            st.error(f"Connection failed: {exc}")

        st.divider()

        # 2. Model & API key
        env_key = _get_secret_or_env("GROQ_API_KEY")
        manual_key = st.session_state.get("manual_groq_key", "").strip()
        effective_key = manual_key or env_key

        with st.expander("Model & API key", icon=":material/tune:", expanded=not bool(effective_key)):
            user_key_input = st.text_input(
                "Groq API key",
                value=manual_key,
                type="password",
                placeholder="gsk_..." if not env_key else "Using configured key (override optional)",
                help="Get a free API key at https://console.groq.com/keys",
            )
            st.session_state["manual_groq_key"] = user_key_input.strip()
            effective_key = user_key_input.strip() or env_key

            available_models = _cached_model_list(effective_key) if effective_key else list(SUPPORTED_GROQ_MODELS)
            prev_choice = st.session_state.get("groq_model_choice")
            if prev_choice not in available_models:
                st.session_state.pop("groq_model_choice", None)

            selected_model = st.selectbox(
                "Model",
                options=available_models,
                key="groq_model_choice",
                help="Live list of chat models available to your Groq key.",
            )

        if effective_key:
            st.badge("Analyst ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("Groq API key needed", icon=":material/key:", color="orange")

        # 3. Schema viewer
        db_mgr: DatabaseManager | None = st.session_state.get("db_manager")
        if db_mgr is not None:
            schema_meta = db_mgr.get_schema_metadata()
            with st.expander("Table schema", icon=":material/table_chart:", expanded=False):
                for tbl in schema_meta:
                    st.markdown(f"**`{tbl['table_name']}`** · {tbl['row_count']:,} rows")
                    st.markdown(
                        "\n".join(f"- `{c['name']}` <small>{c['badge']}</small>" for c in tbl["columns"]),
                        unsafe_allow_html=True,
                    )

        if st.session_state.get("messages"):
            if st.button("Clear conversation", width="stretch", icon=":material/delete_sweep:"):
                st.session_state["messages"] = []
                st.rerun()

        st.space("medium")
        st.markdown(
            '<div class="dg-foot">Light or dark? <kbd>⋮</kbd> → <kbd>Settings</kbd> → Theme'
            "<br>LangGraph · Groq · Streamlit</div>",
            unsafe_allow_html=True,
        )

    return effective_key, selected_model


# ---------------------------------------------------------------------------
# Header & KPI strip
# ---------------------------------------------------------------------------

def render_header(api_key: str, model_name: str) -> None:
    """Dataset identity, status chips, and month-over-month headline tiles."""
    clean_df: pd.DataFrame | None = st.session_state.get("clean_df")
    source = st.session_state.get("active_source", "demo")
    breadcrumb = {
        "demo": "Demo Retail Workspace",
        "csv": "Uploaded CSV",
        "postgres": "Live PostgreSQL Connection",
    }.get(source, "Workspace")

    pills: list[tuple[str, str]] = []
    if clean_df is not None:
        pills.append((f"{len(clean_df):,} rows × {len(clean_df.columns)} columns", ""))
    rng = date_range_label(clean_df)
    if rng:
        pills.append((rng, ""))
    pills.append((f"Model: {model_name}", "ok") if api_key else ("API Key Missing", "warn"))

    st.markdown(
        hero_html(breadcrumb, st.session_state.get("active_dataset_name", "Dataset"), pills),
        unsafe_allow_html=True,
    )

    kpis = _cached_headline_kpis(clean_df)
    if kpis:
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            with col:
                st.metric(
                    kpi.label,
                    kpi.value,
                    delta=kpi.delta,
                    delta_description=kpi.delta_description,
                    border=True,
                    chart_data=kpi.chart or None,
                    chart_type="area",
                    help=kpi.help,
                )


# ---------------------------------------------------------------------------
# Tab 1: Ask
# ---------------------------------------------------------------------------

def render_answer(result: AnalystResult, key_suffix: str) -> None:
    """Render one assistant answer: summary, chart, pipeline strip, and the trace panel."""
    st.markdown(result.executive_summary)
    if result.figure is not None:
        st.plotly_chart(result.figure, width="stretch", key=f"chart_{key_suffix}")
    if result.trace_steps:
        st.markdown(pipeline_html(result.trace_steps), unsafe_allow_html=True)
    render_trace_expander(result, key_suffix=key_suffix)


def render_trace_expander(result: AnalystResult, key_suffix: str) -> None:
    """Render the collapsed 'Under the hood' transparency panel."""
    retry_note = f" · {result.retry_count} self-correction(s)" if result.retry_count > 0 else ""
    with st.expander(f"Under the hood{retry_note}", icon=":material/code:", expanded=False):
        tab_titles = ["SQL", "Result table", "Stats & steps"]
        if getattr(result, "chart_code", None):
            tab_titles.append("Plotly code")
        sub_tabs = st.tabs(tab_titles)

        with sub_tabs[0]:
            if result.sql_explanation:
                st.caption(result.sql_explanation)
            st.code(result.sql_query or "-- No SQL generated", language="sql")

        with sub_tabs[1]:
            if not result.dataframe.empty:
                st.dataframe(result.dataframe, width="stretch", hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=result.dataframe.to_csv(index=False).encode("utf-8"),
                    file_name="datagenie_result.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key=f"dl_{key_suffix}",
                )
            else:
                st.caption("No rows returned.")

        with sub_tabs[2]:
            stats_lines = result.stats_summary.get("summary_lines", [])
            if stats_lines:
                st.markdown("**Verified Python statistics**")
                st.markdown("\n".join(f"- {line}" for line in stats_lines))

            if result.trace_steps:
                st.markdown("**Agent steps**")
                for step in result.trace_steps:
                    icon = {
                        "done": ":material/check_circle:",
                        "retry": ":material/autorenew:",
                    }.get(step["status"], ":material/error:")
                    st.caption(f"{icon} **{step['node']}** · {step['duration_ms']} ms — {step['detail']}")

        if len(sub_tabs) > 3:
            with sub_tabs[3]:
                st.caption("Sandboxed AI-generated Python Plotly code:")
                st.code(result.chart_code, language="python")


def render_analyst_tab(api_key: str, model_name: str) -> None:
    """Render Tab 1: conversational AI business analyst."""
    messages: list[dict[str, Any]] = st.session_state["messages"]
    queued_prompt: str | None = None

    if not messages:
        st.markdown(
            '<div class="dg-empty-state">'
            '<div class="headline">What would you like to explore?</div>'
            '<div class="subtext">Ask any business question in plain English. DataGenie writes verified SQL, validates rows, crunches Python statistics, and builds executive charts.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        starters = (
            DEMO_STARTER_QUESTIONS
            if st.session_state.get("active_source") == "demo"
            else GENERIC_STARTER_QUESTIONS
        )
        with st.container(key="starters"):
            cols = st.columns(len(starters))
            for idx, (icon, q_text) in enumerate(starters):
                with cols[idx]:
                    if st.button(q_text, key=f"starter_{idx}", icon=icon, width="stretch"):
                        queued_prompt = q_text

    for idx, msg in enumerate(messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar=":material/query_stats:"):
                render_answer(msg["result"], key_suffix=f"hist_{idx}")

    # Question bar anchored at the bottom with quick actions dot button
    col_input, col_dot = st.columns([12, 1], vertical_alignment="bottom")
    with col_input:
        chat_prompt = st.chat_input("Ask about revenue, margins, trends, regions…")
    with col_dot:
        with st.popover("•••", help="Quick options & model info"):
            st.markdown(f"**Model**: `{model_name}`")
            if st.button("🧹 Clear chat", width="stretch", key="popover_clear"):
                st.session_state["messages"] = []
                st.rerun()

    active_question = queued_prompt or chat_prompt

    if not active_question:
        return

    if not api_key:
        st.warning(
            "Add your **Groq API key** in the sidebar (**Model & API key**) to start asking questions.",
            icon=":material/key:",
        )
        return

    # Render newly asked question
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(active_question)

    with st.chat_message("assistant", avatar=":material/query_stats:"):
        col_stat, col_stop = st.columns([5, 1], vertical_alignment="center")
        with col_stop:
            stop_clicked = st.button("⏹ Stop", key=f"stop_{len(messages)}", type="secondary", help="Stop generation")
        with col_stat:
            status_box = st.status("Thinking it through…", expanded=False)

        if stop_clicked:
            status_box.update(label="Analysis cancelled", state="error", expanded=False)
            st.toast("Evaluation stopped by user.", icon=":material/stop_circle:")
            st.rerun()
            return

        step_labels = {
            "sql_writer": "Writing SQL…",
            "sql_critic": "Checking & running the query…",
            "stats_and_chart": "Crunching numbers & drawing a chart…",
            "insight_analyst": "Writing up the findings…",
        }

        def _on_step(node_name: str, node_update: dict[str, Any]) -> None:
            if node_name == "sql_critic" and node_update.get("sql_error"):
                attempt = node_update.get("retry_count", 1)
                status_box.update(label=f"Fixing the query (attempt {attempt}/3)…")
            else:
                status_box.update(label=step_labels.get(node_name, "Analyzing…"))

        try:
            result = run_analyst_query(
                question=active_question,
                db_manager=st.session_state["db_manager"],
                api_key=api_key,
                model_name=model_name,
                on_step_callback=_on_step,
            )
            total_ms = sum(int(s.get("duration_ms", 0)) for s in result.trace_steps)
            status_box.update(label=f"Done in {total_ms / 1000:.1f}s", state="complete", expanded=False)
        except Exception as exc:
            status_box.update(label="Analysis failed", state="error", expanded=True)
            err_msg = str(exc)
            st.error(f"Could not complete analysis: {err_msg}")
            if "model_not_found" in err_msg or "does not exist" in err_msg:
                st.info(
                    "Switch the **Model** in the sidebar (under **Model & API key**) to "
                    "`openai/gpt-oss-120b` or `qwen/qwen3.8-27b`.",
                    icon=":material/lightbulb:",
                )
            return

        render_answer(result, key_suffix=f"live_{len(messages)}")
        messages.append({"role": "user", "content": active_question})
        messages.append({"role": "assistant", "result": result})
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Data health
# ---------------------------------------------------------------------------

def render_data_health_tab() -> None:
    """Render Tab 2: Data health & automated EDA."""
    report: CleaningReport | None = st.session_state.get("cleaning_report")
    raw_df: pd.DataFrame | None = st.session_state.get("raw_df")
    clean_df: pd.DataFrame | None = st.session_state.get("clean_df")
    db_mgr: DatabaseManager = st.session_state["db_manager"]

    # When connected to external PostgreSQL, allow picking any table for Auto-EDA preview
    if st.session_state.get("active_source") == "postgres":
        tables = db_mgr.get_table_names()
        if not tables:
            st.info("Connected database has no user tables to profile.")
            return
        selected_tbl = st.selectbox("Table to profile (500-row sample)", options=tables)
        try:
            clean_df = db_mgr.run_safe_query(f'SELECT * FROM "{selected_tbl}" LIMIT 500')
        except Exception as exc:
            st.error(f"Could not load sample from `{selected_tbl}`: {exc}")
            return

    if clean_df is None or clean_df.empty:
        st.info("Load a dataset from the sidebar to view data health and automated insights.")
        return

    # 1. Cleaning summary
    if report is not None:
        st.markdown(
            section_header_html(
                "Automated Data Cleaning",
                "Heuristic corrections applied to sanitize types, impute missing values, and trim duplicates.",
            ),
            unsafe_allow_html=True,
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Clean Rows",
            f"{report.rows_after:,}",
            delta=f"from {report.rows_before:,} raw",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        k2.metric("Duplicates Removed", f"{report.duplicates_removed:,}", border=True)
        k3.metric("Missing Values Fixed", f"{report.missing_before - report.missing_after:,}", border=True)
        k4.metric("Standardized Columns", f"{report.columns_standardized:,}", border=True)

        with st.expander("View Cleaning Action Log & Raw vs. Clean Comparison", icon=":material/cleaning_services:"):
            if report.actions:
                st.markdown("\n".join(f"- {act.message}" for act in report.actions))
            else:
                st.caption("Dataset was already clean — no transformations needed.")

            prev_clean_tab, prev_raw_tab = st.tabs(["Cleaned (first 20 rows)", "Raw input (first 20 rows)"])
            with prev_clean_tab:
                st.dataframe(clean_df.head(20), width="stretch", hide_index=True)
            with prev_raw_tab:
                if raw_df is not None:
                    st.dataframe(raw_df.head(20), width="stretch", hide_index=True)

    # 2. Anomaly detection
    default_date, default_metric = find_default_timeseries_columns(clean_df)
    numeric_cols = get_quantitative_numeric_columns(clean_df)

    if default_date and numeric_cols:
        st.markdown(
            section_header_html(
                "Time-Series Anomaly Detection",
                "IsolationForest combined with rolling statistical corridors (±2σ) to flag demand shocks and outliers.",
            ),
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            metric_idx = numeric_cols.index(default_metric) if default_metric in numeric_cols else 0
            chosen_metric = st.selectbox(
                "Metric to monitor",
                options=numeric_cols,
                index=metric_idx,
                key="eda_anomaly_metric",
                format_func=lambda c: c.replace("_", " ").title(),
            )

            try:
                ts_df, anom_fig = _cached_anomaly_detection(
                    clean_df,
                    date_col=default_date,
                    metric_col=chosen_metric,
                )
                flagged = ts_df[ts_df["is_anomaly"]]

                # Summary metric indicators above the chart
                s1, s2, s3 = st.columns(3)
                total_days = len(ts_df)
                anom_count = len(flagged)
                anom_pct = (anom_count / max(1, total_days)) * 100
                worst_sigma = float(flagged["rolling_zscore"].abs().max()) if not flagged.empty else 0.0

                s1.metric("Monitored Period", f"{total_days:,} Days", help="Unique chronological dates analyzed.")
                s2.metric("Outliers Flagged", f"{anom_count:,} ({anom_pct:.1f}%)", help="Days flagged by IsolationForest & |z| >= 2.0σ.")
                s3.metric("Peak Outlier Shock", f"{worst_sigma:.1f}σ Deviation" if not flagged.empty else "None", help="Highest statistical deviation magnitude.")

                st.plotly_chart(anom_fig, width="stretch", key="eda_anomaly_chart")

                if not flagged.empty:
                    with st.expander(f"Inspect Flagged Outlier Days ({len(flagged)} records)", icon=":material/flag:"):
                        flagged_table = flagged[[default_date, chosen_metric, "rolling_mean", "rolling_zscore"]].copy()
                        flagged_table.columns = ["Date", "Recorded Value", "14-Day Baseline", "Deviation (σ)"]
                        st.dataframe(flagged_table, width="stretch", hide_index=True)
            except Exception as exc:
                st.caption(f"Time-series anomaly chart unavailable for this column selection: {exc}")

    # 3. Relationships & Column Profiles
    st.markdown(
        section_header_html(
            "Feature Relationships & Data Completeness",
            "Identify strong linear drivers, negative trade-offs, and data quality across all table columns.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            corr_fig, key_pairs = _cached_correlation_analysis(clean_df)
            if corr_fig is not None:
                st.plotly_chart(corr_fig, width="stretch", key="eda_corr_chart")
                if key_pairs:
                    st.markdown("**Key Correlation Drivers**")
                    st.markdown(correlation_insights_html(key_pairs), unsafe_allow_html=True)
            else:
                st.caption("At least two numeric columns are required to compute a correlation matrix.")
    with right:
        profiles_df = _cached_column_profiles(clean_df)
        st.dataframe(
            profiles_df,
            width="stretch",
            height=540,
            hide_index=True,
            column_config={
                "column": st.column_config.TextColumn("Column", pinned=True),
                "type": st.column_config.TextColumn("Type", width="small"),
                "non_null": st.column_config.NumberColumn("Non-null", format="%d"),
                "missing_pct": st.column_config.ProgressColumn(
                    "Missing", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
                "unique": st.column_config.NumberColumn("Unique", format="%d"),
                "profile": st.column_config.TextColumn("Profile", width="large"),
            },
        )


def main() -> None:
    """Main Streamlit entrypoint."""
    api_key, model_name = render_sidebar()

    # Prominent Top Navigation Tabs
    tab_chat, tab_health = st.tabs([":material/forum: Ask DataGenie", ":material/monitor_heart: Data health"])

    with tab_chat:
        render_header(api_key, model_name)
        render_analyst_tab(api_key=api_key, model_name=model_name)

    with tab_health:
        render_header(api_key, model_name)
        render_data_health_tab()


if __name__ == "__main__":
    main()
