"""Comprehensive automated verification suite for DataGenie (Phases 1–5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.agent_graph import (
    SUPPORTED_GROQ_MODELS,
    escape_currency_dollars,
    fetch_available_groq_models,
    run_analyst_query,
)
from src.data_cleaning import clean_dataframe, to_snake_case
from src.db_manager import (
    DatabaseManager,
    UnsafeQueryError,
    _badge_type,
    normalize_database_url,
    validate_readonly_sql,
)
from src.demo_data import generate_messy_sales_dataframe, load_or_create_demo_csv
from src.eda_engine import (
    build_correlation_heatmap,
    compute_column_profiles,
    compute_correlation_matrix,
    compute_dataset_overview,
    detect_timeseries_anomalies,
    find_default_timeseries_columns,
)
from src.stats_analyzer import (
    ChartSpec,
    _identify_columns,
    build_plotly_chart,
    classify_metric_unit,
    execute_sandboxed_plotly_code,
    format_metric_value,
    get_metric_unit_affixes,
    get_quantitative_numeric_columns,
    infer_default_chart_spec,
    is_binary_bool_indicator,
    is_id_or_code_column,
    is_non_additive_metric,
    is_plotly_code_safe,
    run_statistical_checks,
)
from src.ui import _fmt_number, compute_headline_kpis


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _MockSelfCorrectingChatModel(BaseChatModel):
    """Deterministic mock LLM that tests SQL Critic -> SQL Writer self-correction."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "mock-datagenie-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        prompt_text = "\n".join(str(m.content) for m in messages)

        # 1st SQL attempt intentionally references a nonexistent column `nonexistent_rev`
        if "Senior BI SQL Engineer" in prompt_text and "CRITIC ERROR" not in prompt_text:
            content = (
                '{"sql_query": "SELECT region, SUM(nonexistent_rev) AS rev FROM sales_data GROUP BY region", '
                '"explanation": "Initial attempt with wrong column name."}'
            )
        # 2nd SQL attempt fixes the SQL after Critic feedback
        elif "Senior BI SQL Engineer" in prompt_text and "CRITIC ERROR" in prompt_text:
            content = (
                '{"sql_query": "SELECT region, ROUND(SUM(revenue), 2) AS total_revenue '
                'FROM sales_data GROUP BY region ORDER BY total_revenue DESC", '
                '"explanation": "Corrected column name to revenue."}'
            )
        else:
            content = (
                "**North leads total regional sales at $950,000 across the 6-month window.**\n\n"
                "- **North** and **South** account for over half of company revenue.\n"
                "- **Recommendation:** Prioritize inventory replenishment in top regions."
            )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def test_deployment_and_config_files_exist() -> None:
    """Verify Streamlit Cloud, Heroku, and Docker deployment configuration files."""
    assert (PROJECT_ROOT / "requirements.txt").exists()
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert (PROJECT_ROOT / ".streamlit" / "config.toml").exists()
    assert (PROJECT_ROOT / "Procfile").exists()
    assert (PROJECT_ROOT / "Dockerfile").exists()
    assert (PROJECT_ROOT / ".env.example").exists()
    assert (PROJECT_ROOT / ".gitignore").exists()
    assert (PROJECT_ROOT / "README.md").exists()
    assert (PROJECT_ROOT / "app.py").exists()


def test_demo_data_and_cleaning_pipeline(tmp_path: Path) -> None:
    """Verify messy demo CSV generation, Pandas 3.x cleaning, and embedded business story."""
    csv_file = tmp_path / "messy_regional_sales.csv"
    raw_df = load_or_create_demo_csv(csv_path=csv_file, force_rebuild=True)

    assert csv_file.exists()
    assert len(raw_df) == 668
    assert int(raw_df.isna().sum().sum()) == 74

    clean_df, report = clean_dataframe(raw_df)

    assert report.rows_before == 668
    assert report.rows_after == 650
    assert report.duplicates_removed == 18
    assert report.missing_before == 74
    assert report.missing_after == 0
    assert report.columns_standardized == 9
    assert int(clean_df.isna().sum().sum()) == 0

    expected_cols = {
        "order_id",
        "order_date",
        "region",
        "product_category",
        "units_sold",
        "discount_pct",
        "shipping_delay_days",
        "revenue",
        "profit",
    }
    assert set(clean_df.columns) == expected_cols
    assert pd.api.types.is_float_dtype(clean_df["revenue"])
    assert pd.api.types.is_float_dtype(clean_df["profit"])
    assert pd.api.types.is_float_dtype(clean_df["discount_pct"])

    assert set(clean_df["region"].unique()) == {"North", "South", "East", "West"}
    assert set(clean_df["product_category"].unique()) == {
        "Electronics",
        "Apparel",
        "Home & Kitchen",
        "Beauty & Health",
    }
    assert clean_df["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    north_sep = clean_df[
        (clean_df["region"] == "North")
        & (clean_df["order_date"].between("2026-09-01", "2026-09-30"))
    ]["revenue"].sum()
    north_oct = clean_df[
        (clean_df["region"] == "North")
        & (clean_df["order_date"].between("2026-10-01", "2026-10-31"))
    ]["revenue"].sum()
    pct_change = (north_oct - north_sep) / north_sep * 100.0
    assert -36.0 <= pct_change <= -28.0


def test_categorical_imputation_and_snake_case() -> None:
    """Verify categorical missing values fill with 'Unknown' under Pandas 3.x string dtypes."""
    assert to_snake_case("Gross Revenue ($)") == "gross_revenue"
    assert to_snake_case("Promo Discount (%)") == "promo_discount_pct"

    custom_messy = pd.DataFrame(
        {
            "Customer Segment ": [" enterprise ", "SMB", None, "Enterprise"],
            "Net Sales ($)": ["$1,200.50", "$800.00", "$950.25", None],
        }
    )
    cleaned, rep = clean_dataframe(custom_messy)
    assert list(cleaned.columns) == ["customer_segment", "net_sales"]
    assert "Unknown" in set(cleaned["customer_segment"])
    assert rep.missing_after == 0


def test_eda_and_anomaly_detection() -> None:
    """Verify dataset overview, 5x5 Pearson correlation heatmap, and IsolationForest anomalies."""
    raw_df = generate_messy_sales_dataframe(seed=42)
    clean_df, _ = clean_dataframe(raw_df)

    overview = compute_dataset_overview(clean_df)
    assert overview["rows"] == 650
    assert overview["columns"] == 9
    assert len(overview["numeric_columns"]) == 5

    profiles = compute_column_profiles(clean_df)
    assert len(profiles) == 9

    corr = compute_correlation_matrix(clean_df)
    assert corr.shape == (5, 5)

    heatmap_fig = build_correlation_heatmap(clean_df)
    assert heatmap_fig is not None

    ts_df, anomaly_fig = detect_timeseries_anomalies(
        clean_df, date_col="order_date", metric_col="revenue"
    )
    assert not ts_df.empty
    flagged = ts_df[ts_df["is_anomaly"]]
    assert len(flagged) >= 2
    flagged_dates = set(flagged["order_date"])
    assert "2026-06-18" in flagged_dates or "2026-08-24" in flagged_dates
    assert len(anomaly_fig.data) >= 3


def test_database_manager_safety_and_url_normalization() -> None:
    """Verify psycopg v3 URL normalization, schema context, SELECT execution, and SQL guardrails."""
    assert (
        normalize_database_url("postgres://user:pass@db.supabase.co:5432/postgres")
        == "postgresql+psycopg://user:pass@db.supabase.co:5432/postgres"
    )
    assert (
        normalize_database_url("postgresql://user:pass@localhost:5432/mydb")
        == "postgresql+psycopg://user:pass@localhost:5432/mydb"
    )

    raw_df = generate_messy_sales_dataframe(seed=42)
    clean_df, _ = clean_dataframe(raw_df)

    db = DatabaseManager.from_dataframe(clean_df, table_name="sales_data")
    try:
        assert db.dialect == "sqlite"
        assert db.get_table_names() == ["sales_data"]
        assert db.get_schema_metadata()[0]["row_count"] == 650

        schema_ctx = db.get_schema_context()
        assert "sales_data" in schema_ctx
        assert "'North'" in schema_ctx
        assert "'Electronics'" in schema_ctx

        res = db.run_safe_query(
            """```sql
            SELECT region, ROUND(SUM(revenue), 2) AS total_revenue
            FROM sales_data
            GROUP BY region
            ORDER BY total_revenue DESC;
            ```"""
        )
        assert len(res) == 4

        for unsafe_sql in [
            "DROP TABLE sales_data",
            "DELETE FROM sales_data WHERE region = 'North'",
            "UPDATE sales_data SET revenue = 0",
            "INSERT INTO sales_data VALUES ('1','2026-01-01','North','Apparel',1,0,1,10,5)",
            "ALTER TABLE sales_data ADD COLUMN hacked INT",
            "SELECT * FROM sales_data; DROP TABLE sales_data;",
        ]:
            with pytest.raises(UnsafeQueryError):
                validate_readonly_sql(unsafe_sql)
    finally:
        db.close()


def test_stats_analyzer_and_plotly_builder() -> None:
    """Verify deterministic statistical checks and Plotly chart rendering across chart types."""
    df = pd.DataFrame(
        {
            "month": ["2026-07", "2026-08", "2026-09", "2026-10"],
            "revenue": [175000.0, 180000.0, 184200.0, 124900.0],
            "units_sold": [920, 950, 980, 610],
        }
    )

    stats = run_statistical_checks(df)
    assert stats["row_count"] == 4
    assert stats["primary_metric"] == "revenue"
    assert stats["metrics"]["latest_period_delta_pct"] < -30.0
    assert "top_contributor" in stats["metrics"]

    inferred = infer_default_chart_spec(df, question="Show monthly revenue trend")
    assert inferred.chart_type == "line"

    box_inferred = infer_default_chart_spec(df, question="Give me a box plot of revenue by month")
    assert box_inferred.chart_type == "box"

    for c_type in ("bar", "line", "area", "pie", "scatter", "box", "violin", "histogram"):
        spec = ChartSpec(
            chart_type=c_type,  # type: ignore[arg-type]
            x_col="units_sold" if c_type in ("scatter", "histogram") else "month",
            y_col="revenue",
            title=f"Test {c_type}",
        )
        fig = build_plotly_chart(df, chart_spec=spec)
        assert fig is not None

    # Invalid column names in ChartSpec should gracefully fall back without crashing
    bad_spec = ChartSpec(chart_type="bar", x_col="missing_x", y_col="missing_y")
    fallback_fig = build_plotly_chart(df, chart_spec=bad_spec)
    assert fallback_fig is not None


def test_plotly_sandbox_security_and_execution() -> None:
    """Verify Option 2 AST security validation and sandboxed Plotly execution."""
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "revenue": [200.0, 150.0, 180.0, 220.0],
        }
    )

    # 1. Safe dynamic box plot code
    safe_code = """
import plotly.express as px
fig = px.box(df, x="region", y="revenue", points="outliers")
"""
    is_safe, msg = is_plotly_code_safe(safe_code)
    assert is_safe
    fig = execute_sandboxed_plotly_code(safe_code, df)
    assert fig is not None

    # 2. Block os/subprocess import
    unsafe_import = "import os; os.system('echo test')"
    is_safe, msg = is_plotly_code_safe(unsafe_import)
    assert not is_safe
    assert "Forbidden import" in msg
    assert execute_sandboxed_plotly_code(unsafe_import, df) is None

    # 3. Block open() call
    unsafe_open = "with open('secret.txt', 'w') as f: f.write('bad')"
    is_safe, msg = is_plotly_code_safe(unsafe_open)
    assert not is_safe
    assert "Forbidden identifier" in msg
    assert execute_sandboxed_plotly_code(unsafe_open, df) is None

    # 4. Block dunder sandbox escapes
    unsafe_dunder = "subclasses = ().__class__.__subclasses__()"
    is_safe, msg = is_plotly_code_safe(unsafe_dunder)
    assert not is_safe
    assert "Forbidden attribute access" in msg
    assert execute_sandboxed_plotly_code(unsafe_dunder, df) is None

    # 5. Verify skewed box plot auto-scale un-squishing
    skewed_df = pd.DataFrame(
        {
            "region": ["North", "North", "North", "North", "North"],
            "revenue": [100.0, 110.0, 120.0, 130.0, 42000.0],
        }
    )
    box_spec = ChartSpec(chart_type="box", x_col="region", y_col="revenue")
    box_fig = build_plotly_chart(skewed_df, chart_spec=box_spec)
    assert box_fig is not None
    assert box_fig.layout.yaxis.type == "log"
    assert len(box_fig.layout.updatemenus) > 0

    # 6. Verify exactly 3 models available, with gpt-oss-120b as primary
    assert SUPPORTED_GROQ_MODELS == [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
    ]
    models = fetch_available_groq_models(None)
    assert models[0] == "openai/gpt-oss-120b"
    assert len(models) == 3


def test_langgraph_self_correction_loop() -> None:
    """Verify the 4-node LangGraph workflow self-corrects broken SQL via SQL Critic."""
    assert escape_currency_dollars("Fell from $184K to $125K") == r"Fell from \$184K to \$125K"

    raw_df = generate_messy_sales_dataframe(seed=42)
    clean_df, _ = clean_dataframe(raw_df)
    db = DatabaseManager.from_dataframe(clean_df, table_name="sales_data")

    try:
        mock_llm = _MockSelfCorrectingChatModel()
        step_events: list[str] = []

        result = run_analyst_query(
            question="Show total revenue by region",
            db_manager=db,
            llm=mock_llm,
            on_step_callback=lambda node, _upd: step_events.append(node),
        )

        # Verify retry happened (sql_writer -> sql_critic -> sql_writer -> sql_critic -> stats_and_chart -> insight_analyst)
        assert result.retry_count == 1
        assert result.error is None
        assert len(result.dataframe) == 4
        assert result.figure is not None
        assert r"\$950,000" in result.executive_summary
        assert step_events == [
            "sql_writer",
            "sql_critic",
            "sql_writer",
            "sql_critic",
            "stats_and_chart",
            "insight_analyst",
        ]
    finally:
        db.close()


def test_streamlit_app_headless_smoke() -> None:
    """Run app.py headlessly via Streamlit AppTest to verify zero runtime or parameter errors."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    # Verify both tabs, the headline KPI tiles, cleaning metrics, and anomaly metrics rendered cleanly
    assert len(at.tabs) >= 2
    assert len(at.metric) >= 8


COLUMN_CLASSIFICATION_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "product_category": ["Electronics", "Furniture", "Office", "Audio"],
                "shipping_delay_days": [0, 2, 5, 1],
            }
        ),
        ["shipping_delay_days"],
        ["product_category"],
        None,
        id="int-duration-days-metric",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "region": ["North", "South", "East"],
                "product_category": ["Electronics", "Furniture", "Audio"],
                "avg_shipping_delay_days": [1.85, 2.40, 1.10],
            }
        ),
        ["avg_shipping_delay_days"],
        ["region", "product_category"],
        None,
        id="float-avg-delay-days-metric",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "department": ["Sales", "Ops", "Eng"],
                "delivery_days": [3, 4, 2],
                "tenure_years": [2, 5, 1],
                "warranty_months": [12, 24, 36],
                "grace_period": [7, 14, 30],
            }
        ),
        ["delivery_days", "tenure_years", "warranty_months", "grace_period"],
        ["department"],
        None,
        id="uploaded-csv-duration-columns",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_year": [2024, 2025, 2026],
                "month": [1, 2, 3],
                "revenue": [100.0, 200.0, 300.0],
            }
        ),
        ["revenue"],
        ["order_year", "month"],
        "order_year",
        id="integer-calendar-time-columns",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-01", "2026-02", "2026-03"],
                "shipping_delay_days": [2, 3, 1],
            }
        ),
        ["shipping_delay_days"],
        ["order_month"],
        "order_month",
        id="string-calendar-plus-delay-metric",
    ),
    pytest.param(
        pd.DataFrame({"order_year": [2026], "shipping_delay_days": [4]}),
        ["order_year", "shipping_delay_days"],
        [],
        None,
        id="single-row-boundary",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "is_delayed": [True, False, True],
                "shipping_delay_days": [2, 0, 5],
            }
        ),
        ["shipping_delay_days"],
        ["is_delayed"],
        None,
        id="bool-excluded-from-numeric",
    ),
]


@pytest.mark.parametrize(
    ("df", "expected_numeric", "expected_dims", "expected_time_col"),
    COLUMN_CLASSIFICATION_CASES,
)
def test_identify_columns_duration_vs_calendar(
    df: pd.DataFrame,
    expected_numeric: list[str],
    expected_dims: list[str],
    expected_time_col: str | None,
) -> None:
    """Verify numeric duration/delay metrics stay numeric while real calendar buckets remain time dimensions."""
    numeric_cols, dimension_cols, time_col = _identify_columns(df)
    assert numeric_cols == expected_numeric
    assert dimension_cols == expected_dims
    assert time_col == expected_time_col


COMPLEX_CHART_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "product_category": ["Electronics", "Electronics", "Furniture", "Furniture", "Audio", "Audio"],
                "shipping_delay_days": [1, 3, 2, 5, 0, 2],
            }
        ),
        "Show a violin plot with an embedded box plot displaying the distribution and spread of shipping delay days across different product categories.",
        "violin",
        "violin",
        id="violin-with-embedded-box-prompt",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "warehouse": ["East", "East", "West", "West"],
                "delivery_days": [2, 5, 1, 4],
            }
        ),
        "Plot a box plot of delivery days across warehouses.",
        "box",
        "box",
        id="boxplot-uploaded-delivery-days",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "product_category": ["Electronics", "Electronics", "Furniture", "Furniture"],
                "region": ["North", "South", "North", "South"],
                "avg_shipping_delay_days": [1.5, 2.8, 3.1, 1.9],
            }
        ),
        "Generate a 2D heatmap showing the average shipping delay days across regions on the Y-axis versus product categories on the X-axis.",
        "heatmap",
        "heatmap",
        id="2d-heatmap-avg-delay-days",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "product_category": ["Electronics", "Furniture", "Audio"],
                "avg_shipping_delay_days": [1.5, 2.8, 1.2],
            }
        ),
        "Generate a 2D heatmap showing average shipping delay days across product categories.",
        "bar",
        "bar",
        id="heatmap-single-dimension-fallback-bar",
    ),
]


@pytest.mark.parametrize(
    ("df", "question", "expected_spec_type", "expected_trace_type"),
    COMPLEX_CHART_CASES,
)
def test_complex_chart_inference_and_rendering(
    df: pd.DataFrame,
    question: str,
    expected_spec_type: str,
    expected_trace_type: str,
) -> None:
    """Verify violin (overriding embedded box phrasing) and 2D heatmap specs infer and render cleanly."""
    spec = infer_default_chart_spec(df, question=question)
    assert spec.chart_type == expected_spec_type
    fig = build_plotly_chart(df, chart_spec=spec, question=question)
    assert fig is not None
    assert len(fig.data) >= 1
    assert fig.data[0].type == expected_trace_type


CHART_EDGE_AND_ERROR_CASES = [
    pytest.param(
        pd.DataFrame(),
        None,
        "Empty table",
        None,
        id="empty-df-returns-none",
    ),
    pytest.param(
        pd.DataFrame({"region": ["North"], "shipping_delay_days": [2.5]}),
        None,
        "Single row",
        None,
        id="single-row-returns-none",
    ),
    pytest.param(
        pd.DataFrame({"region": ["North", "South"], "shipping_delay_days": [2.0, 3.0]}),
        ChartSpec(chart_type="none", title="Scalar"),
        "None spec",
        None,
        id="explicit-none-spec-returns-none",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "product_category": ["Electronics", "Electronics", "Furniture", "Furniture"],
                "region": ["North", "South", "North", "South"],
                "avg_shipping_delay_days": [1.5, 2.8, 3.1, 1.9],
            }
        ),
        ChartSpec(
            chart_type="heatmap",
            x_col="product_category",
            y_col="region",
            color_col="avg_shipping_delay_days",
            title="Heatmap Custom Orientation",
        ),
        "Heatmap explicit axes",
        "heatmap",
        id="heatmap-with-dim-as-y-col-and-metric-as-color",
    ),
    pytest.param(
        pd.DataFrame({"product_category": ["A", "B"], "shipping_delay_days": [1.0, 2.0]}),
        ChartSpec(chart_type="bar", x_col="missing_col", y_col="shipping_delay_days"),
        "Show bar chart",
        "bar",
        id="mismatched-spec-recovers-via-fallback",
    ),
]


@pytest.mark.parametrize(
    ("df", "spec", "question", "expected_trace_type"),
    CHART_EDGE_AND_ERROR_CASES,
)
def test_build_plotly_chart_edge_and_error_cases(
    df: pd.DataFrame,
    spec: ChartSpec | None,
    question: str,
    expected_trace_type: str | None,
) -> None:
    """Verify boundary inputs, alternative spec orientations, and invalid-column fallbacks."""
    fig = build_plotly_chart(df, chart_spec=spec, question=question)
    if expected_trace_type is None:
        assert fig is None
    else:
        assert fig is not None
        assert fig.data[0].type == expected_trace_type


# ---------------------------------------------------------------------------
# Regression Suites for Items 1–7
# ---------------------------------------------------------------------------

CHART_KEYWORD_BOUNDARY_AND_PRIORITY_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "stage": ["Visit", "Signup", "Trial", "Paid"],
                "order_month": ["2026-01", "2026-02", "2026-03", "2026-04"],
                "users": [1000, 420, 180, 75],
            }
        ),
        "Show a conversion funnel by stage",
        "funnel",
        "funnel",
        id="happy-funnel-beats-time-fallback",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-01", "2026-02", "2026-03", "2026-04"],
                "region": ["North", "South", "East", "West"],
                "discount_pct": [5.0, 10.0, 15.0, 20.0],
                "revenue": [1200.0, 1500.0, 1100.0, 900.0],
            }
        ),
        "Show a scatter plot of discount vs revenue over time",
        "scatter",
        "scatter",
        id="happy-scatter-beats-time-and-dim-fallbacks",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "quarter": ["Q1", "Q2", "Q3", "Q4"],
                "revenue": [12000.0, 18000.0, 15000.0, 22000.0],
            }
        ),
        "Show a donut chart of revenue by quarter",
        "pie",
        "pie",
        id="happy-donut-beats-time-fallback",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "region": ["North", "South", "East", "West"],
                "revenue": [12000.0, 18000.0, 15000.0, 22000.0],
            }
        ),
        "Show a doughnut of revenue by region",
        "pie",
        "pie",
        id="happy-doughnut-maps-to-pie",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-01", "2026-02", "2026-03"],
                "revenue": [1000.0, 1200.0, 1400.0],
            }
        ),
        "Show monthly Xbox sales",
        "line",
        "scatter",
        id="boundary-xbox-does-not-trigger-box",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "studio": ["Alpha", "Beta", "Gamma"],
                "revenue": [50000.0, 42000.0, 31000.0],
            }
        ),
        "Show box office revenue by studio",
        "bar",
        "bar",
        id="boundary-box-office-does-not-trigger-box",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "category": ["Hardware", "Accessories", "Audio"],
                "revenue": [45.0, 25.0, 60.0],
            }
        ),
        "Show revenue per piece by category",
        "bar",
        "bar",
        id="boundary-per-piece-does-not-trigger-pie",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "hotel": ["Grand", "Harbor", "Plaza"],
                "rooms": [120, 85, 95],
            }
        ),
        "Show occupied rooms by hotel",
        "bar",
        "bar",
        id="boundary-occupied-does-not-trigger-pie",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-01", "2026-02", "2026-03"],
                "revenue": [1000.0, 1300.0, 1150.0],
            }
        ),
        "Show monthly revenue in the North area",
        "line",
        "scatter",
        id="boundary-north-area-does-not-trigger-area",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-01", "2026-02", "2026-03"],
                "revenue": [1000.0, 1300.0, 1150.0],
            }
        ),
        "Show an area chart of monthly revenue",
        "area",
        "scatter",
        id="happy-area-chart-triggers-area",
    ),
]


@pytest.mark.parametrize(
    ("df", "question", "expected_spec_type", "expected_trace_type"),
    CHART_KEYWORD_BOUNDARY_AND_PRIORITY_CASES,
)
def test_chart_keyword_priority_and_word_boundaries(
    df: pd.DataFrame,
    question: str,
    expected_spec_type: str,
    expected_trace_type: str,
) -> None:
    """Verify new funnel/scatter/donut rules and word-boundary immunity against substring traps."""
    spec = infer_default_chart_spec(df, question=question)
    assert spec.chart_type == expected_spec_type
    fig = build_plotly_chart(df, chart_spec=spec, question=question)
    assert fig is not None
    assert fig.data[0].type == expected_trace_type


ZERO_DIM_TREEMAP_FUNNEL_CASES = [
    pytest.param(
        pd.DataFrame({"revenue": [100.0, 200.0, 300.0], "profit": [20.0, 45.0, 80.0]}),
        ChartSpec(chart_type="treemap", x_col="", y_col="revenue"),
        "Show treemap of revenue",
        "scatter",
        id="zero-dim-treemap-2-numerics-fallback-scatter",
    ),
    pytest.param(
        pd.DataFrame({"revenue": [100.0, 200.0, 300.0], "profit": [20.0, 45.0, 80.0]}),
        ChartSpec(chart_type="funnel", x_col="", y_col="revenue"),
        "Show funnel of revenue",
        "scatter",
        id="zero-dim-funnel-2-numerics-fallback-scatter",
    ),
    pytest.param(
        pd.DataFrame({"revenue": [100.0, 200.0, 300.0]}),
        ChartSpec(chart_type="treemap", x_col="", y_col="revenue"),
        "Show treemap of revenue",
        "histogram",
        id="zero-dim-treemap-1-numeric-fallback-histogram",
    ),
    pytest.param(
        pd.DataFrame({"revenue": [100.0, 200.0, 300.0]}),
        None,
        "Show funnel of revenue",
        "histogram",
        id="zero-dim-funnel-inferred-1-numeric-histogram",
    ),
    pytest.param(
        pd.DataFrame({"region": ["North", "South", "East"], "revenue": [100.0, 200.0, 300.0]}),
        ChartSpec(chart_type="treemap", x_col="", y_col="revenue"),
        "Show treemap by region",
        "treemap",
        id="empty-xcol-with-dim-recovers-treemap",
    ),
    pytest.param(
        pd.DataFrame({"revenue": [100.0]}),
        ChartSpec(chart_type="funnel", x_col="", y_col="revenue"),
        "Show funnel of revenue",
        None,
        id="single-row-funnel-returns-none",
    ),
]


@pytest.mark.parametrize(
    ("df", "spec", "question", "expected_trace_type"),
    ZERO_DIM_TREEMAP_FUNNEL_CASES,
)
def test_zero_dim_treemap_and_funnel_fallback(
    df: pd.DataFrame,
    spec: ChartSpec | None,
    question: str,
    expected_trace_type: str | None,
) -> None:
    """Verify 0-dimension treemap/funnel specs recover via fallback instead of dropping charts."""
    fig = build_plotly_chart(df, chart_spec=spec, question=question)
    if expected_trace_type is None:
        assert fig is None
    else:
        assert fig is not None
        assert fig.data[0].type == expected_trace_type


SCHEMA_BADGE_CASES = [
    pytest.param("candidate_name", "TEXT", "TEXT", id="candidate-name-text"),
    pytest.param("update_count", "INTEGER", "INT", id="update-count-int"),
    pytest.param("update_count", "TEXT", "TEXT", id="update-count-text"),
    pytest.param("is_validated", "BOOLEAN", "BOOL", id="is-validated-bool"),
    pytest.param("is_validated", "INTEGER", "INT", id="is-validated-int"),
    pytest.param("days_since_update", "REAL", "REAL", id="days-since-update-real"),
    pytest.param("days_since_update", "DOUBLE PRECISION", "REAL", id="days-since-update-double"),
    pytest.param("order_date", "TEXT", "DATE", id="order-date-text"),
    pytest.param("orderDate", "VARCHAR(32)", "DATE", id="camel-order-date-varchar"),
    pytest.param("created_at", "TIMESTAMP WITH TIME ZONE", "DATE", id="created-at-timestamptz"),
    pytest.param("start_date", "BIGINT", "INT", id="explicit-int-takes-precedence"),
]


@pytest.mark.parametrize(("col_name", "raw_type", "expected_badge"), SCHEMA_BADGE_CASES)
def test_schema_badge_type_no_date_substring_overmatch(
    col_name: str,
    raw_type: str,
    expected_badge: str,
) -> None:
    """Verify SQL types and word-boundary date checks prevent false DATE badges."""
    assert _badge_type(col_name, raw_type) == expected_badge


def test_schema_context_respects_non_date_columns_with_date_substring() -> None:
    """Verify get_schema_context avoids running MIN/MAX date bounds on candidate/update columns."""
    df = pd.DataFrame(
        {
            "candidate_name": ["Alice", "Bob", "Carol"],
            "update_count": [1, 4, 2],
            "is_validated": [True, False, True],
            "days_since_update": [3.5, 1.0, 7.2],
            "order_date": ["2026-01-05", "2026-02-10", "2026-03-15"],
        }
    )
    db = DatabaseManager.from_dataframe(df, table_name="candidates")
    try:
        meta = {c["name"]: c["badge"] for c in db.get_schema_metadata()[0]["columns"]}
        assert meta["candidate_name"] == "TEXT"
        assert meta["update_count"] == "INT"
        assert meta["days_since_update"] == "REAL"
        assert meta["order_date"] == "DATE"

        ctx = db.get_schema_context()
        assert "candidate_name (TEXT / TEXT) [distinct values: 'Alice', 'Bob', 'Carol']" in ctx
        assert "order_date (TEXT / DATE) [date range: '2026-01-05' to '2026-03-15']" in ctx
        assert "update_count (BIGINT / INT) [date range" not in ctx
    finally:
        db.close()


UNIT_CLASSIFICATION_AND_FORMATTING_CASES = [
    pytest.param("profit_margin_pct", 24.5, "percent", ("", "%"), "24.5%", id="profit-margin-pct"),
    pytest.param("revenue_share", 38.24, "percent", ("", "%"), "38.2%", id="revenue-share"),
    pytest.param("items_rated", 120.0, "none", ("", ""), "120", id="items-rated-not-pct"),
    pytest.param("corporate_spend", 2500.0, "currency", ("$", ""), "$2.5K", id="corporate-spend-not-pct"),
    pytest.param("sales_count", 450.0, "none", ("", ""), "450", id="sales-count-not-currency"),
    pytest.param("total_sales_orders", 12500.0, "none", ("", ""), "12.5K", id="sales-orders-not-currency"),
    pytest.param("revenue", -12500.0, "currency", ("$", ""), "-$12.5K", id="negative-currency-sign-order"),
    pytest.param("revenue_per_order", 85.5, "currency", ("$", ""), "$85.5", id="revenue-per-order"),
    pytest.param("discount_amount", 50.0, "currency", ("$", ""), "$50", id="discount-amount"),
]


@pytest.mark.parametrize(
    ("col_name", "value", "expected_unit", "expected_affixes", "expected_formatted"),
    UNIT_CLASSIFICATION_AND_FORMATTING_CASES,
)
def test_metric_unit_classification_and_formatting(
    col_name: str,
    value: float,
    expected_unit: str,
    expected_affixes: tuple[str, str],
    expected_formatted: str,
) -> None:
    """Verify mutually exclusive currency vs percentage units and unified formatting."""
    assert classify_metric_unit(col_name) == expected_unit
    assert get_metric_unit_affixes(col_name) == expected_affixes
    assert format_metric_value(value, col_name) == expected_formatted


def test_chart_hovertemplate_prevents_dollar_percent_collision() -> None:
    """Verify profit_margin_pct renders '%' without '$' and sales_count renders neither."""
    df = pd.DataFrame(
        {
            "region": ["North", "South", "East"],
            "profit_margin_pct": [24.5, 31.2, 18.0],
            "sales_count": [120, 150, 95],
        }
    )
    margin_fig = build_plotly_chart(
        df,
        ChartSpec(chart_type="bar", x_col="region", y_col="profit_margin_pct"),
    )
    assert margin_fig is not None
    margin_hover = margin_fig.data[0].hovertemplate
    assert "$" not in margin_hover
    assert "}%" in margin_hover
    assert list(margin_fig.data[0].text) == ["24.5%", "31.2%", "18.0%"]

    count_fig = build_plotly_chart(
        df,
        ChartSpec(chart_type="bar", x_col="region", y_col="sales_count"),
    )
    assert count_fig is not None
    count_hover = count_fig.data[0].hovertemplate
    assert "$" not in count_hover
    assert "}%" not in count_hover
    assert list(count_fig.data[0].text) == ["120", "150", "95"]
    assert _fmt_number(24.5, pct=True) == "24.5%"


ID_AND_BOOL_EXCLUSION_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "customer_id": [101, 102, 103, 104],
                "store_code": [10, 20, 10, 30],
                "zip_code": [90210, 10001, 60601, 94105],
                "postal_code": [11111, 22222, 33333, 44444],
                "is_active": [True, False, True, True],
                "is_vip": pd.Series([True, False, pd.NA, True], dtype="boolean"),
                "order_date": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
                "revenue": [500.0, 250.0, 800.0, 620.0],
                "units_sold": [5, 2, 8, 6],
            }
        ),
        ["revenue", "units_sold"],
        ["id", "customer_id", "store_code", "zip_code", "postal_code", "is_active", "is_vip", "order_date"],
        "revenue",
        (2, 2),
        id="ids-codes-and-bools-excluded-everywhere",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "customer_id": [101, 102, 103],
                "zip_code": [90210, 10001, 60601],
                "is_active": [True, False, True],
            }
        ),
        [],
        ["customer_id", "zip_code", "is_active"],
        None,
        (0, 0),
        id="only-ids-and-bools-yields-no-quantitative-metrics",
    ),
]


@pytest.mark.parametrize(
    ("df", "expected_numeric", "expected_dims", "expected_primary", "expected_corr_shape"),
    ID_AND_BOOL_EXCLUSION_CASES,
)
def test_numeric_ids_codes_and_bools_excluded_across_modules(
    df: pd.DataFrame,
    expected_numeric: list[str],
    expected_dims: list[str],
    expected_primary: str | None,
    expected_corr_shape: tuple[int, int],
) -> None:
    """Verify IDs, codes, zip/postal codes, and booleans stay out of quantitative metrics."""
    num_cols, dim_cols, _ = _identify_columns(df)
    assert num_cols == expected_numeric
    assert dim_cols == expected_dims
    assert get_quantitative_numeric_columns(df) == expected_numeric

    overview = compute_dataset_overview(df)
    assert overview["numeric_columns"] == expected_numeric

    corr = compute_correlation_matrix(df)
    assert corr.shape == expected_corr_shape

    stats = run_statistical_checks(df)
    assert stats["primary_metric"] == expected_primary

    profiles = compute_column_profiles(df).set_index("column")
    if "customer_id" in profiles.index:
        assert profiles.loc["customer_id", "type"] == "ID"
    if "is_active" in profiles.index:
        assert profiles.loc["is_active", "type"] == "BOOL"

    _, default_metric = find_default_timeseries_columns(df)
    assert default_metric == expected_primary

    kpis = compute_headline_kpis(df)
    if expected_numeric:
        kpi_labels = " ".join(k.label for k in kpis)
        assert "Customer Id" not in kpi_labels
        assert "Zip Code" not in kpi_labels
        assert "Is Active" not in kpi_labels
    else:
        numeric_tile = next(k for k in kpis if k.label == "Numeric Columns")
        assert numeric_tile.value == "0"


def test_detect_timeseries_anomalies_rejects_boolean_metric() -> None:
    """Verify passing a boolean metric directly to anomaly detection fails cleanly."""
    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "is_active": [True, False, True],
        }
    )
    with pytest.raises(ValueError, match="Boolean"):
        detect_timeseries_anomalies(df, date_col="order_date", metric_col="is_active")


NON_ADDITIVE_STATS_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-09", "2026-09", "2026-10", "2026-10"],
                "region": ["North", "South", "North", "South"],
                "profit_margin_pct": [20.0, 30.0, 10.0, 20.0],
            }
        ),
        "profit_margin_pct",
        True,
        -40.0,
        id="non-additive-margin-uses-mean-no-total-or-share",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-09", "2026-09", "2026-10", "2026-10"],
                "region": ["North", "South", "North", "South"],
                "avg_shipping_delay_days": [2.0, 4.0, 6.0, 6.0],
            }
        ),
        "avg_shipping_delay_days",
        True,
        100.0,
        id="non-additive-delay-uses-mean-period-delta",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_month": ["2026-09", "2026-09", "2026-10", "2026-10"],
                "region": ["North", "South", "North", "South"],
                "items_rated": [10, 20, 30, 30],
            }
        ),
        "items_rated",
        False,
        100.0,
        id="additive-items-rated-keeps-total-and-sum",
    ),
]


@pytest.mark.parametrize(
    ("df", "metric_col", "expect_non_additive", "expected_delta_pct"),
    NON_ADDITIVE_STATS_CASES,
)
def test_non_additive_primary_metric_statistical_checks(
    df: pd.DataFrame,
    metric_col: str,
    expect_non_additive: bool,
    expected_delta_pct: float,
) -> None:
    """Verify run_statistical_checks never sums or computes % of total for non-additive metrics."""
    assert is_non_additive_metric(metric_col) is expect_non_additive
    stats = run_statistical_checks(df)
    assert stats["primary_metric"] == metric_col
    assert stats["metrics"]["is_non_additive"] is expect_non_additive
    assert stats["metrics"]["latest_period_delta_pct"] == pytest.approx(expected_delta_pct, abs=0.01)

    if expect_non_additive:
        assert "total" not in stats["metrics"]
        assert "top_contributor" not in stats["metrics"]
        assert "Total:" not in stats["summary_text"]
        assert "% of total" not in stats["summary_text"]
        assert "Mean:" in stats["summary_text"]
    else:
        assert "total" in stats["metrics"]
        assert "Total:" in stats["summary_text"]
        assert "% of total" in stats["summary_text"]


def test_headline_kpis_and_anomalies_use_mean_for_non_additive_metrics() -> None:
    """Verify compute_headline_kpis and detect_timeseries_anomalies average non-additive metrics."""
    dates = ["2026-05-01", "2026-05-01", "2026-06-01", "2026-06-01"]
    df = pd.DataFrame(
        {
            "order_date": dates,
            "revenue": [1000.0, 2000.0, 3000.0, 3000.0],
            "profit_margin_pct": [20.0, 40.0, 10.0, 30.0],
            "shipping_delay_days": [2.0, 6.0, 1.0, 3.0],
        }
    )

    kpis = compute_headline_kpis(df, max_metrics=3)
    by_prefix = {k.label.split(" (")[0]: k for k in kpis}
    assert by_prefix["Revenue"].value == "$6,000"
    assert by_prefix["Profit Margin Pct"].value == "20.0%"
    assert by_prefix["Profit Margin Pct"].chart == [30.0, 20.0]
    assert "Monthly average" in (by_prefix["Profit Margin Pct"].help or "")
    assert by_prefix["Shipping Delay Days"].value == "2"
    assert by_prefix["Shipping Delay Days"].chart == [4.0, 2.0]

    ts_margin, _ = detect_timeseries_anomalies(df, date_col="order_date", metric_col="profit_margin_pct")
    assert list(ts_margin["profit_margin_pct"]) == [30.0, 20.0]

    ts_rev, _ = detect_timeseries_anomalies(df, date_col="order_date", metric_col="revenue")
    assert list(ts_rev["revenue"]) == [3000.0, 6000.0]


DATE_CLEANING_CASES = [
    pytest.param(
        pd.DataFrame({"pickup_time": ["09:15", "14:30", "18:45"], "val": [1, 2, 3]}),
        "pickup_time",
        ["09:15", "14:30", "18:45"],
        id="time-only-strings-preserved",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "created_at": ["2026-05-01 09:15:00", "2026-05-01 14:30:00", "2026-05-02 08:00:00"],
                "val": [1, 1, 2],
            }
        ),
        "created_at",
        ["2026-05-01 09:15:00", "2026-05-01 14:30:00", "2026-05-02 08:00:00"],
        id="subday-timestamps-preserved",
    ),
    pytest.param(
        pd.DataFrame({"billing_month": ["2026-01", "2026-02", "2026-03"], "val": [10, 20, 30]}),
        "billing_month",
        ["2026-01", "2026-02", "2026-03"],
        id="yyyy-mm-month-buckets-preserved",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_date": ["01/15/2026", "02/20/2026", "03/05/2026", "04/10/2026"],
                "val": [1, 2, 3, 4],
            }
        ),
        "order_date",
        ["2026-01-15", "2026-02-20", "2026-03-05", "2026-04-10"],
        id="us-mm-dd-yyyy-inferred-dayfirst-false",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "order_date": ["15/01/2026", "20/02/2026", "05/03/2026", "10/04/2026"],
                "val": [1, 2, 3, 4],
            }
        ),
        "order_date",
        ["2026-01-15", "2026-02-20", "2026-03-05", "2026-04-10"],
        id="eu-dd-mm-yyyy-inferred-dayfirst-true",
    ),
]


@pytest.mark.parametrize(("raw_df", "col_name", "expected_values"), DATE_CLEANING_CASES)
def test_data_cleaning_date_heuristics(
    raw_df: pd.DataFrame,
    col_name: str,
    expected_values: list[str],
) -> None:
    """Verify date cleaning avoids time/month corruption, keeps sub-day times, and infers dayfirst."""
    cleaned, _ = clean_dataframe(raw_df)
    assert list(cleaned[col_name]) == expected_values


DIRTY_NUMERIC_AND_MISSING_CASES = [
    pytest.param(
        pd.DataFrame(
            {
                "profit": ["($120.00)", "$250.50", "( $1,000.00 )", "-$45.00"],
                "discount_pct": ["(12.5%)", "15.0%", "0%", "5%"],
            }
        ),
        {
            "profit": [-120.0, 250.5, -1000.0, -45.0],
            "discount_pct": [-12.5, 15.0, 0.0, 5.0],
        },
        id="accounting-negatives-currency-and-pct",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "region": ["north", " N/A ", " NULL ", "None", "NaN"],
                "amount": ["$100", "$200", "$300", "$400", "$500"],
            }
        ),
        {
            "region": ["North", "Unknown", "Unknown", "Unknown", "Unknown"],
        },
        id="case-insensitive-missing-tokens",
    ),
]


@pytest.mark.parametrize(("raw_df", "expected_cols"), DIRTY_NUMERIC_AND_MISSING_CASES)
def test_data_cleaning_accounting_negatives_and_missing_tokens(
    raw_df: pd.DataFrame,
    expected_cols: dict[str, list[Any]],
) -> None:
    """Verify accounting negatives ($120.00) and case-insensitive missing tokens clean accurately."""
    cleaned, _ = clean_dataframe(raw_df)
    for col_name, expected_vals in expected_cols.items():
        assert list(cleaned[col_name]) == expected_vals


def test_data_cleaning_nullable_int_bool_and_numeric_id_imputation() -> None:
    """Verify clean_dataframe handles nullable boolean/Int64 without TypeError and skips ID median."""
    raw_df = pd.DataFrame(
        {
            "customer_id": [101.0, 104.0, np.nan, 110.0],
            "zip_code": pd.Series([90210, 10001, pd.NA, 60601], dtype="Int64"),
            "is_active": pd.Series([True, False, pd.NA, True], dtype="boolean"),
            "units_sold": pd.Series([10, 20, pd.NA, 40], dtype="Int64"),
        }
    )
    cleaned, _ = clean_dataframe(raw_df)

    # Boolean nullable column filled with modal boolean without TypeError
    assert pd.api.types.is_bool_dtype(cleaned["is_active"])
    assert int(cleaned["is_active"].isna().sum()) == 0
    assert bool(cleaned["is_active"].iloc[2]) is True

    # Nullable Int64 metric filled with integer median (20), preserving integer dtype
    assert pd.api.types.is_integer_dtype(cleaned["units_sold"])
    assert int(cleaned["units_sold"].iloc[2]) == 20

    # Numeric ID/zip columns preserve NULL rather than fabricating fractional median IDs (104.0 / 102.5)
    assert pd.isna(cleaned["customer_id"].iloc[2])
    assert pd.isna(cleaned["zip_code"].iloc[2])


SANDBOX_SAFETY_CASES = [
    pytest.param(
        "import statsmodels.api as sm\nfig = px.scatter(df, x='a', y='b')",
        False,
        "Forbidden import",
        id="reject-statsmodels-import",
    ),
    pytest.param(
        "from statsmodels.nonparametric.smoothers_lowess import lowess\nfig = px.scatter(df, x='a', y='b')",
        False,
        "Forbidden import",
        id="reject-statsmodels-from-import",
    ),
    pytest.param(
        "fig = px.scatter(df, x='a', y='b', trendline='ols')",
        False,
        "statsmodels",
        id="reject-px-scatter-trendline-ols",
    ),
    pytest.param(
        "fig = px.scatter(df, x='a', y='b')",
        True,
        "Safe",
        id="allow-safe-px-scatter",
    ),
]


@pytest.mark.parametrize(("code", "expected_safe", "expected_msg_part"), SANDBOX_SAFETY_CASES)
def test_sandbox_rejects_uninstalled_statsmodels_dependencies(
    code: str,
    expected_safe: bool,
    expected_msg_part: str,
) -> None:
    """Verify AST sandbox check aligns with installed packages and rejects statsmodels usage."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    is_safe, msg = is_plotly_code_safe(code)
    assert is_safe is expected_safe
    assert expected_msg_part in msg
    fig = execute_sandboxed_plotly_code(code, df)
    if expected_safe:
        assert fig is not None
    else:
        assert fig is None


SQL_READONLY_VALIDATION_CASES = [
    pytest.param(
        "SELECT REPLACE(region, ' ', '_') AS clean_region FROM sales_data",
        True,
        id="allow-select-replace-function",
    ),
    pytest.param(
        "WITH cte AS (SELECT REPLACE(note, '--', '-') AS clean_note FROM logs) SELECT * FROM cte",
        True,
        id="allow-cte-replace-with-dashes-in-literal",
    ),
    pytest.param(
        'SELECT "call", "update", "create", "delete", "replace", "copy" FROM audit_log',
        True,
        id="allow-double-quoted-keyword-column-names",
    ),
    pytest.param(
        'SELECT * FROM "update" WHERE "call" = \'DROP TABLE sales_data\'',
        True,
        id="allow-double-quoted-table-and-single-quoted-keyword",
    ),
    pytest.param(
        "REPLACE INTO sales_data VALUES ('1', 'North')",
        False,
        id="block-replace-into-dml",
    ),
    pytest.param(
        "INSERT OR REPLACE INTO sales_data VALUES ('1', 'North')",
        False,
        id="block-insert-or-replace",
    ),
    pytest.param(
        "CREATE OR REPLACE VIEW v AS SELECT 1",
        False,
        id="block-create-or-replace",
    ),
    pytest.param(
        "CALL cleanup_orders()",
        False,
        id="block-call-statement",
    ),
    pytest.param(
        "WITH cte AS (SELECT 1) CALL cleanup_orders()",
        False,
        id="block-cte-followed-by-call",
    ),
    pytest.param(
        'DELETE FROM "call"',
        False,
        id="block-delete-targeting-quoted-column-table",
    ),
    pytest.param(
        'SELECT "call" FROM logs; DROP TABLE logs',
        False,
        id="block-stacked-statement-after-quoted-col",
    ),
]


@pytest.mark.parametrize(("sql", "should_pass"), SQL_READONLY_VALIDATION_CASES)
def test_validate_readonly_sql_benign_vs_unsafe(sql: str, should_pass: bool) -> None:
    """Verify SELECT REPLACE(...) and double-quoted identifiers pass while DML/stacked SQL fail."""
    if should_pass:
        cleaned = validate_readonly_sql(sql)
        assert cleaned
    else:
        with pytest.raises(UnsafeQueryError):
            validate_readonly_sql(sql)


def test_database_manager_executes_replace_and_quoted_keyword_columns() -> None:
    """Verify SQLite end-to-end execution works for SELECT REPLACE(...) and quoted identifier columns."""
    df = pd.DataFrame(
        {
            "call": ["inbound", "outbound"],
            "update": ["2026-01-01", "2026-01-02"],
            "note": ["foo -- bar", "baz -- qux"],
        }
    )
    db = DatabaseManager.from_dataframe(df, table_name="audit_log")
    try:
        res = db.run_safe_query(
            'SELECT "call", "update", REPLACE(note, \'--\', \'-\') AS clean_note FROM audit_log ORDER BY "update"'
        )
        assert list(res["call"]) == ["inbound", "outbound"]
        assert list(res["update"]) == ["2026-01-01", "2026-01-02"]
        assert list(res["clean_note"]) == ["foo - bar", "baz - qux"]
    finally:
        db.close()


@pytest.mark.parametrize(
    ("question", "expected_chart_type"),
    [
        ("Show box plots of revenue by region", "box"),
        ("Plot boxplots for revenue by region", "box"),
        ("Draw a box-plot of revenue", "box"),
        ("Display heatmaps of metrics", "heatmap"),
        ("Create treemaps of revenue by region", "treemap"),
        ("Show area charts over time", "area"),
        ("Plot violin plots of revenue", "violin"),
        ("Show scatter plots of orders vs revenue", "scatter"),
        ("Display funnel charts by stage", "funnel"),
        ("Show donut charts by region", "pie"),
        ("Show North area revenue by region", "line"),
        ("Compare Xbox revenue by region", "line"),
        ("Check per piece revenue by region", "line"),
    ],
)
def test_plural_and_hyphenated_chart_keyword_inference(
    question: str,
    expected_chart_type: str,
) -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "region": ["East", "West", "East", "West"],
            "revenue": [100.0, 200.0, 150.0, 250.0],
            "orders": [10, 20, 15, 25],
        }
    )
    spec = infer_default_chart_spec(df, question=question)
    assert spec.chart_type == expected_chart_type


def test_build_plotly_chart_fallback_to_heatmap_populates_heat_cols() -> None:
    df = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West"],
            "channel": ["Web", "Mobile", "Web", "Mobile"],
            "revenue": [100.0, 200.0, 150.0, 250.0],
        }
    )
    invalid_spec = ChartSpec(chart_type="bar", x_col="missing_col", y_col="revenue")
    fig = build_plotly_chart(
        df,
        chart_spec=invalid_spec,
        question="Show heatmap of revenue by region and channel",
    )
    assert fig is not None
    assert len(fig.data) > 0
    assert fig.data[0].type == "heatmap"


@pytest.mark.parametrize(
    "treemap_df",
    [
        pd.DataFrame({"region": [None, None], "revenue": [100.0, 200.0]}),
        pd.DataFrame({"region": ["East", "West"], "revenue": [0.0, -50.0]}),
    ],
)
def test_build_plotly_chart_empty_or_nonpositive_treemap_returns_none(
    treemap_df: pd.DataFrame,
) -> None:
    spec = ChartSpec(chart_type="treemap", x_col="region", y_col="revenue")
    fig = build_plotly_chart(treemap_df, chart_spec=spec)
    assert fig is None


@pytest.mark.parametrize(
    ("df_data", "expected_color_col"),
    [
        (
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "region": ["East", "West", "East", "West"],
                "orders": [5, 10, 8, 12],
                "revenue": [100.0, 200.0, 150.0, 250.0],
            },
            "region",
        ),
        (
            {
                "customer_id": ["C1", "C2", "C3", "C4"],
                "orders": [5, 10, 8, 12],
                "revenue": [100.0, 200.0, 150.0, 250.0],
            },
            None,
        ),
    ],
)
def test_scatter_color_col_excludes_id_columns(
    df_data: dict[str, list[Any]],
    expected_color_col: str | None,
) -> None:
    df = pd.DataFrame(df_data)
    spec = infer_default_chart_spec(df, question="scatter plot of orders vs revenue")
    assert spec.chart_type == "scatter"
    assert spec.color_col == expected_color_col


@pytest.mark.parametrize(
    ("col_name", "expected_unit", "expected_non_additive"),
    [
        ("total_clicks_spend", "currency", False),
        ("customers_revenue", "currency", False),
        ("transactions_amount", "currency", False),
        ("margin_amount", "currency", False),
        ("sales_dollars", "currency", False),
        ("unit_cost", "currency", True),
        ("cac", "currency", True),
        ("high_margin_orders", "none", False),
        ("discount_pct_count", "none", False),
        ("delay_count", "none", False),
        ("order_count_pct", "percent", True),
    ],
)
def test_unit_and_additivity_precedence(
    col_name: str,
    expected_unit: str,
    expected_non_additive: bool,
) -> None:
    assert classify_metric_unit(col_name) == expected_unit
    assert is_non_additive_metric(col_name) is expected_non_additive


@pytest.mark.parametrize(
    ("col_name", "expected_is_id"),
    [
        ("product_sku", True),
        ("item_upc", True),
        ("book_isbn", True),
        ("customer_key", True),
        ("revenue", False),
        ("total_clicks_spend", False),
    ],
)
def test_id_or_code_suffix_extensions(col_name: str, expected_is_id: bool) -> None:
    assert is_id_or_code_column(col_name) is expected_is_id


@pytest.mark.parametrize(
    ("col_name", "values"),
    [
        ("is_active", [True, False, True, False]),
        ("has_churned", [1, 0, 1, 0]),
        ("verified_flag", [0, 1, 0, 1]),
    ],
)
def test_sqlite_roundtripped_binary_boolean_indicator_excluded_everywhere(
    col_name: str,
    values: list[Any],
) -> None:
    from src.eda_engine import compute_column_profiles, detect_timeseries_anomalies

    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "region": ["North", "North", "South", "South"],
            col_name: values,
            "revenue": [100.0, 200.0, 150.0, 250.0],
        }
    )
    db = DatabaseManager.from_dataframe(df, table_name="flags_tbl")
    try:
        roundtripped = db.run_safe_query("SELECT * FROM flags_tbl")
    finally:
        db.close()

    assert is_binary_bool_indicator(col_name, roundtripped[col_name]) is True
    assert col_name not in get_quantitative_numeric_columns(roundtripped)

    numeric_cols, dimension_cols, time_col = _identify_columns(roundtripped)
    assert time_col == "order_date"
    assert col_name not in numeric_cols
    assert col_name in dimension_cols

    profiles = compute_column_profiles(roundtripped)
    row = profiles.loc[profiles["column"] == col_name]
    assert len(row) == 1
    assert row.iloc[0]["type"] == "BOOL"

    with pytest.raises(ValueError, match="Boolean column"):
        detect_timeseries_anomalies(roundtripped, date_col="order_date", metric_col=col_name)


@pytest.mark.parametrize(
    ("raw_dates", "expected_dates"),
    [
        (
            ["2026-05-01T09:15:00Z", "2026-05-02T14:30:00Z"],
            ["2026-05-01 09:15:00", "2026-05-02 14:30:00"],
        ),
        (
            ["2026-05-01T09:15:00Z", "2026-05-02T14:30:00+02:00"],
            ["2026-05-01 09:15:00", "2026-05-02 12:30:00"],
        ),
        (
            pd.to_datetime(["2026-05-01T09:15:00+02:00", "2026-05-02T00:00:00+02:00"]),
            ["2026-05-01 07:15:00", "2026-05-01 22:00:00"],
        ),
        (
            ["15.01.2026", "20.02.2026"],
            ["2026-01-15", "2026-02-20"],
        ),
        (
            ["05/01/2026 09:15:00.123", "05/02/2026 14:30:00.456"],
            ["2026-05-01 09:15:00", "2026-05-02 14:30:00"],
        ),
    ],
)
def test_clean_dataframe_handles_timezones_dot_dates_and_fractional_seconds(
    raw_dates: Any,
    expected_dates: list[str],
) -> None:
    from src.data_cleaning import clean_dataframe

    raw_df = pd.DataFrame(
        {
            "event_ts": raw_dates,
            "revenue": [100.0, 200.0],
        }
    )
    cleaned_df, _ = clean_dataframe(raw_df)
    assert list(cleaned_df["event_ts"]) == expected_dates


def test_clean_dataframe_binary_indicator_uses_modal_fill_not_mean_or_median() -> None:
    from src.data_cleaning import clean_dataframe

    raw_df = pd.DataFrame(
        {
            "is_active": [1.0, 0.0, None, 1.0],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        }
    )
    cleaned_df, _ = clean_dataframe(raw_df)
    assert cleaned_df["is_active"].isna().sum() == 0
    assert set(cleaned_df["is_active"].unique()) <= {0.0, 1.0}
    assert cleaned_df.loc[2, "is_active"] == 1.0


@pytest.mark.parametrize(
    ("sql", "expected_valid"),
    [
        ("SELECT\n  REPLACE\n  (name, 'a', 'b') AS cleaned\nFROM orders", True),
        ("SELECT id FROM orders /* comment ; DROP TABLE orders */ WHERE status = 'ok'", True),
        ("SELECT id FROM orders WHERE note = 'it''s ; DROP TABLE orders -- safe'", True),
        ('SELECT "call", "update", "replace" FROM audit_log', True),
        ("REPLACE INTO orders (id) VALUES (1)", False),
        ("INSERT OR REPLACE INTO orders (id) VALUES (1)", False),
        ("CALL my_proc('x')", False),
        ("SELECT 1; DROP TABLE orders", False),
        ("SELECT * FROM orders WHERE note = 'x' -- ok\n; DELETE FROM orders", False),
    ],
)
def test_validate_readonly_sql_additional_edge_cases(sql: str, expected_valid: bool) -> None:
    from src.db_manager import UnsafeQueryError, validate_readonly_sql

    if expected_valid:
        cleaned = validate_readonly_sql(sql)
        assert isinstance(cleaned, str) and len(cleaned) > 0
    else:
        with pytest.raises(UnsafeQueryError):
            validate_readonly_sql(sql)


@pytest.mark.parametrize(
    ("question", "expected_chart_type"),
    [
        # Negated explicit chart phrases must not override default or positive intent
        (
            "Show average shipping delay days by month across regions comparing trends for North and South as a line chart, avoiding area-chart styling even though it covers regional sales areas.",
            "line",
        ),
        ("Show revenue over time as a line chart, not an area chart", "line"),
        ("Show revenue trends without stacked area styling", "line"),
        ("Do not use a box plot; show a scatter plot of orders vs revenue", "scatter"),
        ("Rather than showing a violin plot, draw a box plot of revenue by region", "box"),
        ("Instead of a funnel chart, show a donut chart of revenue by region", "pie"),
        ("Don't show an embedded box plot or pie chart; plot orders vs revenue over time", "line"),
        ("Show non-area-chart monthly trends for revenue", "line"),
        # Negation first, then explicit positive mention of the same type
        ("Avoid a broken basic area chart; show a clean stacked area chart over time", "area"),
        # Negation elsewhere in prompt (not before chart keyword) must remain positive
        ("Show revenue without discounts as an area chart over time", "area"),
        ("Show piano sales as a pie chart by region", "pie"),
        ("Show revenue over time as an area chart, avoiding scatter plots", "area"),
    ],
)
def test_negated_chart_keyword_inference(
    question: str,
    expected_chart_type: str,
) -> None:
    df = pd.DataFrame(
        {
            "order_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "region": ["East", "West", "North", "South"],
            "revenue": [100.0, 200.0, 150.0, 250.0],
            "orders": [10, 20, 15, 25],
        }
    )
    spec = infer_default_chart_spec(df, question=question)
    assert spec.chart_type == expected_chart_type
