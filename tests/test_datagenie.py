"""Comprehensive automated verification suite for DataGenie (Phases 1–5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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
)
from src.stats_analyzer import (
    ChartSpec,
    _identify_columns,
    build_plotly_chart,
    execute_sandboxed_plotly_code,
    infer_default_chart_spec,
    is_plotly_code_safe,
    run_statistical_checks,
)


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
