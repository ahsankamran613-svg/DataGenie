"""4-Node LangGraph v1 Multi-Agent Workflow for DataGenie.

Graph Topology:
    START -> sql_writer -> sql_critic --(valid SQL + rows)--> stats_and_chart -> insight_analyst -> END
                  ^            |
                  +-- (retry <= 3 on SQL error / 0 rows) --+
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Callable, Literal, TypedDict
import pandas as pd
import plotly.graph_objects as go
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.db_manager import DatabaseManager, validate_readonly_sql
from src.stats_analyzer import (
    ChartSpec,
    build_plotly_chart,
    infer_default_chart_spec,
    run_statistical_checks,
)


SUPPORTED_GROQ_MODELS: list[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
]

MAX_SQL_RETRIES = 3


def fetch_available_groq_models(api_key: str | None = None) -> list[str]:
    """Fetch live chat/text models available to this Groq API key, falling back to SUPPORTED_GROQ_MODELS."""
    if not api_key or not api_key.strip():
        return list(SUPPORTED_GROQ_MODELS)
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "User-Agent": "DataGenie/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_ids = [m["id"] for m in data.get("data", []) if "id" in m]
            excluded_terms = ("whisper", "guard", "orpheus", "safeguard")
            valid_models = [
                m_id for m_id in raw_ids
                if not any(term in m_id.lower() for term in excluded_terms)
            ]
            if valid_models:
                priority = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
                ordered = [m for m in priority if m in valid_models]
                others = [m for m in sorted(valid_models) if m not in ordered]
                return ordered + others
    except Exception:
        pass
    return list(SUPPORTED_GROQ_MODELS)


class SQLDraft(BaseModel):
    """Structured output schema for Node 1 (`sql_writer_node`)."""

    sql_query: str = Field(
        description="Single read-only SELECT (or WITH ... SELECT) SQL query matching the target dialect."
    )
    explanation: str = Field(
        default="",
        description="Brief 1-sentence explanation of how this query answers the business question.",
    )


class AgentState(TypedDict):
    """Typed state passed through the 4-node LangGraph workflow."""

    question: str
    dialect: str
    schema_context: str
    sql_query: str
    sql_explanation: str
    sql_error: str | None
    retry_count: int
    result_records: list[dict[str, Any]]
    result_columns: list[str]
    stats_summary: dict[str, Any]
    chart_spec: dict[str, Any] | None
    executive_summary: str
    trace_steps: list[dict[str, Any]]


@dataclass
class AnalystResult:
    """Clean container returned to the Streamlit UI after running the LangGraph pipeline."""

    question: str
    executive_summary: str
    sql_query: str
    sql_explanation: str
    dataframe: pd.DataFrame
    figure: go.Figure | None
    stats_summary: dict[str, Any]
    chart_spec: dict[str, Any] | None
    retry_count: int
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def escape_currency_dollars(markdown_text: str) -> str:
    """Escape `$` immediately preceding digits so Streamlit does not misinterpret prices as LaTeX."""
    return re.sub(r"(?<!\\)\$(?=\d)", r"\\$", markdown_text)


def _extract_sql_fallback(raw_content: str) -> SQLDraft:
    """Fallback parser if an open-weight model returns JSON or fenced SQL text directly."""
    text = raw_content.strip()

    # 1. Try JSON object if present
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict) and "sql_query" in payload:
                return SQLDraft(
                    sql_query=str(payload["sql_query"]),
                    explanation=str(payload.get("explanation", "")),
                )
        except Exception:
            pass

    # 2. Try ```sql ... ``` fenced block
    fence_match = re.search(r"```(?:sql|postgresql|sqlite)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return SQLDraft(sql_query=fence_match.group(1).strip(), explanation="Extracted from SQL block.")

    # 3. Assume raw SQL starting at SELECT or WITH
    sel_match = re.search(r"\b(WITH|SELECT)\b.*", text, flags=re.DOTALL | re.IGNORECASE)
    if sel_match:
        return SQLDraft(sql_query=sel_match.group(0).strip(), explanation="Direct SQL output.")

    return SQLDraft(sql_query=text, explanation="")


def create_groq_llm(
    api_key: str,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
) -> ChatGroq:
    """Instantiate a configured `ChatGroq` client."""
    if not api_key or not api_key.strip():
        raise ValueError("A valid Groq API Key is required to run the AI Business Analyst.")
    return ChatGroq(
        api_key=api_key.strip(),
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )


def build_analyst_graph(
    db_manager: DatabaseManager,
    llm: BaseChatModel,
):
    """Build and compile the 4-node LangGraph v1 StateGraph."""

    # ------------------------------------------------------------------
    # Node 1: SQL Writer Agent
    # ------------------------------------------------------------------
    def sql_writer_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        dialect = state["dialect"]
        retry_num = state.get("retry_count", 0)
        prev_error = state.get("sql_error")

        dialect_hint = (
            "Use SQLite date string functions such as ` strftime('%Y-%m', order_date) ` or ISO string comparisons "
            "(since dates are stored as 'YYYY-MM-DD' strings)."
            if dialect == "sqlite"
            else "Use standard PostgreSQL syntax (e.g., `DATE_TRUNC`, `TO_CHAR`, `::numeric`)."
        )

        system_prompt = f"""You are a Senior BI SQL Engineer writing a single read-only {dialect.upper()} query.

DATABASE SCHEMA & VALUE CONTEXT:
{state['schema_context']}

CRITICAL SQL RULES:
1. Output ONLY a single read-only `SELECT` or `WITH ... SELECT` statement. Never write DROP, DELETE, UPDATE, or INSERT.
2. {dialect_hint}
3. RELATIVE TIME ANCHORING: Always check the `[date range: ...]` in the schema context above! When the user asks about "last month", "this month", or "prior month", anchor relative to the LATEST month present in the dataset (for example, if max date is '2026-10-31', "last month" = '2026-10' and the comparison prior month = '2026-09'). Never use `CURRENT_DATE` if it falls outside the dataset's date range.
4. DIAGNOSTIC ("WHY") QUESTIONS: When the user asks *why* a metric changed (e.g., "Why did revenue drop in the North region last month?"), compare the relevant periods (e.g., prior month vs latest month) grouped by key breakdown dimensions (like `product_category`) and include operational metrics (`SUM(revenue)`, `SUM(units_sold)`, `ROUND(AVG(shipping_delay_days), 2)`, `ROUND(AVG(discount_pct), 2)`) so the root cause is clear!
5. PROFIT MARGIN QUESTIONS: Compute profit margin accurately as `ROUND(SUM(profit) * 100.0 / NULLIF(SUM(revenue), 0), 2) AS profit_margin_pct` alongside `ROUND(SUM(revenue), 2) AS total_revenue` and `ROUND(SUM(profit), 2) AS total_profit`.
6. Keep results aggregated and concise (at most 60 rows) so charts and executive summaries are crystal clear."""

        user_prompt = f"Business Question: {state['question']}"
        if prev_error:
            user_prompt += (
                f"\n\nPREVIOUS SQL ATTEMPT:\n{state.get('sql_query', '')}\n\n"
                f"CRITIC ERROR / FEEDBACK TO FIX (Attempt {retry_num}/{MAX_SQL_RETRIES}):\n{prev_error}\n"
                "Please fix the exact issue above and return a corrected read-only SQL query."
            )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        draft: SQLDraft
        try:
            structured_llm = llm.with_structured_output(SQLDraft)
            res = structured_llm.invoke(messages)
            if isinstance(res, SQLDraft):
                draft = res
            elif isinstance(res, dict):
                draft = SQLDraft(**res)
            else:
                raw_resp = llm.invoke(messages)
                draft = _extract_sql_fallback(str(raw_resp.content))
        except Exception:
            raw_resp = llm.invoke(messages)
            draft = _extract_sql_fallback(str(raw_resp.content))

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        step_label = "SQL Writer" if retry_num == 0 else f"SQL Writer (Retry {retry_num})"
        new_trace = list(state.get("trace_steps", [])) + [
            {
                "node": step_label,
                "status": "done",
                "detail": draft.explanation or "Generated SQL query.",
                "duration_ms": elapsed_ms,
            }
        ]

        return {
            "sql_query": draft.sql_query.strip(),
            "sql_explanation": draft.explanation.strip(),
            "sql_error": None,
            "trace_steps": new_trace,
        }

    # ------------------------------------------------------------------
    # Node 2: SQL Critic & Safe Executor
    # ------------------------------------------------------------------
    def sql_critic_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw_sql = state.get("sql_query", "")
        retry_count = state.get("retry_count", 0)
        trace = list(state.get("trace_steps", []))

        try:
            clean_sql = validate_readonly_sql(raw_sql)
            df = db_manager.run_safe_query(clean_sql)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            err_msg = f"{type(exc).__name__}: {exc}"
            trace.append(
                {
                    "node": "SQL Critic",
                    "status": "retry" if retry_count < MAX_SQL_RETRIES else "error",
                    "detail": f"Validation/Execution failed — {err_msg}",
                    "duration_ms": elapsed_ms,
                }
            )
            return {
                "sql_error": err_msg,
                "retry_count": retry_count + 1,
                "trace_steps": trace,
            }

        # Semantic check: if 0 rows returned on early attempt, ask SQL Writer to verify date/filter values
        if df.empty and retry_count < 2:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            feedback = (
                "Query executed without syntax errors but returned 0 rows. "
                "Verify exact column values (case-sensitive in SQLite) and check the dataset's "
                "actual `[date range: ...]` in the schema context instead of using today's wall-clock date."
            )
            trace.append(
                {
                    "node": "SQL Critic",
                    "status": "retry",
                    "detail": "Query returned 0 rows — requesting filter/date range correction.",
                    "duration_ms": elapsed_ms,
                }
            )
            return {
                "sql_query": clean_sql,
                "sql_error": feedback,
                "retry_count": retry_count + 1,
                "trace_steps": trace,
            }

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        trace.append(
            {
                "node": "SQL Critic",
                "status": "done",
                "detail": f"Verified read-only safety & fetched {len(df)} rows × {len(df.columns)} columns.",
                "duration_ms": elapsed_ms,
            }
        )

        # Cap stored records to 500 rows for lightweight state serialization
        capped_df = df.head(500)
        return {
            "sql_query": clean_sql,
            "sql_error": None,
            "result_records": capped_df.to_dict(orient="records"),
            "result_columns": list(capped_df.columns),
            "trace_steps": trace,
        }

    def route_after_critic(
        state: AgentState,
    ) -> Literal["sql_writer", "stats_and_chart", "insight_analyst"]:
        if state.get("sql_error"):
            if state.get("retry_count", 0) <= MAX_SQL_RETRIES:
                return "sql_writer"
            return "insight_analyst"
        return "stats_and_chart"

    # ------------------------------------------------------------------
    # Node 3: Deterministic Python Stats & Plotly Chart Spec Builder
    # ------------------------------------------------------------------
    def stats_and_chart_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        cols = state.get("result_columns", [])
        records = state.get("result_records", [])
        df = pd.DataFrame(records, columns=cols) if cols else pd.DataFrame()

        stats_info = run_statistical_checks(df)
        default_spec = infer_default_chart_spec(df, question=state["question"])
        chosen_spec = default_spec

        if len(df) >= 2 and default_spec.chart_type != "none":
            col_dtypes = {c: str(df[c].dtype) for c in df.columns}
            sample_preview = df.head(6).to_string(index=False)
            chart_prompt = f"""Choose the clearest Plotly chart configuration to visualize this SQL result for a business user.

User Question: {state['question']}
Available Columns & Dtypes (MUST match exact column names): {json.dumps(col_dtypes)}
Sample Rows:
{sample_preview}

Rules:
- Choose `bar` for category comparisons or 2-period grouped comparisons (`color_col` = period/month).
- Choose `line` or `area` for multi-period trends over time (`>= 3` time points).
- Choose `pie` ONLY for simple part-to-whole shares (`<= 6` categories, no negative numbers, no `color_col`).
- `x_col`, `y_col`, and optional `color_col` MUST be exact keys from Available Columns & Dtypes."""
            try:
                chart_llm = llm.with_structured_output(ChartSpec)
                spec_res = chart_llm.invoke([HumanMessage(content=chart_prompt)])
                if isinstance(spec_res, ChartSpec):
                    chosen_spec = spec_res
                elif isinstance(spec_res, dict):
                    chosen_spec = ChartSpec(**spec_res)
            except Exception:
                chosen_spec = default_spec

            # Ensure columns exist on df; otherwise keep deterministic fallback
            if chosen_spec.x_col not in df.columns or chosen_spec.y_col not in df.columns:
                chosen_spec = default_spec

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        trace = list(state.get("trace_steps", [])) + [
            {
                "node": "Stats & Chart",
                "status": "done",
                "detail": f"Computed Python stats & configured `{chosen_spec.chart_type}` chart.",
                "duration_ms": elapsed_ms,
            }
        ]

        return {
            "stats_summary": stats_info,
            "chart_spec": chosen_spec.model_dump(),
            "trace_steps": trace,
        }

    # ------------------------------------------------------------------
    # Node 4: Executive Insight Analyst
    # ------------------------------------------------------------------
    def insight_analyst_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        trace = list(state.get("trace_steps", []))

        if state.get("sql_error"):
            err_summary = (
                "I wasn't able to run a valid SQL query for that question after multiple self-correction attempts.\n\n"
                f"- **Details**: `{state['sql_error']}`\n"
                "- **Tip**: Try mentioning a specific metric or column from the dataset (check **Table Schema** in the sidebar)."
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            trace.append(
                {
                    "node": "Insight Analyst",
                    "status": "error",
                    "detail": "Returned graceful error guidance after max SQL retries.",
                    "duration_ms": elapsed_ms,
                }
            )
            return {"executive_summary": err_summary, "trace_steps": trace}

        cols = state.get("result_columns", [])
        records = state.get("result_records", [])
        df = pd.DataFrame(records, columns=cols) if cols else pd.DataFrame()
        stats_text = state.get("stats_summary", {}).get("summary_text", "")

        if df.empty:
            empty_msg = (
                "The query ran cleanly, but **no matching records** were found for those filters.\n\n"
                "- Try broadening the date range or checking available categories in the sidebar schema."
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            trace.append(
                {
                    "node": "Insight Analyst",
                    "status": "done",
                    "detail": "Reported empty result set cleanly.",
                    "duration_ms": elapsed_ms,
                }
            )
            return {"executive_summary": empty_msg, "trace_steps": trace}

        table_preview = df.head(35).to_string(index=False)
        analyst_system = """You are a clear, executive-level Business Intelligence Analyst.
Write a concise, easy-to-read answer for a non-technical business leader based STRICTLY on the SQL results and verified Python statistics below.

Formatting Guidelines:
- Start with a **1-sentence headline takeaway** in bold with the key numbers.
- Follow with **2 to 3 short bullet points** explaining the main drivers, comparisons, or patterns.
- End with a short **Recommendation:** bullet if an actionable takeaway is clear.
- Keep language simple, calm, and jargon-free (never mention SQL syntax, DataFrames, or Z-score formulas in the main answer).
- Never invent or extrapolate numbers that are not in the data."""

        analyst_user = f"""User Question: {state['question']}

Verified Python Statistical Summary:
{stats_text}

SQL Query Result Table (up to 35 rows):
{table_preview}"""

        resp = llm.invoke(
            [SystemMessage(content=analyst_system), HumanMessage(content=analyst_user)]
        )
        summary_md = escape_currency_dollars(str(resp.content).strip())

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        trace.append(
            {
                "node": "Insight Analyst",
                "status": "done",
                "detail": "Synthesized executive summary from SQL data & Python stats.",
                "duration_ms": elapsed_ms,
            }
        )
        return {"executive_summary": summary_md, "trace_steps": trace}

    # Build LangGraph v1 StateGraph
    workflow = StateGraph(AgentState)
    workflow.add_node("sql_writer", sql_writer_node)
    workflow.add_node("sql_critic", sql_critic_node)
    workflow.add_node("stats_and_chart", stats_and_chart_node)
    workflow.add_node("insight_analyst", insight_analyst_node)

    workflow.add_edge(START, "sql_writer")
    workflow.add_edge("sql_writer", "sql_critic")
    workflow.add_conditional_edges(
        "sql_critic",
        route_after_critic,
        {
            "sql_writer": "sql_writer",
            "stats_and_chart": "stats_and_chart",
            "insight_analyst": "insight_analyst",
        },
    )
    workflow.add_edge("stats_and_chart", "insight_analyst")
    workflow.add_edge("insight_analyst", END)

    return workflow.compile()


def run_analyst_query(
    question: str,
    db_manager: DatabaseManager,
    api_key: str | None = None,
    model_name: str = "openai/gpt-oss-120b",
    llm: BaseChatModel | None = None,
    on_step_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> AnalystResult:
    """Execute the full 4-node LangGraph workflow for a user question and return `AnalystResult`."""
    active_llm = llm if llm is not None else create_groq_llm(api_key=api_key or "", model_name=model_name)
    graph = build_analyst_graph(db_manager=db_manager, llm=active_llm)

    initial_state: AgentState = {
        "question": question.strip(),
        "dialect": db_manager.dialect,
        "schema_context": db_manager.get_schema_context(),
        "sql_query": "",
        "sql_explanation": "",
        "sql_error": None,
        "retry_count": 0,
        "result_records": [],
        "result_columns": [],
        "stats_summary": {},
        "chart_spec": None,
        "executive_summary": "",
        "trace_steps": [],
    }

    final_state: dict[str, Any] = dict(initial_state)

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in event.items():
            if isinstance(node_update, dict):
                final_state.update(node_update)
                if on_step_callback is not None:
                    on_step_callback(node_name, node_update)

    cols = final_state.get("result_columns", [])
    records = final_state.get("result_records", [])
    result_df = pd.DataFrame(records, columns=cols) if cols else pd.DataFrame()
    chart_spec_dict = final_state.get("chart_spec")

    fig = (
        build_plotly_chart(result_df, chart_spec=chart_spec_dict, question=question)
        if not result_df.empty and not final_state.get("sql_error")
        else None
    )

    return AnalystResult(
        question=question,
        executive_summary=final_state.get("executive_summary", ""),
        sql_query=final_state.get("sql_query", ""),
        sql_explanation=final_state.get("sql_explanation", ""),
        dataframe=result_df,
        figure=fig,
        stats_summary=final_state.get("stats_summary", {}),
        chart_spec=chart_spec_dict,
        retry_count=int(final_state.get("retry_count", 0)),
        trace_steps=final_state.get("trace_steps", []),
        error=final_state.get("sql_error"),
    )
