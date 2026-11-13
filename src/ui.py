"""Presentation and UI helpers for the DataGenie Business Analyst dashboard.

Designed following the `frontend-design` principles:
- Grounded in high-velocity executive business intelligence (clean, punchy, tabular).
- Eliminates AI-generated tells (no tracked-out all-caps labels, no fake squiggles, no fake 01/02 numbering).
- Dynamic CSS tokens that adapt naturally to light and dark Streamlit themes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from typing import Any

import pandas as pd

from src.eda_engine import find_default_timeseries_columns
from src.stats_analyzer import (
    KPI_COMPACT_K_THRESHOLD,
    PREFERRED_ADDITIVE_METRIC_TOKENS,
    format_metric_value,
    get_quantitative_numeric_columns,
    is_non_additive_metric,
    tokenize_col_name,
)


APP_CSS = """
<style>
/* ---------- Layout & Container ---------------------------------------- */
.block-container {
  max-width: 1220px;
  padding-top: 1.2rem;
  padding-bottom: 4.5rem;
}
[data-testid="stHeader"] { background: transparent; }

/* ---------- Unified Reactive Design Tokens ---------------------------- */
:root {
  color-scheme: light dark;
}

/* Base Dark Tokens */
:root, .stApp {
  --dg-bg-canvas: #0C1322;
  --dg-surface: #131E35;
  --dg-surface-hover: #1A2846;
  --dg-border: #1E2F52;
  --dg-text-primary: #F1F5F9;
  --dg-text-muted: #8195B0;
  --dg-coral: #FF6B6B;
  --dg-blue: #38BDF8;
  --dg-sky: #38BDF8;
  --dg-emerald: #10B981;
  --dg-amber: #F59E0B;
  
  --dg-sidebar-bg: #080E1B;
  --dg-sidebar-border: #182744;
  --dg-tab-container-bg: #131E35;
  --dg-tab-container-border: #1E2F52;
  --dg-tab-text: #8195B0;
  --dg-tab-hover-bg: rgba(255, 255, 255, 0.06);
  --dg-tab-hover-text: #F1F5F9;
  --dg-tab-active-gradient: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
  --dg-tab-active-shadow: 0 4px 14px rgba(255, 107, 107, 0.42);
  --dg-metric-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
  --dg-brand-gradient: linear-gradient(135deg, #FF6B6B 0%, #38BDF8 100%);
  --dg-brand-shadow: 0 4px 14px rgba(255, 107, 107, 0.32);
}

/* Browser-native light-dark() support */
@supports (color-scheme: light dark) {
  :root, .stApp {
    --dg-bg-canvas: light-dark(#F4F8FC, #0C1322);
    --dg-surface: light-dark(#FFFFFF, #131E35);
    --dg-surface-hover: light-dark(#EDF4FB, #1A2846);
    --dg-border: light-dark(#D4E3F2, #1E2F52);
    --dg-text-primary: light-dark(#0F1E36, #F1F5F9);
    --dg-text-muted: light-dark(#596F8C, #8195B0);
    --dg-coral: light-dark(#FF5A5F, #FF6B6B);
    --dg-blue: light-dark(#0284C7, #38BDF8);
    --dg-sky: light-dark(#0284C7, #38BDF8);
    --dg-emerald: light-dark(#10B981, #10B981);
    --dg-amber: light-dark(#F59E0B, #F59E0B);
    
    --dg-sidebar-bg: light-dark(#E6EFF7, #080E1B);
    --dg-sidebar-border: light-dark(#CCE0F2, #182744);
    --dg-tab-container-bg: light-dark(#FFFFFF, #131E35);
    --dg-tab-container-border: light-dark(#D4E3F2, #1E2F52);
    --dg-tab-text: light-dark(#596F8C, #8195B0);
    --dg-tab-hover-bg: light-dark(#EDF4FB, rgba(255, 255, 255, 0.06));
    --dg-tab-hover-text: light-dark(#0F1E36, #F1F5F9);
    --dg-tab-active-gradient: light-dark(linear-gradient(135deg, #FF5A5F 0%, #FF4146 100%), linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%));
    --dg-tab-active-shadow: light-dark(0 4px 14px rgba(255, 90, 95, 0.35), 0 4px 14px rgba(255, 107, 107, 0.42));
    --dg-metric-shadow: light-dark(0 2px 10px rgba(0, 0, 0, 0.05), 0 4px 16px rgba(0, 0, 0, 0.25));
    --dg-brand-gradient: light-dark(linear-gradient(135deg, #FF5A5F 0%, #0284C7 100%), linear-gradient(135deg, #FF6B6B 0%, #38BDF8 100%));
    --dg-brand-shadow: light-dark(0 4px 12px rgba(255, 90, 95, 0.28), 0 4px 14px rgba(255, 107, 107, 0.32));
  }
}

/* Explicit Light Mode Overrides for when Streamlit sets colorScheme: light */
.stApp[style*="color-scheme: light"],
[data-theme="light"] .stApp,
.stApp.st-theme-light {
  --dg-bg-canvas: #F4F8FC !important;
  --dg-surface: #FFFFFF !important;
  --dg-surface-hover: #EDF4FB !important;
  --dg-border: #D4E3F2 !important;
  --dg-text-primary: #0F1E36 !important;
  --dg-text-muted: #596F8C !important;
  --dg-coral: #FF5A5F !important;
  --dg-blue: #0284C7 !important;
  --dg-sky: #0284C7 !important;
  
  --dg-sidebar-bg: #E6EFF7 !important;
  --dg-sidebar-border: #CCE0F2 !important;
  --dg-tab-container-bg: #FFFFFF !important;
  --dg-tab-container-border: #D4E3F2 !important;
  --dg-tab-text: #596F8C !important;
  --dg-tab-hover-bg: #EDF4FB !important;
  --dg-tab-hover-text: #0F1E36 !important;
  --dg-tab-active-gradient: linear-gradient(135deg, #FF5A5F 0%, #FF4146 100%) !important;
  --dg-tab-active-shadow: 0 4px 14px rgba(255, 90, 95, 0.35) !important;
  --dg-metric-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
  --dg-brand-gradient: linear-gradient(135deg, #FF5A5F 0%, #0284C7 100%) !important;
  --dg-brand-shadow: 0 4px 12px rgba(255, 90, 95, 0.28) !important;
}

.stApp {
  background-color: var(--dg-bg-canvas);
  color: var(--dg-text-primary);
}

/* ---------- Sidebar High Contrast ------------------------------------- */
section[data-testid="stSidebar"] {
  background-color: var(--dg-sidebar-bg) !important;
  border-right: 1px solid var(--dg-sidebar-border) !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] [data-testid="stRadioOption"] p,
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p,
section[data-testid="stSidebar"] [data-testid="stRadioOption"] span {
  color: var(--dg-text-primary) !important;
  font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadioCaption"] p,
section[data-testid="stSidebar"] [data-testid="stRadioCaption"] span,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .dg-foot {
  color: var(--dg-text-muted) !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
  background: var(--dg-surface) !important;
  border: 1px solid var(--dg-border) !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary span,
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] button p,
section[data-testid="stSidebar"] button span {
  color: var(--dg-text-primary) !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
  background: var(--dg-surface) !important;
  color: var(--dg-text-primary) !important;
  border: 1px solid var(--dg-border) !important;
}
div[data-testid="stSkillsNudge"] {
  display: none !important;
}

/* ---------- Brand Header (Sidebar) ------------------------------------ */
.dg-brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 0.1rem 0 0.4rem;
}
.dg-brand .mark {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: var(--dg-brand-gradient);
  color: #FFFFFF;
  display: grid;
  place-items: center;
  font: 700 1.05rem "Plus Jakarta Sans", sans-serif;
  box-shadow: var(--dg-brand-shadow);
}
.dg-brand .title {
  font: 700 1.35rem/1.1 "Plus Jakarta Sans", sans-serif;
  letter-spacing: -0.02em;
  color: var(--dg-text-primary);
}
.dg-brand .subtitle {
  font-size: 0.78rem;
  color: var(--dg-text-muted);
  margin-top: 2px;
}

/* ---------- Workspace Hero Banner ------------------------------------- */
.dg-hero {
  padding: 0.2rem 0 1rem;
  border-bottom: 1px solid var(--dg-border);
  margin-bottom: 1.2rem;
}
.dg-breadcrumb {
  font: 500 0.8rem "Inter", sans-serif;
  color: var(--dg-text-muted);
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.dg-hero-title {
  font: 800 2.15rem/1.15 "Plus Jakarta Sans", sans-serif;
  letter-spacing: -0.03em;
  margin: 0.35rem 0 0.75rem;
  color: var(--dg-text-primary);
}
.dg-pill-group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}
.dg-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.24rem 0.75rem;
  border: 1px solid var(--dg-border);
  border-radius: 999px;
  background: var(--dg-surface);
  color: var(--dg-text-primary);
  font: 500 0.8rem "Inter", sans-serif;
  white-space: nowrap;
}
.dg-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dg-emerald);
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.65);
}
.dg-dot.warn {
  background: var(--dg-amber);
  box-shadow: 0 0 8px rgba(245, 158, 11, 0.65);
}

/* ---------- Section Titles -------------------------------------------- */
.dg-section-header {
  margin: 1.8rem 0 0.8rem;
}
.dg-section-title {
  font: 700 1.28rem "Plus Jakarta Sans", sans-serif;
  letter-spacing: -0.02em;
  color: var(--dg-text-primary);
  margin: 0 0 0.2rem;
}
.dg-section-subtitle {
  color: var(--dg-text-muted);
  font-size: 0.88rem;
  margin: 0;
}

/* ---------- Headline KPI Cards & Transparent Sparklines --------------- */
div[data-testid="stMetric"] {
  background: var(--dg-surface) !important;
  border: 1px solid var(--dg-border) !important;
  border-radius: 12px !important;
  padding: 16px 18px !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  box-shadow: var(--dg-metric-shadow);
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: var(--dg-coral) !important;
}
[data-testid="stMetricValue"] {
  font-family: "Plus Jakarta Sans", sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.03em !important;
  font-size: 1.85rem !important;
  color: var(--dg-text-primary) !important;
}
[data-testid="stMetricLabel"] p {
  font: 600 0.78rem "Inter", sans-serif !important;
  color: var(--dg-text-muted) !important;
}

/* Transparent Vega-Lite Sparkline Charts (Eliminates Black/Dark Boxes) */
div[data-testid="stMetric"] [data-testid="stVegaLiteChart"],
div[data-testid="stMetricChart"] svg,
div[data-testid="stMetricChart"] svg rect.background,
div[data-testid="stMetricChart"] .vega-embed {
  background: transparent !important;
  fill: transparent !important;
  margin-top: 4px;
}
div[data-testid="stMetricChart"] svg g.mark-area path {
  opacity: 0.2 !important;
}
div[data-testid="stMetricChart"] svg g.mark-line path {
  opacity: 1 !important;
  stroke-width: 2.2px !important;
}

/* ---------- Prominent Top Navigation Tabs (React-Aria & BaseWeb) ------ */
div[data-testid="stTabs"] [role="tablist"]::after,
[role="tablist"]::after {
  display: none !important;
}
div[data-testid="stTabs"] [role="tablist"],
[role="tablist"],
[data-baseweb="tab-list"] {
  display: inline-flex !important;
  gap: 0.4rem !important;
  padding: 0.3rem 0.4rem !important;
  background: var(--dg-tab-container-bg) !important;
  border: 1px solid var(--dg-tab-container-border) !important;
  border-radius: 999px !important;
  box-shadow: var(--dg-metric-shadow) !important;
  margin-top: 0.1rem !important;
  margin-bottom: 1.2rem !important;
  width: auto !important;
}
.react-aria-SelectionIndicator,
div[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTab"] .react-aria-SelectionIndicator,
[data-baseweb="tab-highlight"] {
  display: none !important;
}
div[data-testid="stTabs"] [data-testid="stTab"],
div[data-testid="stTabs"] button[data-testid="stTab"],
div[data-testid="stTabs"] div[data-testid="stTab"],
div[data-testid="stTabs"] [role="tab"],
[data-baseweb="tab"] {
  font-family: "Plus Jakarta Sans", sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
  color: var(--dg-tab-text) !important;
  border-radius: 999px !important;
  padding: 0.45rem 1.3rem !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.16s cubic-bezier(0.16, 1, 0.3, 1) !important;
  height: auto !important;
  cursor: pointer !important;
}
div[data-testid="stTabs"] [data-testid="stTab"] p,
div[data-testid="stTabs"] [data-testid="stTab"] span,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] [role="tab"] span {
  color: var(--dg-tab-text) !important;
}
div[data-testid="stTabs"] [data-testid="stTab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover,
[data-baseweb="tab"]:hover {
  color: var(--dg-tab-hover-text) !important;
  background: var(--dg-tab-hover-bg) !important;
}
div[data-testid="stTabs"] [data-testid="stTab"]:hover p,
div[data-testid="stTabs"] [role="tab"]:hover p {
  color: var(--dg-tab-hover-text) !important;
}
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
div[data-testid="stTabs"] [data-testid="stTab"][data-selected="true"],
div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-baseweb="tab"][aria-selected="true"] {
  color: #FFFFFF !important;
  background: var(--dg-tab-active-gradient) !important;
  box-shadow: var(--dg-tab-active-shadow) !important;
}
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p,
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] span,
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] span {
  color: #FFFFFF !important;
}

/* ---------- Interactive Starter Prompts ------------------------------- */
.st-key-starters button {
  min-height: 88px;
  justify-content: flex-start;
  align-items: flex-start;
  text-align: left;
  padding: 1rem 1.1rem;
  border-radius: 12px;
  border: 1px solid var(--dg-border);
  background: var(--dg-surface);
  transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: var(--dg-metric-shadow);
}
.st-key-starters button p {
  text-align: left;
  font-size: 0.93rem;
  font-weight: 500;
  line-height: 1.4;
  color: var(--dg-text-primary) !important;
}
.st-key-starters button:hover {
  transform: translateY(-2px);
  border-color: var(--dg-coral);
  background: var(--dg-surface-hover);
}

/* ---------- Execution Pipeline Strip --------------------------------- */
.dg-pipeline {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  margin: 0.6rem 0 0.3rem;
}
.dg-pipeline-step {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.22rem 0.65rem;
  border-radius: 7px;
  border: 1px solid var(--dg-border);
  background: var(--dg-surface);
  color: var(--dg-text-primary);
  font: 500 0.76rem "JetBrains Mono", monospace;
}
.dg-pipeline-step .ms {
  color: var(--dg-text-muted);
}
.dg-pipeline-step.retry {
  border-color: var(--dg-amber);
  background: color-mix(in srgb, var(--dg-amber) 16%, transparent);
}
.dg-pipeline-step.error {
  border-color: var(--dg-coral);
  background: color-mix(in srgb, var(--dg-coral) 16%, transparent);
}
.dg-pipeline-arrow {
  color: var(--dg-text-muted);
  font-size: 0.82rem;
}

/* ---------- Insight Highlight Cards ---------------------------------- */
.dg-insight-card {
  padding: 0.85rem 1.05rem;
  border-radius: 10px;
  border: 1px solid var(--dg-border);
  background: var(--dg-surface);
  color: var(--dg-text-primary);
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.dg-insight-card .title {
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--dg-text-primary);
}
.dg-insight-card .desc {
  color: var(--dg-text-muted);
  font-size: 0.8rem;
  margin-top: 2px;
}
.dg-insight-card .badge {
  font: 600 0.82rem "JetBrains Mono", monospace;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  background: color-mix(in srgb, var(--dg-coral) 14%, transparent);
  color: var(--dg-coral);
}

/* ---------- Empty State ---------------------------------------------- */
.dg-empty-state {
  margin: 0.6rem 0 1.2rem;
}
.dg-empty-state .headline {
  font: 700 1.45rem "Plus Jakarta Sans", sans-serif;
  letter-spacing: -0.02em;
  color: var(--dg-text-primary);
}
.dg-empty-state .subtext {
  color: var(--dg-text-muted);
  font-size: 0.92rem;
  margin-top: 0.25rem;
}

/* ---------- Containers & Widgets ------------------------------------- */
div[data-testid="stExpander"] {
  background: var(--dg-surface) !important;
  border: 1px solid var(--dg-border) !important;
  border-radius: 10px !important;
}
div[data-testid="stStatusWidget"] {
  background: var(--dg-surface) !important;
  border: 1px solid var(--dg-border) !important;
  border-radius: 10px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: var(--dg-border) !important;
  background-color: var(--dg-surface);
  border-radius: 12px !important;
}
div[data-testid="stChatMessage"] {
  background: var(--dg-surface);
  border: 1px solid var(--dg-border);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
}

/* ---------- Sidebar Footer ------------------------------------------- */
.dg-foot, .dg-sidebar-foot {
  color: var(--dg-text-muted);
  font-size: 0.78rem;
  line-height: 1.5;
}
.dg-foot kbd, .dg-sidebar-foot kbd {
  font: 500 0.72rem "JetBrains Mono", monospace;
  padding: 1px 6px;
  border-radius: 4px;
  border: 1px solid var(--dg-border);
  background: var(--dg-surface);
  color: var(--dg-text-primary);
}
</style>
"""

_DARK_APP_CSS = APP_CSS
_LIGHT_APP_CSS = APP_CSS


def get_app_css() -> str:
    """Return unified dynamic CSS matching the active Streamlit theme."""
    return APP_CSS


def esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation into markdown markup."""
    return html.escape(str(value), quote=True)


def brand_html() -> str:
    """Render the high-energy brand monogram and title in the sidebar."""
    return (
        '<div class="dg-brand">'
        '<div class="mark">DG</div>'
        '<div>'
        '<div class="title">DataGenie</div>'
        '<div class="subtitle">Autonomous Data Analyst</div>'
        '</div>'
        '</div>'
    )


def hero_html(breadcrumb: str, title: str, pills: list[tuple[str, str]]) -> str:
    """Render the dashboard workspace hero banner with metadata pills."""
    pill_elements: list[str] = []
    for text, kind in pills:
        dot_cls = "warn" if kind == "warn" else ""
        dot_html = f'<span class="dg-dot {dot_cls}"></span>' if kind in ("ok", "warn") else ""
        pill_elements.append(
            f'<span class="dg-pill">{dot_html}<span>{esc(text)}</span></span>'
        )
    return (
        '<div class="dg-hero">'
        f'<div class="dg-breadcrumb"><span>Workspace</span> <span>/</span> <span>{esc(breadcrumb)}</span></div>'
        f'<div class="dg-hero-title">{esc(title)}</div>'
        f'<div class="dg-pill-group">{"".join(pill_elements)}</div>'
        '</div>'
    )


def section_header_html(title: str, subtitle: str = "") -> str:
    """Render a clean section title with conversational subtext (no fake numbering)."""
    sub = f'<p class="dg-section-subtitle">{esc(subtitle)}</p>' if subtitle else ""
    return (
        '<div class="dg-section-header">'
        f'<div class="dg-section-title">{esc(title)}</div>'
        f'{sub}'
        '</div>'
    )


def pipeline_html(trace_steps: list[dict[str, Any]]) -> str:
    """Render a clean, high-tech horizontal execution chain for the agent workflow."""
    parts: list[str] = []
    for i, step in enumerate(trace_steps):
        status = step.get("status", "done")
        css = "retry" if status == "retry" else ("error" if status == "error" else "")
        duration = int(step.get("duration_ms", 0))
        dur_str = f"{duration / 1000:.1f}s" if duration >= 1000 else f"{duration}ms"
        parts.append(
            f'<span class="dg-pipeline-step {css}" title="{esc(step.get("detail", ""))}">'
            f'<span>{esc(step.get("node", ""))}</span>'
            f'<span class="ms">{dur_str}</span>'
            f'</span>'
        )
        if i < len(trace_steps) - 1:
            parts.append('<span class="dg-pipeline-arrow">➔</span>')
    return f'<div class="dg-pipeline">{"".join(parts)}</div>'


def correlation_insights_html(pairs: list[dict[str, Any]]) -> str:
    """Render executive summary cards for key Pearson correlation pairs."""
    if not pairs:
        return ""
    cards: list[str] = []
    for p in pairs:
        r = p["r"]
        is_pos = r > 0
        desc = "Strong positive driver" if abs(r) >= 0.7 else ("Moderate positive" if is_pos else ("Strong inverse driver" if abs(r) >= 0.7 else "Moderate inverse"))
        badge_style = "color: #10B981; background: rgba(16, 185, 129, 0.12);" if is_pos else "color: #F43F5E; background: rgba(244, 63, 94, 0.12);"
        cards.append(
            f'<div class="dg-insight-card">'
            f'<div>'
            f'<div class="title">{esc(p["col1_label"])} ↔ {esc(p["col2_label"])}</div>'
            f'<div class="desc">{desc}</div>'
            f'</div>'
            f'<span class="badge" style="{badge_style}">r = {r:+.2f}</span>'
            f'</div>'
        )
    return "".join(cards)


# ---------------------------------------------------------------------------
# Headline KPI computation
# ---------------------------------------------------------------------------

@dataclass
class Kpi:
    label: str
    value: str
    delta: str | None = None
    delta_description: str | None = None
    chart: list[float] = field(default_factory=list)
    help: str | None = None


def _fmt_number(val: float, money: bool = False, pct: bool = False) -> str:
    col_hint = "pct" if pct else ("revenue" if money else "value")
    return format_metric_value(val, col_hint, compact_k_threshold=KPI_COMPACT_K_THRESHOLD)


def compute_headline_kpis(df: pd.DataFrame | None, max_metrics: int = 3) -> list[Kpi]:
    """Derive up to 4 headline tiles (latest month vs. prior month + monthly sparkline)."""
    if df is None or df.empty:
        return []

    date_col, _ = find_default_timeseries_columns(df)
    numeric_cols = get_quantitative_numeric_columns(df)
    additive_cols = [c for c in numeric_cols if not is_non_additive_metric(c)]
    non_additive_cols = [c for c in numeric_cols if is_non_additive_metric(c)]

    ranked: list[str] = []
    for pref in PREFERRED_ADDITIVE_METRIC_TOKENS:
        ranked += [
            c
            for c in additive_cols
            if pref in tokenize_col_name(c) and c not in ranked
        ]
    ranked += [c for c in additive_cols if c not in ranked]
    ranked += [c for c in non_additive_cols if c not in ranked]
    metrics = ranked[:max_metrics]

    kpis: list[Kpi] = []

    if date_col and metrics:
        dates = pd.to_datetime(df[date_col], format="mixed", errors="coerce")
        work = df.assign(_period=dates.dt.to_period("M")).dropna(subset=["_period"])
        agg_map = {
            col: ("mean" if is_non_additive_metric(col) else "sum")
            for col in metrics
        }
        monthly = work.groupby("_period").agg(agg_map).sort_index()
        counts = work.groupby("_period").size().sort_index()

        if len(monthly) >= 1:
            latest = monthly.index[-1]
            prior_label = f"vs {monthly.index[-2].strftime('%b')}" if len(monthly) >= 2 else None
            for col in metrics:
                series = pd.to_numeric(monthly[col], errors="coerce").astype(float)
                cur = float(series.iloc[-1])
                delta = None
                if (
                    len(series) >= 2
                    and pd.notna(cur)
                    and pd.notna(series.iloc[-2])
                    and float(series.iloc[-2]) != 0
                ):
                    delta = f"{(cur - float(series.iloc[-2])) / abs(float(series.iloc[-2])) * 100:+.1f}%"
                agg_word = "average" if is_non_additive_metric(col) else "sum"
                kpis.append(
                    Kpi(
                        label=f"{col.replace('_', ' ').title()} ({latest.strftime('%b %Y')})",
                        value=format_metric_value(
                            cur,
                            col,
                            compact_k_threshold=KPI_COMPACT_K_THRESHOLD,
                        ),
                        delta=delta,
                        delta_description=prior_label,
                        chart=[round(float(v), 2) if pd.notna(v) else 0.0 for v in series.tolist()],
                        help=f"Monthly {agg_word} of `{col}`; sparkline shows entire dataset range.",
                    )
                )
            cur_n = int(counts.iloc[-1])
            delta_n = None
            if len(counts) >= 2 and int(counts.iloc[-2]) != 0:
                delta_n = f"{(cur_n - int(counts.iloc[-2])) / int(counts.iloc[-2]) * 100:+.1f}%"
            kpis.append(
                Kpi(
                    label=f"Total Orders ({latest.strftime('%b %Y')})",
                    value=f"{cur_n:,}",
                    delta=delta_n,
                    delta_description=prior_label,
                    chart=[int(v) for v in counts.tolist()],
                    help="Monthly completed orders count.",
                )
            )
            return kpis[:4]

    # Fallback when there is no usable date column
    missing_pct = float(df.isna().sum().sum()) / max(1, df.size) * 100
    return [
        Kpi("Total Rows", f"{len(df):,}"),
        Kpi("Columns", f"{len(df.columns):,}"),
        Kpi("Numeric Columns", f"{len(numeric_cols):,}"),
        Kpi("Data Completeness", f"{100.0 - missing_pct:.1f}%"),
    ]


def date_range_label(df: pd.DataFrame | None) -> str | None:
    """Format the dataset chronological span into a clean label."""
    if df is None or df.empty:
        return None
    date_col, _ = find_default_timeseries_columns(df)
    if not date_col:
        return None
    dates = pd.to_datetime(df[date_col], format="mixed", errors="coerce").dropna()
    if dates.empty:
        return None
    return f"{dates.min():%b %Y} → {dates.max():%b %Y}"
