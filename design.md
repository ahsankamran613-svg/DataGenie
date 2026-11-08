# DataGenie — Design Spec (for Stitch)

## 1. Product Overview
**DataGenie** is an autonomous Business Intelligence & Data Analyst web app. Non-technical users load a demo dataset, upload a messy CSV, or connect Supabase/PostgreSQL, then ask business questions in plain English (e.g. *"Why did revenue drop in the North region last month?"*). A multi-agent AI returns an executive summary, an interactive chart, and a transparent "Under the Hood" trace of the SQL and statistics.

**Audience:** recruiters, hiring managers and business users viewing a portfolio demo.
**Feel:** modern, trustworthy, data-product. Think Linear, Vercel and Hex in dark mode. Calm, dense but breathable.

### Design principles
1. **Answer first**: the executive summary and chart are the hero. Technical detail is tucked into an expander.
2. **Show the work**: transparency (SQL, stats, agent steps) builds trust.
3. **Zero-friction start**: one click loads the demo and starts the experience.
4. **Streamlit-feasible**: every element must be reproducible with Streamlit components plus injected CSS (see §3).

---

## 2. Design Tokens

### Colors (dark theme)
| Token | Hex | Use |
|---|---|---|
| `bg-base` | `#0B1020` | App background |
| `bg-surface` | `#121933` | Cards, sidebar |
| `bg-elevated` | `#1A2347` | Hover, expanders, inputs |
| `border` | `#2A3560` | 1px card borders |
| `primary` | `#6366F1` | Buttons, active tab, links (indigo) |
| `accent` | `#2DD4BF` | Positive deltas, success, highlights (teal) |
| `danger` | `#F43F5E` | Negative deltas, anomalies, errors |
| `warning` | `#F59E0B` | Cleaning warnings, retries |
| `text-primary` | `#E6E9F5` | Headings, body |
| `text-muted` | `#8A93B8` | Captions, labels |

Chart series palette: `#6366F1`, `#2DD4BF`, `#F59E0B`, `#EC4899`, `#38BDF8`, `#A3E635`.
Cards use a subtle glass effect: `background: rgba(18,25,51,0.7)`, 1px `border`, soft shadow `0 8px 24px rgba(0,0,0,0.35)`.

### Typography
- Font: **Inter** (UI), **JetBrains Mono** (SQL/code).
- H1 28/700, H2 20/600, H3 16/600, body 14/400, caption 12/400 muted. KPI numbers 28/700.

### Shape & spacing
- Radius: cards 14px, buttons/inputs 10px, chips 999px.
- Spacing scale: 4 / 8 / 12 / 16 / 24 / 32.
- Primary button: indigo fill, white text, subtle glow on hover.

---

## 3. Layout & Streamlit Constraints
- **Layout:** left **sidebar** (≈300px) + main area with **2 tabs** at the top: `💬 AI Business Analyst` and `🧹 Data Health & Auto-EDA`.
- Max content width ≈ 1100px, centered.
- Allowed components: `st.sidebar`, `st.tabs`, `st.chat_message`, `st.chat_input`, `st.metric`, `st.columns`, `st.expander`, `st.dataframe`, `st.plotly_chart`, `st.status`, `st.button`, `st.selectbox`, `st.text_input`, `st.radio`.
- **Do not design:** custom top navbars, drag-and-drop dashboards, modals with complex animation, sticky floating panels, or multi-page routing.
- Responsive: columns collapse to a single column on narrow screens. Keep the layout simple.

---

## 4. Component Library
| Component | Spec |
|---|---|
| **Logo header** | 🧞 "DataGenie" wordmark in indigo→teal gradient text, tagline "Ask your data anything" in muted caption |
| **Status chip** | Pill with a colored dot: green "API key connected", amber "No key", red "Connection failed" |
| **KPI card** | Surface card: muted label, large number, delta chip (teal ▲ / red ▼) |
| **User bubble** | Right-leaning, indigo-tinted background |
| **Assistant message** | Full-width card: "Executive Summary" heading, bullet insights, chart below |
| **Starter question chip** | Outlined rounded button with ✨ icon; indigo border on hover |
| **Trace expander** | `🔍 View Agent Trace, SQL & Python Stats`; collapsed by default; contains sub-sections |
| **Code block** | JetBrains Mono, `bg-elevated`, copy icon, SQL keywords in indigo/teal |
| **Agent step row** | Icon + node name + status (✓ done, ⟳ running, ○ pending) + duration |
| **Cleaning log item** | Row with an icon, action text and affected count |
| **Inputs** | `bg-elevated`, 1px border, indigo focus ring |

---

## 5. Screens

### Screen A — Sidebar (shared across all screens)
Top to bottom:
1. Logo + tagline
2. **API status chip** and a "Groq API Key" password input (collapsed under "Use your own key")
3. **Model** select: `llama-3.3-70b-versatile`, `meta-llama/llama-4-scout-17b-16e-instruct`, `openai/gpt-oss-20b`
4. **Data Source** radio: `⚡ 1-Click Demo Dataset` / `📄 Upload Messy CSV` / `🔌 Connect Supabase / PostgreSQL`
5. **Schema viewer** expander: table name, columns with type badges (`DATE`, `TEXT`, `REAL`), row count
6. Footer: "Built with LangGraph · Groq · Streamlit" in muted caption

### Screen 1 — Tab 1: AI Business Analyst (populated)
- Chat history shows the question: *"Why did revenue drop in the North region last month?"*
- Assistant card, **Executive Summary**:
  - "North revenue fell **32%** vs. the prior month ($184K → $125K)."
  - "Driver: **Electronics** units sold dropped 41% while shipping delays rose from 2 to 9 days."
  - "Promo discounts ended on Oct 1, contributing an estimated 8 points of the decline."
  - "Recommendation: restock Electronics in the North and reinstate targeted promos."
- A row of 3 mini KPI cards: Revenue Δ `▼ 32%`, Top driver `Electronics`, Anomalies `3`.
- Interactive Plotly **bar chart** (revenue by category, North, this month vs. last month), dark template.
- Collapsed expander `🔍 View Agent Trace, SQL & Python Stats`. When opened it shows:
  - Agent steps (SQL Writer → SQL Critic → Stats & Chart → Insight Analyst) with retry count.
  - Final SQL in a code block.
  - Python stats table (total, mean, median, top/bottom contributors, z-score outliers).
- A `st.chat_input` is pinned at the bottom: "Ask a business question…".

### Screen 2 — Tab 2: Data Health & Auto-EDA (populated)
- Header row of 4 **Before/After KPI cards**: Rows `668 → 650`, Duplicates `18 → 0`, Missing values `74 → 0`, Columns standardized `9`.
- **Cleaning action log**, a list with icons, e.g. "🧽 Removed 18 duplicate rows", "💲 Converted `revenue` from currency text to numeric", "📅 Normalized 3 date formats to YYYY-MM-DD", "🔤 Merged `' north '` / `'North'`".
- **Before/After** preview as two side-by-side `st.dataframe` tables (raw vs. clean).
- **Anomaly detection chart**: Plotly daily-revenue line with **red circular markers** on 3 anomalous days, and a legend entry "IsolationForest + rolling Z-score".
- **Correlation heatmap**: indigo→teal diverging scale on a dark background.

### Screen 3 — Empty / Onboarding state (Tab 1, before any question)
- Centered hero: 🧞 icon, H1 "What do you want to know about your data?", a muted subline "Load a dataset, then ask in plain English."
- If no data is loaded: a prominent primary button **"⚡ Load demo dataset"**.
- Once data is loaded: a success chip "Demo loaded · 650 rows" and **3 starter question chips**:
  - ✨ "Why did revenue drop in the North region last month?"
  - ✨ "Which product category has the highest profit margin?"
  - ✨ "Show monthly revenue trend by region."
- Chat input is shown at the bottom.

### Screen 4 — Loading state (agent running)
- The user bubble is shown, followed by an assistant card containing an `st.status`-style progress panel with 4 rows:
  1. ✍️ SQL Writer: *Generating query…* (running spinner, indigo)
  2. 🛡️ SQL Critic: *Validating & executing…* (pending)
  3. 📊 Stats & Chart: *Computing statistics* (pending)
  4. 🧠 Insight Analyst: *Writing summary* (pending)
- An amber "Retry 1/3: fixing SQL" chip appears if the critic loops back.
- Skeleton shimmer placeholders where the chart and the summary will appear.

### Screen 5 — Connect Supabase / PostgreSQL form
- Sidebar Data Source set to `🔌 Connect…`. The main area shows a centered card, "Connect your database":
  - Text input (password style): `postgresql://user:password@host:5432/dbname`, with helper caption "Supabase: Project Settings → Database → Connection string".
  - Info note: "Queries are read-only. DROP/DELETE/UPDATE are blocked."
  - Primary button **Connect**.
- States: connecting (spinner), success (teal chip "Connected · 4 tables found" with a schema preview), error (red inline alert "Could not connect: check host & password").

---

## 6. Chart Styling (Plotly)
- Template: dark; transparent paper/plot background; grid lines `#2A3560` at 40% opacity.
- Font Inter 12 `#E6E9F5`; title left-aligned; hover tooltips with `bg-elevated`.
- Positive values teal, negative values red; anomalies = red circles, size 10, with white outline.
- Legend horizontal above the chart. Rounded card container around every chart.

---

## 7. Do / Don't
**Do:** keep generous whitespace, use one primary action per view, use emoji icons consistently, keep copy short and business-friendly, and show realistic data.
**Don't:** use pure black or pure white, add more than 2 tabs, show raw SQL by default, invent features beyond those listed (auth, billing, dashboards), or use light-mode surfaces.

---

## 8. UI Simplicity & UX Guardrails
- **Calm, uncluttered layout**: Do **not** follow complex multi-panel Stitch mockups or dense dashboard walls. Every view should feel immediately approachable and effortless for a first-time visitor.
- **Auto-ready zero-friction start**: Load the demo dataset automatically on first launch so users can click a starter question or type in the chat box immediately without setup steps.
- **Progressive disclosure**:
  - Show **Executive Answer + 1 Interactive Chart** cleanly in the main chat flow.
  - Keep SQL queries, raw result tables, Python statistical checks, and agent step traces tucked inside a single collapsed `🔍 Under the Hood` expander.
  - In **Data Health & Auto-EDA**, lead with 4 clear summary metrics and clean visuals, keeping raw/cleaned table comparisons and detailed logs neatly organized inside expanders/tabs so the page never feels overwhelming.


