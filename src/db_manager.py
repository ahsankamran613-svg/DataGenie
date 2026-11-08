"""Unified SQLAlchemy 2.0 Database Layer for Local SQLite & Supabase PostgreSQL.

Features:
- Session-isolated SQLite database creation from cleaned DataFrames (`from_dataframe`)
- External Supabase / PostgreSQL connection with automatic `psycopg` v3 URL normalization
  (`postgres://` / `postgresql://` -> `postgresql+psycopg://`)
- Rich schema introspection (`get_schema_metadata` for UI badges, `get_schema_context` for LLM SQL generation)
- Strict read-only SQL validation & execution (`run_safe_query`)
"""

from __future__ import annotations

from pathlib import Path
import re
import tempfile
from typing import Any
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


class UnsafeQueryError(ValueError):
    """Raised when an LLM or user query attempts non-SELECT or destructive SQL."""


FORBIDDEN_SQL_KEYWORDS = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "VACUUM",
    "COPY",
    "CALL",
    "EXEC",
    "EXECUTE",
    "MERGE",
)

_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_SQL_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def normalize_database_url(db_url: str) -> str:
    """Normalize PostgreSQL/Supabase URLs to use the modern `psycopg` v3 driver."""
    cleaned = db_url.strip()
    if not cleaned:
        raise ValueError("Database connection URL cannot be empty.")

    if cleaned.startswith("postgres://"):
        return "postgresql+psycopg://" + cleaned[len("postgres://") :]
    if cleaned.startswith("postgresql://"):
        return "postgresql+psycopg://" + cleaned[len("postgresql://") :]
    return cleaned


def sanitize_identifier(name: str, fallback: str = "sales_data") -> str:
    """Ensure a SQL table/identifier is strictly alphanumeric + underscores."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip()).strip("_").lower()
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"tbl_{cleaned}"
    return cleaned


def _strip_sql_comments_and_fences(sql: str) -> str:
    """Strip markdown code blocks and SQL comments before AST/keyword validation."""
    s = sql.strip()
    # Remove ```sql ... ``` fences if present
    s = re.sub(r"^```(?:sql|postgresql|sqlite)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # Remove /* ... */ block comments and -- line comments
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = re.sub(r"--[^\r\n]*", " ", s)
    return s.strip().rstrip(";").strip()


def validate_readonly_sql(sql: str) -> str:
    """Validate that `sql` is a single read-only SELECT / WITH query and return clean SQL."""
    cleaned = _strip_sql_comments_and_fences(sql)
    if not cleaned:
        raise UnsafeQueryError("SQL query cannot be empty.")

    # Block stacked multi-statement SQL injection (any internal semicolon outside literals)
    without_single_quotes = re.sub(r"'([^']|'')*'", "''", cleaned)
    if ";" in without_single_quotes:
        raise UnsafeQueryError("Multiple SQL statements are not permitted in read-only mode.")

    if not re.match(r"^(SELECT|WITH)\b", without_single_quotes, flags=re.IGNORECASE):
        raise UnsafeQueryError("Only read-only SELECT (or WITH ... SELECT) queries are permitted.")

    match = _FORBIDDEN_PATTERN.search(without_single_quotes)
    if match:
        keyword = match.group(1).upper()
        raise UnsafeQueryError(
            f"Blocked unsafe SQL keyword `{keyword}`. DataGenie enforces strict read-only execution."
        )

    return cleaned


def _badge_type(col_name: str, raw_type: str) -> str:
    """Map SQLAlchemy column type strings to clean UI badge labels (`DATE`, `TEXT`, `REAL`, `INT`)."""
    upper = raw_type.upper()
    if "DATE" in upper or "TIME" in upper or "date" in col_name.lower():
        return "DATE"
    if any(tok in upper for tok in ("INT", "BIGINT", "SMALLINT", "SERIAL")):
        return "INT"
    if any(tok in upper for tok in ("FLOAT", "REAL", "DOUBLE", "NUMERIC", "DECIMAL")):
        return "REAL"
    if "BOOL" in upper:
        return "BOOL"
    return "TEXT"


class DatabaseManager:
    """Unified read-only SQL execution & schema introspection manager."""

    def __init__(
        self,
        engine: Engine,
        source_label: str = "Local SQLite",
        temp_db_path: Path | None = None,
    ) -> None:
        self.engine = engine
        self.source_label = source_label
        self._temp_db_path = temp_db_path

    @property
    def dialect(self) -> str:
        """Return SQLAlchemy dialect name (e.g., `'sqlite'` or `'postgresql'`)."""
        return str(self.engine.dialect.name)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        table_name: str = "sales_data",
        db_path: str | Path | None = None,
        source_label: str = "Local SQLite",
    ) -> "DatabaseManager":
        """Load a cleaned DataFrame into an isolated local SQLite database."""
        safe_table = sanitize_identifier(table_name)
        owned_temp_path: Path | None = None

        if db_path is None:
            tmp = tempfile.NamedTemporaryFile(prefix="datagenie_", suffix=".db", delete=False)
            tmp.close()
            target_path = Path(tmp.name)
            owned_temp_path = target_path
        else:
            target_path = Path(db_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

        sqlite_url = f"sqlite:///{target_path.resolve().as_posix()}"
        engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

        with engine.begin() as conn:
            df.to_sql(safe_table, conn, if_exists="replace", index=False)

        return cls(engine=engine, source_label=source_label, temp_db_path=owned_temp_path)

    @classmethod
    def from_connection_url(
        cls,
        db_url: str,
        connect_timeout: int = 10,
        source_label: str = "PostgreSQL / Supabase",
    ) -> "DatabaseManager":
        """Connect to an external PostgreSQL / Supabase (or SQLite) database and verify access."""
        normalized_url = normalize_database_url(db_url)
        connect_args: dict[str, Any] = {}
        if normalized_url.startswith("postgresql"):
            connect_args["connect_timeout"] = connect_timeout
        elif normalized_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        engine = create_engine(
            normalized_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        # Verify live connectivity upfront
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return cls(engine=engine, source_label=source_label)

    def get_table_names(self) -> list[str]:
        """List user tables in the connected database."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        return [t for t in tables if not t.startswith("sqlite_")]

    def get_schema_metadata(self, max_tables: int = 10) -> list[dict[str, Any]]:
        """Return structured table/column metadata for the Sidebar Schema Viewer."""
        inspector = inspect(self.engine)
        tables = self.get_table_names()[:max_tables]
        metadata: list[dict[str, Any]] = []

        with self.engine.connect() as conn:
            for tbl in tables:
                safe_tbl = f'"{tbl}"'
                try:
                    row_count = int(
                        conn.execute(text(f"SELECT COUNT(*) FROM {safe_tbl}")).scalar() or 0
                    )
                except Exception:
                    row_count = 0

                cols_info = []
                for col in inspector.get_columns(tbl):
                    col_name = str(col["name"])
                    raw_type = str(col["type"])
                    cols_info.append(
                        {
                            "name": col_name,
                            "raw_type": raw_type,
                            "badge": _badge_type(col_name, raw_type),
                        }
                    )

                metadata.append(
                    {
                        "table_name": tbl,
                        "row_count": row_count,
                        "columns": cols_info,
                    }
                )

        return metadata

    def get_schema_context(self, max_tables: int = 8, sample_rows: int = 3) -> str:
        """Build rich LLM schema context including date bounds, category values, and sample rows."""
        inspector = inspect(self.engine)
        tables = self.get_table_names()[:max_tables]
        if not tables:
            return f"SQL Dialect: {self.dialect}\nNo tables found in database."

        sections: list[str] = [f"SQL Dialect: {self.dialect}"]

        with self.engine.connect() as conn:
            for tbl in tables:
                safe_tbl = f'"{tbl}"'
                try:
                    row_count = int(
                        conn.execute(text(f"SELECT COUNT(*) FROM {safe_tbl}")).scalar() or 0
                    )
                except Exception:
                    row_count = 0

                columns = inspector.get_columns(tbl)
                col_lines: list[str] = []

                for col in columns:
                    c_name = str(col["name"])
                    c_type = str(col["type"])
                    badge = _badge_type(c_name, c_type)
                    safe_col = f'"{c_name}"'
                    extra = ""

                    try:
                        if badge == "DATE":
                            bounds = conn.execute(
                                text(f"SELECT MIN({safe_col}), MAX({safe_col}) FROM {safe_tbl}")
                            ).fetchone()
                            if bounds and bounds[0] is not None:
                                extra = f" [date range: '{bounds[0]}' to '{bounds[1]}']"
                        elif badge == "TEXT" and not c_name.endswith("_id"):
                            distinct_count = int(
                                conn.execute(
                                    text(f"SELECT COUNT(DISTINCT {safe_col}) FROM {safe_tbl}")
                                ).scalar()
                                or 0
                            )
                            if 1 <= distinct_count <= 15:
                                vals_res = conn.execute(
                                    text(
                                        f"SELECT DISTINCT {safe_col} FROM {safe_tbl} "
                                        f"WHERE {safe_col} IS NOT NULL ORDER BY {safe_col} LIMIT 15"
                                    )
                                ).fetchall()
                                vals = [repr(str(r[0])) for r in vals_res]
                                extra = f" [distinct values: {', '.join(vals)}]"
                    except Exception:
                        extra = ""

                    col_lines.append(f"  - {c_name} ({c_type} / {badge}){extra}")

                sample_str = ""
                if row_count > 0 and sample_rows > 0:
                    try:
                        sample_df = pd.read_sql(
                            text(f"SELECT * FROM {safe_tbl} LIMIT {int(sample_rows)}"),
                            conn,
                        )
                        sample_str = f"\nSample rows ({len(sample_df)}):\n{sample_df.to_string(index=False)}"
                    except Exception:
                        sample_str = ""

                sections.append(
                    f"Table: {tbl} (Total rows: {row_count})\nColumns:\n"
                    + "\n".join(col_lines)
                    + sample_str
                )

        return "\n\n".join(sections)

    def run_safe_query(self, sql: str) -> pd.DataFrame:
        """Validate read-only safety and execute `sql`, returning results as a DataFrame."""
        clean_sql = validate_readonly_sql(sql)
        with self.engine.connect() as conn:
            return pd.read_sql(text(clean_sql), conn)

    def close(self) -> None:
        """Dispose engine connections and delete owned temporary SQLite file if applicable."""
        self.engine.dispose()
        if self._temp_db_path is not None and self._temp_db_path.exists():
            try:
                self._temp_db_path.unlink()
            except OSError:
                pass
