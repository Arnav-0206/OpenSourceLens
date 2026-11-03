"""Schema DDL generator for OpenSourceLens.

Generates ANSI SQL / PostgreSQL and SQLite DDL scripts directly from
SQLAlchemy metadata definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

from backend.app.database.base import Base
import backend.app.database.models  # noqa: F401 (ensure models registered)


def generate_ddl_for_dialect(dialect_name: str = "postgresql") -> str:
    """Generate DDL script string for a specific SQL dialect.

    Args:
        dialect_name: Dialect name ('postgresql' or 'sqlite').

    Returns:
        Formatted DDL string with table definitions and indexes.
    """
    if dialect_name == "postgresql":
        dialect = postgresql.dialect()
    elif dialect_name == "sqlite":
        dialect = sqlite.dialect()
    else:
        raise ValueError(f"Unsupported dialect: {dialect_name}")

    statements = []
    statements.append(f"-- ========================================================")
    statements.append(f"-- OpenSourceLens Database Schema ({dialect_name.upper()})")
    statements.append(f"-- Auto-generated from SQLAlchemy 2.0 declarative models")
    statements.append(f"-- ========================================================\n")

    # Generate CREATE TABLE statements in dependency order
    for table in Base.metadata.sorted_tables:
        create_table = str(
            CreateTable(table).compile(dialect=dialect)
        ).strip()
        statements.append(f"{create_table};\n")

        # Generate standalone indexes
        for index in table.indexes:
            create_index = str(
                CreateIndex(index).compile(dialect=dialect)
            ).strip()
            statements.append(f"{create_index};\n")

    return "\n".join(statements)


def export_schema_file(output_path: Optional[Path] = None) -> Path:
    """Export standard schema SQL file.

    Args:
        output_path: Target path. Defaults to backend/app/database/schema.sql.

    Returns:
        Path to written schema file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "schema.sql"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pg_ddl = generate_ddl_for_dialect("postgresql")
    output_path.write_text(pg_ddl, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = export_schema_file()
    print(f"Exported database schema DDL to: {out}")
