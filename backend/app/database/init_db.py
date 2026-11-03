"""Database initialization, healthcheck, and schema provisioning CLI.

Supports PostgreSQL and SQLite engines via SQLAlchemy 2.0 metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from backend.app.database.base import Base
import backend.app.database.models  # noqa: F401 (ensure all models registered)
from backend.app.database.session import build_engine, engine as default_engine, get_database_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_connection(target_engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Test connectivity to the database.

    Args:
        target_engine: Optional SQLAlchemy engine. Defaults to configured engine.

    Returns:
        Status dict with connectivity, dialect, latency_ms, or error details.
    """
    eng = target_engine or default_engine
    t0 = time.perf_counter()
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "connected",
            "dialect": eng.dialect.name,
            "driver": eng.driver,
            "url_masked": eng.url.render_as_string(hide_password=True),
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }
    except SQLAlchemyError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "unreachable",
            "dialect": eng.dialect.name,
            "driver": eng.driver,
            "url_masked": eng.url.render_as_string(hide_password=True),
            "latency_ms": round(latency_ms, 2),
            "error": str(exc),
        }


def get_table_status(target_engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Inspect existing tables, column counts, and index counts.

    Args:
        target_engine: Optional SQLAlchemy engine. Defaults to configured engine.

    Returns:
        Dict detailing existing tables and metadata.
    """
    eng = target_engine or default_engine
    inspector = inspect(eng)
    existing_tables = inspector.get_table_names()
    registered_tables = list(Base.metadata.tables.keys())

    table_details = {}
    for tbl in existing_tables:
        cols = inspector.get_columns(tbl)
        indexes = inspector.get_indexes(tbl)
        table_details[tbl] = {
            "columns_count": len(cols),
            "indexes_count": len(indexes),
        }

    return {
        "existing_tables": existing_tables,
        "registered_models": registered_tables,
        "is_complete": set(registered_tables).issubset(set(existing_tables)),
        "table_details": table_details,
    }


def create_database_tables(target_engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Create all registered database tables in the target database.

    Args:
        target_engine: Optional SQLAlchemy engine. Defaults to configured engine.

    Returns:
        Summary dict containing created tables and status.
    """
    eng = target_engine or default_engine
    conn_check = check_connection(eng)
    if conn_check["status"] != "connected":
        raise ConnectionError(
            f"Cannot initialize database. Target engine is unreachable: {conn_check['error']}"
        )

    Base.metadata.create_all(bind=eng)
    status = get_table_status(eng)

    return {
        "status": "success",
        "dialect": eng.dialect.name,
        "created_tables": status["existing_tables"],
        "table_count": len(status["existing_tables"]),
    }


def drop_database_tables(
    target_engine: Optional[Engine] = None, confirm: bool = False
) -> Dict[str, Any]:
    """Drop all registered database tables (destructive operation).

    Args:
        target_engine: Optional SQLAlchemy engine.
        confirm: Safety confirmation boolean. Must be True.

    Returns:
        Summary dict of dropped tables.
    """
    if not confirm:
        raise ValueError("Cannot drop tables without explicit confirm=True.")

    eng = target_engine or default_engine
    Base.metadata.drop_all(bind=eng)
    return {
        "status": "dropped",
        "dialect": eng.dialect.name,
        "remaining_tables": inspect(eng).get_table_names(),
    }


def main() -> int:
    """CLI Entrypoint for database management."""
    parser = argparse.ArgumentParser(
        description="OpenSourceLens Database Management & Provisioning CLI"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom database connection URL (defaults to env DATABASE_URL)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connectivity to the database",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create all tables and indexes defined in SQLAlchemy models",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Inspect existing database tables and models",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all registered tables (requires --yes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive drop operation",
    )

    args = parser.parse_args()
    target_eng = build_engine(args.url) if args.url else default_engine

    try:
        if not any([args.check, args.create, args.status, args.drop]):
            # Default behavior: check connection and status
            args.check = True
            args.status = True

        if args.check:
            res = check_connection(target_eng)
            print("\n--- Database Connectivity Check ---")
            print(json.dumps(res, indent=2))
            if res["status"] != "connected":
                return 1

        if args.drop:
            if not args.yes:
                print("Error: --drop requires --yes confirmation.", file=sys.stderr)
                return 1
            res = drop_database_tables(target_eng, confirm=True)
            print("\n--- Dropped Database Tables ---")
            print(json.dumps(res, indent=2))

        if args.create:
            res = create_database_tables(target_eng)
            print("\n--- Created Database Tables ---")
            print(json.dumps(res, indent=2))

        if args.status:
            res = get_table_status(target_eng)
            print("\n--- Database Table Status ---")
            print(json.dumps(res, indent=2))

        return 0
    finally:
        target_eng.dispose()


if __name__ == "__main__":
    sys.exit(main())
