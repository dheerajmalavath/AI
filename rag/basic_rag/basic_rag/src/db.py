import sqlite3
import os
from datetime import datetime
from loguru import logger

# Resolve database path from env or default
DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/sqlite.db")
# Strip the sqlite:/// prefix to get the file path
DB_PATH = DB_URL.replace("sqlite:///", "")


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection with row_factory set."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS query_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type  TEXT    NOT NULL,   -- 'nutrition' or 'knowledge'
                query       TEXT    NOT NULL,
                response    TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        logger.info(f"Database initialised at {DB_PATH}")
    except Exception as e:
        logger.error(f"DB init error: {e}")


def save_query(query_type: str, query: str, response: str) -> None:
    """Persist a query + response to the history table."""
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO query_history (query_type, query, response, created_at) VALUES (?, ?, ?, ?)",
            (query_type, query, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB save error: {e}")


def get_history(limit: int = 20) -> list[dict]:
    """Return the most recent queries, newest first."""
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return []


def clear_history() -> None:
    """Delete all rows from query_history."""
    try:
        conn = get_connection()
        conn.execute("DELETE FROM query_history")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB clear error: {e}")


# Auto-init on import
init_db()
