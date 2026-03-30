import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/digest.db")

def init_db(path: Path = DB_PATH):
    """
    Create the database file and tables if they don't exist.
    Returns an open connection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # Enable dictionary-like access to rows

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sent_items (
            url          TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            category     TEXT NOT NULL,
            sent_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id       TEXT PRIMARY KEY,
            started_at   TEXT NOT NULL,
            completed_at TEXT,
            status       TEXT,
            items_sent   INTEGER,
            errors       TEXT
        );
    """)

    conn.commit()
    return conn

def is_sent(conn: sqlite3.Connection, url: str) -> bool:
    """
    Return True if this URL has already been delivered in a previous run.
    """
    row = conn.execute(
        "SELECT 1 FROM sent_items WHERE url = ?", (url,)
    ).fetchone()
    return row is not None


def mark_sent(conn: sqlite3.Connection, articles: list[dict]) -> None:
    """
    Record a batch of articles as sent.
    Each dict must have: url, content_hash, category, sent_at.
    """
    conn.executemany(
        """
        INSERT OR IGNORE INTO sent_items (url, content_hash, category, sent_at)
        VALUES (:url, :content_hash, :category, :sent_at)
        """,
        articles,
    )
    conn.commit()