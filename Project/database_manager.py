# VBCUA/backend/database_manager.py
"""
SQLite database manager for VBCUA.
Stores and retrieves evaluation sessions, audio features, and filler word data.
Uses Python's built-in sqlite3 — no external dependency required.
"""
import sqlite3
import os
import json
from datetime import datetime

# ── Database path ──────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DB_PATH = os.path.join(_ROOT_DIR, "database", "database.db")


# ──────────────────────────────────────────────────────────────────────────────
# Schema Initialization
# ──────────────────────────────────────────────────────────────────────────────
def _table_has_column(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Returns True if the given table has the specified column."""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _drop_stale_tables(conn: sqlite3.Connection) -> bool:
    """
    Checks if the existing schema is compatible with the current definition.
    If any required column is missing, drops ALL tables so init_db() rebuilds them fresh.
    Returns True if tables were dropped (migration happened).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cursor.fetchall()}

    # Key integrity checks — if any fail, wipe and rebuild
    checks = [
        ("evaluations", "session_id"),
        ("audio_features", "session_id"),
        ("filler_words", "session_id"),
        ("metrics_breakdown", "session_id"),
    ]
    needs_migration = any(
        table in existing and not _table_has_column(cursor, table, col)
        for table, col in checks
    )

    if needs_migration:
        for table in ("metrics_breakdown", "filler_words", "audio_features", "evaluations", "sessions"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        return True
    return False


def init_db() -> None:
    """
    Creates all required tables if they don't exist.
    Auto-migrates stale schemas by dropping and recreating tables.
    Safe to call multiple times.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Auto-migrate if schema is stale
    _drop_stale_tables(conn)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            concept     TEXT,
            language    TEXT,
            final_score REAL,
            understanding_level TEXT
        );

        CREATE TABLE IF NOT EXISTS evaluations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      INTEGER NOT NULL,
            semantic_score  REAL,
            coverage_score  REAL,
            transcript      TEXT,
            feedback        TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS audio_features (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       INTEGER NOT NULL,
            tempo            REAL,
            energy           REAL,
            duration         REAL,
            pause_ratio      REAL,
            pause_count      INTEGER,
            longest_pause    REAL,
            speaking_duration REAL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS filler_words (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER NOT NULL,
            total_count   INTEGER,
            frequency_pct REAL,
            breakdown     TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS metrics_breakdown (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER NOT NULL,
            metrics_json TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)

    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Save a Session
# ──────────────────────────────────────────────────────────────────────────────
def save_session(
    concept: str,
    language: str,
    final_score: float,
    understanding_level: str,
    semantic_score: float,
    coverage_score: float,
    transcript: str,
    feedback: str,
    audio_features: dict,
    filler_data: dict,
    metrics: dict,
) -> int:
    """
    Saves a complete evaluation session to the database.
    Returns the new session ID.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Insert into sessions
    cursor.execute(
        "INSERT INTO sessions (timestamp, concept, language, final_score, understanding_level) "
        "VALUES (?, ?, ?, ?, ?)",
        (timestamp, concept, language, final_score, understanding_level)
    )
    session_id = cursor.lastrowid

    # 2. Insert evaluations
    cursor.execute(
        "INSERT INTO evaluations (session_id, semantic_score, coverage_score, transcript, feedback) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, semantic_score, coverage_score, transcript, feedback)
    )

    # 3. Insert audio features
    cursor.execute(
        "INSERT INTO audio_features "
        "(session_id, tempo, energy, duration, pause_ratio, pause_count, longest_pause, speaking_duration) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_id,
            audio_features.get("tempo", 0.0),
            audio_features.get("energy", 0.0),
            audio_features.get("duration", 0.0),
            audio_features.get("pause_ratio", 0.0),
            audio_features.get("pause_count", 0),
            audio_features.get("longest_pause", 0.0),
            audio_features.get("speaking_duration", 0.0),
        )
    )

    # 4. Insert filler words
    cursor.execute(
        "INSERT INTO filler_words (session_id, total_count, frequency_pct, breakdown) "
        "VALUES (?, ?, ?, ?)",
        (
            session_id,
            filler_data.get("total_count", 0),
            filler_data.get("frequency_pct", 0.0),
            json.dumps(filler_data.get("breakdown", {})),
        )
    )

    # 5. Insert metrics breakdown
    cursor.execute(
        "INSERT INTO metrics_breakdown (session_id, metrics_json) VALUES (?, ?)",
        (session_id, json.dumps(metrics))
    )

    conn.commit()
    conn.close()
    return session_id


# ──────────────────────────────────────────────────────────────────────────────
# Retrieve Sessions
# ──────────────────────────────────────────────────────────────────────────────
def get_all_sessions(limit: int = 20) -> list[dict]:
    """
    Returns the most recent `limit` sessions as a list of dicts.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, timestamp, concept, language, final_score, understanding_level "
        "FROM sessions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_session_by_id(session_id: int) -> dict | None:
    """
    Returns full details for a single session, joining all related tables.
    Returns None if not found.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session_row = cursor.fetchone()
    if not session_row:
        conn.close()
        return None

    result = dict(session_row)

    cursor.execute("SELECT * FROM evaluations WHERE session_id = ?", (session_id,))
    eval_row = cursor.fetchone()
    if eval_row:
        result["evaluation"] = dict(eval_row)

    cursor.execute("SELECT * FROM audio_features WHERE session_id = ?", (session_id,))
    audio_row = cursor.fetchone()
    if audio_row:
        result["audio_features"] = dict(audio_row)

    cursor.execute("SELECT * FROM filler_words WHERE session_id = ?", (session_id,))
    filler_row = cursor.fetchone()
    if filler_row:
        fd = dict(filler_row)
        fd["breakdown"] = json.loads(fd.get("breakdown", "{}"))
        result["filler_words"] = fd

    cursor.execute("SELECT * FROM metrics_breakdown WHERE session_id = ?", (session_id,))
    metrics_row = cursor.fetchone()
    if metrics_row:
        md = dict(metrics_row)
        md["metrics"] = json.loads(md.get("metrics_json", "{}"))
        result["metrics_breakdown"] = md

    conn.close()
    return result


def get_session_count() -> int:
    """Returns total number of sessions stored."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    count = cursor.fetchone()[0]
    conn.close()
    return count
