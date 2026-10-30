"""
VoxShield AI - Database Module
SQLite database for persistent analysis history and live system metrics.
Ghost Protocol | AI & Cyber Security
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from config import DB_PATH, LABEL_REAL, LABEL_FAKE


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns an active SQLite database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the SQLite database schema if not already present."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration REAL NOT NULL,
                detection_mode TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                model_name TEXT,
                feature_summary_json TEXT,
                sample_rate INTEGER,
                channels INTEGER
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_history(timestamp DESC)
        """)
        conn.commit()


def save_analysis_record(
    filename: str,
    duration: float,
    detection_mode: str,
    prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    model_name: Optional[str] = "None (Demo Mode)",
    feature_summary: Optional[Dict[str, Any]] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    custom_id: Optional[str] = None
) -> str:
    """
    Inserts a newly completed audio analysis record into the database.
    Note: Audio binary content is NEVER stored to respect data privacy and system integrity.
    """
    init_db()
    record_id = custom_id or f"VX-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feature_json = json.dumps(feature_summary) if feature_summary else None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_history (
                id, filename, timestamp, duration, detection_mode,
                prediction, confidence, model_name, feature_summary_json,
                sample_rate, channels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id, filename, timestamp, duration, detection_mode,
            prediction, confidence, model_name, feature_json,
            sample_rate, channels
        ))
        conn.commit()

    return record_id


def get_all_records(limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieves all analysis history records ordered by recency."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_records_dataframe(
    search_query: Optional[str] = None,
    mode_filter: Optional[str] = None,
    prediction_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Returns filtered analysis history as a pandas DataFrame for UI tables and CSV export.
    """
    init_db()
    query = "SELECT id, filename, timestamp, duration, detection_mode, prediction, confidence, model_name, sample_rate, channels FROM analysis_history WHERE 1=1"
    params: List[Any] = []

    if search_query and search_query.strip():
        query += " AND (filename LIKE ? OR id LIKE ? OR model_name LIKE ?)"
        term = f"%{search_query.strip()}%"
        params.extend([term, term, term])

    if mode_filter and mode_filter != "All":
        query += " AND detection_mode = ?"
        params.append(mode_filter)

    if prediction_filter and prediction_filter != "All":
        if prediction_filter == "None / Unavailable":
            query += " AND (prediction IS NULL OR prediction = '' OR prediction LIKE '%Unavailable%')"
        else:
            query += " AND prediction = ?"
            params.append(prediction_filter)

    query += " ORDER BY timestamp DESC"

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


def get_record_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Fetches full details including feature summary for a specific analysis record."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)
        if data.get("feature_summary_json"):
            try:
                data["feature_summary"] = json.loads(data["feature_summary_json"])
            except Exception:
                data["feature_summary"] = None
        else:
            data["feature_summary"] = None
        return data


def delete_record(record_id: str) -> bool:
    """Deletes a single analysis record by ID."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_history() -> int:
    """Deletes all records from the database. Returns number of rows deleted."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM analysis_history")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM analysis_history")
        conn.commit()
        return count


def get_system_stats() -> Dict[str, int]:
    """
    Computes real, un-fabricated system metrics calculated strictly from the database.
    Counters start at zero if no analyses have been performed.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM analysis_history")
        total_files = cursor.fetchone()[0] or 0

        cursor.execute("SELECT count(*) FROM analysis_history WHERE prediction = ?", (LABEL_REAL,))
        real_count = cursor.fetchone()[0] or 0

        cursor.execute("SELECT count(*) FROM analysis_history WHERE prediction = ?", (LABEL_FAKE,))
        fake_count = cursor.fetchone()[0] or 0

        cursor.execute("SELECT count(*) FROM analysis_history WHERE prediction IS NULL OR prediction = '' OR prediction LIKE '%Unavailable%'")
        demo_count = cursor.fetchone()[0] or 0

    return {
        "total_analyzed": total_files,
        "real_count": real_count,
        "fake_count": fake_count,
        "demo_count": demo_count
    }
