"""
Unit tests for VoxShield AI - SQLite Database Module
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from modules import database
from config import MODE_DEMO, LABEL_REAL, LABEL_FAKE


@pytest.fixture(autouse=True)
def isolated_test_db(monkeypatch):
    """Directs all database operations to a temporary isolated SQLite file during testing."""
    import gc
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test_db_path = Path(tmp.name) / "test_voxshield.db"
    monkeypatch.setattr(database, "DB_PATH", test_db_path)
    database.init_db()
    try:
        yield test_db_path
    finally:
        gc.collect()
        try:
            tmp.cleanup()
        except Exception:
            pass


def test_initial_db_stats_are_zero():
    """
    CRITICAL INTEGRITY TEST:
    Counters must start at zero and never show fabricated statistics.
    """
    stats = database.get_system_stats()
    assert stats["total_analyzed"] == 0
    assert stats["real_count"] == 0
    assert stats["fake_count"] == 0
    assert stats["demo_count"] == 0


def test_insert_and_retrieve_record():
    """Verifies record creation, column storage, and retrieval."""
    rec_id = database.save_analysis_record(
        filename="test_sample.wav",
        duration=3.5,
        detection_mode=MODE_DEMO,
        prediction=None,
        confidence=None,
        model_name="None (Demo Mode)",
        feature_summary={"spectral_centroid": {"mean": 1500.0}},
        sample_rate=16000,
        channels=1
    )

    assert rec_id.startswith("VX-")

    # Verify stats incremented correctly
    stats = database.get_system_stats()
    assert stats["total_analyzed"] == 1
    assert stats["demo_count"] == 1
    assert stats["real_count"] == 0

    # Retrieve record by ID
    rec = database.get_record_by_id(rec_id)
    assert rec is not None
    assert rec["filename"] == "test_sample.wav"
    assert rec["feature_summary"]["spectral_centroid"]["mean"] == 1500.0


def test_search_and_filter():
    """Verifies query filtering by keyword, mode, and prediction."""
    database.save_analysis_record(
        filename="alice_voice.wav",
        duration=2.0,
        detection_mode="Model-Based Detection",
        prediction=LABEL_REAL,
        confidence=0.92
    )
    database.save_analysis_record(
        filename="bob_clone.wav",
        duration=4.1,
        detection_mode="Model-Based Detection",
        prediction=LABEL_FAKE,
        confidence=0.88
    )

    # Search by filename
    df_search = database.get_records_dataframe(search_query="alice")
    assert len(df_search) == 1
    assert df_search.iloc[0]["filename"] == "alice_voice.wav"

    # Filter by prediction
    df_fake = database.get_records_dataframe(prediction_filter=LABEL_FAKE)
    assert len(df_fake) == 1
    assert df_fake.iloc[0]["prediction"] == LABEL_FAKE


def test_delete_and_clear_history():
    """Verifies single record deletion and complete history clear."""
    id1 = database.save_analysis_record(filename="file1.wav", duration=1.0, detection_mode=MODE_DEMO)
    id2 = database.save_analysis_record(filename="file2.wav", duration=2.0, detection_mode=MODE_DEMO)

    assert database.get_system_stats()["total_analyzed"] == 2

    # Delete single
    deleted = database.delete_record(id1)
    assert deleted is True
    assert database.get_system_stats()["total_analyzed"] == 1

    # Clear all
    count = database.clear_all_history()
    assert count == 1
    assert database.get_system_stats()["total_analyzed"] == 0
