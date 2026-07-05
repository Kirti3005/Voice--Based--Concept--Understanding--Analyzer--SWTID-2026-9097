# VBCUA/tests/test_database_manager.py
"""
Pytest tests for backend.database_manager module.
Uses a temporary SQLite database to avoid touching the real database.db.
"""
import os
import sqlite3
import pytest
import tempfile

# ── Patch DB_PATH before importing the module ──────────────────────────────────
import backend.database_manager as db_mod


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirects the database manager to use a fresh temp database for each test.
    Automatically cleans up after each test.
    """
    test_db_path = str(tmp_path / "test_vbcua.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db_path)
    yield test_db_path


def _sample_session_data():
    return dict(
        concept="Machine Learning",
        language="en",
        final_score=78.5,
        understanding_level="Strong Understanding",
        semantic_score=0.75,
        coverage_score=0.60,
        transcript="Machine learning is a subset of AI that uses data.",
        feedback="Good explanation with strong coverage.",
        audio_features={
            "tempo": 112.0,
            "energy": 0.04,
            "duration": 25.0,
            "pause_ratio": 0.15,
            "pause_count": 3,
            "longest_pause": 0.8,
            "speaking_duration": 21.0,
        },
        filler_data={
            "total_count": 2,
            "frequency_pct": 1.5,
            "breakdown": {"um": 1, "uh": 1},
        },
        metrics={
            "semantic": 75.0,
            "coverage": 60.0,
            "fluency": 88.0,
            "confidence": 85.0,
            "pause": 92.0,
            "filler": 88.0,
            "communication": 84.0,
            "quality": 92.0,
        },
    )


class TestInitDb:

    def test_creates_database_file(self, temp_db):
        db_mod.init_db()
        assert os.path.exists(temp_db)

    def test_creates_all_tables(self, temp_db):
        db_mod.init_db()
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"sessions", "evaluations", "audio_features", "filler_words", "metrics_breakdown"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_idempotent_multiple_calls(self, temp_db):
        """Calling init_db multiple times must not raise."""
        db_mod.init_db()
        db_mod.init_db()
        db_mod.init_db()
        assert os.path.exists(temp_db)


class TestSaveSession:

    def test_returns_positive_integer_id(self):
        data = _sample_session_data()
        session_id = db_mod.save_session(**data)
        assert isinstance(session_id, int)
        assert session_id > 0

    def test_increments_session_id(self):
        data = _sample_session_data()
        id1 = db_mod.save_session(**data)
        id2 = db_mod.save_session(**data)
        assert id2 > id1

    def test_session_persists_in_db(self, temp_db):
        data = _sample_session_data()
        session_id = db_mod.save_session(**data)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        assert row is not None

    def test_correct_concept_stored(self):
        data = _sample_session_data()
        session_id = db_mod.save_session(**data)
        result = db_mod.get_session_by_id(session_id)
        assert result["concept"] == "Machine Learning"

    def test_correct_final_score_stored(self):
        data = _sample_session_data()
        session_id = db_mod.save_session(**data)
        result = db_mod.get_session_by_id(session_id)
        assert abs(result["final_score"] - 78.5) < 0.01


class TestGetAllSessions:

    def test_returns_list(self):
        sessions = db_mod.get_all_sessions()
        assert isinstance(sessions, list)

    def test_empty_db_returns_empty_list(self):
        db_mod.init_db()
        sessions = db_mod.get_all_sessions()
        assert sessions == []

    def test_returns_sessions_after_save(self):
        db_mod.save_session(**_sample_session_data())
        sessions = db_mod.get_all_sessions()
        assert len(sessions) >= 1

    def test_limit_respected(self):
        data = _sample_session_data()
        for _ in range(8):
            db_mod.save_session(**data)
        sessions = db_mod.get_all_sessions(limit=5)
        assert len(sessions) <= 5

    def test_ordered_newest_first(self):
        data = _sample_session_data()
        id1 = db_mod.save_session(**data)
        id2 = db_mod.save_session(**data)
        sessions = db_mod.get_all_sessions()
        assert sessions[0]["id"] > sessions[-1]["id"]


class TestGetSessionById:

    def test_returns_none_for_nonexistent_id(self):
        result = db_mod.get_session_by_id(99999)
        assert result is None

    def test_returns_dict_for_existing_id(self):
        sid = db_mod.save_session(**_sample_session_data())
        result = db_mod.get_session_by_id(sid)
        assert isinstance(result, dict)

    def test_contains_evaluation_sub_dict(self):
        sid = db_mod.save_session(**_sample_session_data())
        result = db_mod.get_session_by_id(sid)
        assert "evaluation" in result

    def test_contains_audio_features_sub_dict(self):
        sid = db_mod.save_session(**_sample_session_data())
        result = db_mod.get_session_by_id(sid)
        assert "audio_features" in result

    def test_filler_breakdown_is_dict(self):
        sid = db_mod.save_session(**_sample_session_data())
        result = db_mod.get_session_by_id(sid)
        filler = result.get("filler_words", {})
        assert isinstance(filler.get("breakdown", {}), dict)


class TestGetSessionCount:

    def test_zero_on_empty_db(self):
        db_mod.init_db()
        assert db_mod.get_session_count() == 0

    def test_increments_after_save(self):
        db_mod.save_session(**_sample_session_data())
        assert db_mod.get_session_count() == 1
        db_mod.save_session(**_sample_session_data())
        assert db_mod.get_session_count() == 2
