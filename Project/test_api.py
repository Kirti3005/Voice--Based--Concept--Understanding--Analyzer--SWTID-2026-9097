# VBCUA/tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient
from backend.api import app
import backend.database_manager as db_mgr

client = TestClient(app)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_WAV = os.path.join(_ROOT, "sample.wav")


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirects the database manager to use a fresh temp database for each test.
    Automatically cleans up after each test.
    """
    test_db_path = str(tmp_path / "test_api_vbcua.db")
    monkeypatch.setattr(db_mgr, "DB_PATH", test_db_path)
    db_mgr.init_db()
    yield test_db_path


def test_health_check():
    """Verify that health endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_get_concepts():
    """Verify that preset concepts are loaded and returned."""
    response = client.get("/api/concepts")
    assert response.status_code == 200
    data = response.json()
    assert "concepts" in data
    assert "data" in data
    assert "Machine Learning" in data["concepts"]


def test_analyze_endpoint():
    """Verify that the analyze pipeline endpoint completes successfully on sample audio."""
    assert os.path.exists(SAMPLE_WAV)
    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/api/analyze",
            files={"audio_file": ("sample.wav", f, "audio/wav")},
            data={"concept_name": "Machine Learning", "language_mode": "English"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "transcript" in data
    assert "semantic_score" in data
    assert "audio_features" in data
    assert "metrics" in data
    assert "waveform" in data


def test_save_and_retrieve_session():
    """Verify session storage, list retrieval, details retrieval, and PDF generation."""
    payload = {
        "concept": "Machine Learning",
        "language": "en",
        "final_score": 75.2,
        "understanding_level": "Strong Understanding",
        "semantic_score": 0.75,
        "coverage_score": 0.60,
        "transcript": "Machine learning is a subset of AI.",
        "feedback": "Great explanation.",
        "audio_features": {
            "tempo": 120.0,
            "energy": 0.05,
            "duration": 10.0,
            "pause_ratio": 0.15,
            "pause_count": 2,
            "longest_pause": 0.8,
            "speaking_duration": 8.5
        },
        "filler_data": {
            "total_count": 1,
            "frequency_pct": 2.0,
            "breakdown": {"um": 1}
        },
        "metrics": {
            "semantic": 75.0,
            "coverage": 60.0,
            "fluency": 85.0,
            "confidence": 80.0,
            "pause": 90.0,
            "filler": 85.0,
            "communication": 82.0,
            "quality": 92.0
        }
    }

    # Test save
    response = client.post("/api/sessions", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    session_id = res_data["session_id"]
    assert session_id > 0

    # Test list
    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions_data = response.json()
    assert len(sessions_data["sessions"]) >= 1
    assert sessions_data["sessions"][0]["id"] == session_id

    # Test retrieve by ID
    response = client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    session_details = response.json()
    assert session_details["concept"] == "Machine Learning"
    assert session_details["evaluation"]["transcript"] == "Machine learning is a subset of AI."

    # Test PDF generation
    response = client.get(f"/api/sessions/{session_id}/pdf")
    assert response.status_code == 200, f"Error: {response.status_code} - {response.text}"
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 0
