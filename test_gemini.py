# VBCUA/tests/test_gemini.py
import os
import pytest
from unittest.mock import MagicMock, patch
from backend.gemini_feedback import get_qualitative_feedback, get_static_feedback, generate_gemini_feedback


def test_static_feedback_strong():
    """Verify strong understanding static feedback."""
    fb = get_static_feedback("Strong Understanding", 85.5)
    assert "Excellent" in fb
    assert "85.5/100" in fb


def test_static_feedback_moderate():
    """Verify moderate understanding static feedback."""
    fb = get_static_feedback("Moderate Understanding", 65.0)
    assert "Good effort" in fb
    assert "65.0/100" in fb


def test_static_feedback_poor():
    """Verify poor understanding static feedback."""
    fb = get_static_feedback("Poor Understanding", 32.1)
    assert "needs improvement" in fb
    assert "32.1/100" in fb


def test_fallback_without_key():
    """Verify that omitting key triggers fallback to static feedback."""
    with patch.dict(os.environ, {}, clear=True):
        fb = get_qualitative_feedback(
            user_transcript="Test transcript",
            reference_concept="Test reference",
            semantic_score=0.75,
            metrics={"final_score": 85.5},
            filler_data={"total_count": 0, "frequency_pct": 0.0},
            audio_features={"tempo": 120.0, "pause_ratio": 0.15},
            api_key=None
        )
        assert "Excellent" in fb  # Should fall back to static strong feedback


@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure")
def test_gemini_api_call(mock_configure, mock_gen_model):
    """Verify that providing key calls Gemini API and returns generated text."""
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a mocked Gemini response feedback."
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_instance

    fb = get_qualitative_feedback(
        user_transcript="Test transcript",
        reference_concept="Test reference",
        semantic_score=0.75,
        metrics={"final_score": 85.5},
        filler_data={"total_count": 0, "frequency_pct": 0.0},
        audio_features={"tempo": 120.0, "pause_ratio": 0.15},
        api_key="mocked_key"
    )

    mock_configure.assert_called_once_with(api_key="mocked_key")
    mock_model_instance.generate_content.assert_called_once()
    assert fb == "This is a mocked Gemini response feedback."
