# VBCUA/tests/test_audio_analysis.py
"""
Pytest tests for backend.audio_analysis module.
Uses the sample.wav file that ships with the project.
"""
import os
import pytest
from backend.audio_analysis import get_audio_features

# Locate sample.wav relative to project root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_WAV = os.path.join(_ROOT, "sample.wav")


class TestGetAudioFeatures:

    def test_returns_dict(self):
        """get_audio_features must always return a dict."""
        result = get_audio_features(SAMPLE_WAV)
        assert isinstance(result, dict)

    def test_all_required_keys_present(self):
        """All required keys must be present in the output."""
        required_keys = {
            "tempo", "energy", "duration",
            "pause_ratio", "pause_count", "longest_pause",
            "zcr", "speaking_duration"
        }
        result = get_audio_features(SAMPLE_WAV)
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_tempo_is_positive(self):
        """Tempo must be a positive float."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["tempo"] > 0

    def test_duration_is_positive(self):
        """Duration must be positive."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["duration"] > 0

    def test_energy_is_non_negative(self):
        """RMS energy must be >= 0."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["energy"] >= 0

    def test_pause_ratio_in_range(self):
        """Pause ratio must be between 0 and 1."""
        result = get_audio_features(SAMPLE_WAV)
        assert 0.0 <= result["pause_ratio"] <= 1.0

    def test_pause_count_non_negative(self):
        """Pause count must be >= 0."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["pause_count"] >= 0

    def test_longest_pause_non_negative(self):
        """Longest pause must be >= 0."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["longest_pause"] >= 0.0

    def test_speaking_duration_lte_duration(self):
        """Speaking duration must not exceed total duration."""
        result = get_audio_features(SAMPLE_WAV)
        assert result["speaking_duration"] <= result["duration"] + 0.1  # small tolerance

    def test_fallback_on_invalid_path(self):
        """Must return fallback defaults on a non-existent file path."""
        result = get_audio_features("non_existent_file_xyz.wav")
        assert isinstance(result, dict)
        assert "tempo" in result
        assert "energy" in result
        assert "duration" in result
        assert "pause_ratio" in result

    def test_fallback_values_are_valid(self):
        """Fallback values must be valid numeric types, not None."""
        result = get_audio_features("non_existent_file_xyz.wav")
        for key in ("tempo", "energy", "duration", "pause_ratio"):
            assert result[key] is not None
            assert isinstance(result[key], (int, float))
