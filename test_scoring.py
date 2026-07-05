# VBCUA/tests/test_scoring.py
"""
Pytest tests for backend.scoring module.
"""
import pytest
from backend.scoring import (
    calculate_final_score,
    calculate_communication_score,
    calculate_pause_score,
    get_understanding_level,
    build_metrics,
)


class TestCalculateFinalScore:

    def _perfect_metrics(self):
        return {
            "semantic": 100.0,
            "coverage": 100.0,
            "fluency": 100.0,
            "confidence": 100.0,
            "pause": 100.0,
            "filler": 100.0,
            "communication": 100.0,
            "quality": 100.0,
        }

    def _zero_metrics(self):
        return {
            "semantic": 0.0,
            "coverage": 0.0,
            "fluency": 0.0,
            "confidence": 0.0,
            "pause": 0.0,
            "filler": 0.0,
            "communication": 0.0,
            "quality": 0.0,
        }

    def test_perfect_metrics_return_100(self):
        score = calculate_final_score(self._perfect_metrics())
        assert score == pytest.approx(100.0, abs=0.1)

    def test_zero_metrics_return_0(self):
        score = calculate_final_score(self._zero_metrics())
        assert score == pytest.approx(0.0, abs=0.1)

    def test_score_always_between_0_and_100(self):
        for _ in range(5):
            import random
            metrics = {k: random.uniform(0, 100) for k in
                       ("semantic", "coverage", "fluency", "confidence",
                        "pause", "filler", "communication", "quality")}
            score = calculate_final_score(metrics)
            assert 0.0 <= score <= 100.0, f"Score out of range: {score}"

    def test_returns_float(self):
        score = calculate_final_score(self._perfect_metrics())
        assert isinstance(score, float)

    def test_missing_keys_still_produces_score(self):
        """Partial metrics dict should not raise — missing keys use 0."""
        score = calculate_final_score({"semantic": 80.0, "fluency": 70.0})
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_empty_dict_returns_zero(self):
        score = calculate_final_score({})
        assert score == 0.0

    def test_semantic_dominant_weight(self):
        """A high semantic score should push the final score up significantly."""
        low_all = calculate_final_score({k: 20.0 for k in
                                         ("semantic", "coverage", "fluency", "confidence",
                                          "pause", "filler", "communication", "quality")})
        high_semantic = calculate_final_score({
            "semantic": 100.0, "coverage": 20.0, "fluency": 20.0,
            "confidence": 20.0, "pause": 20.0, "filler": 20.0,
            "communication": 20.0, "quality": 20.0
        })
        assert high_semantic > low_all


class TestCalculateCommunicationScore:

    def test_returns_float_in_range(self):
        score = calculate_communication_score(80.0, 0.15, 5.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_high_filler_lowers_score(self):
        score_low_filler = calculate_communication_score(80.0, 0.10, 1.0)
        score_high_filler = calculate_communication_score(80.0, 0.10, 20.0)
        assert score_low_filler > score_high_filler

    def test_high_pause_ratio_lowers_score(self):
        score_low_pause = calculate_communication_score(80.0, 0.05, 2.0)
        score_high_pause = calculate_communication_score(80.0, 0.60, 2.0)
        assert score_low_pause > score_high_pause


class TestCalculatePauseScore:

    def test_ideal_ratio_returns_high_score(self):
        """Pause ratio of 0.20 is in the ideal window → high score."""
        score = calculate_pause_score(0.20, 5, 60.0)
        assert score >= 85.0

    def test_very_high_pause_ratio_returns_low_score(self):
        """Pause ratio of 0.80 (mostly silence) → low score."""
        score = calculate_pause_score(0.80, 30, 60.0)
        assert score < 50.0

    def test_returns_float_in_range(self):
        score = calculate_pause_score(0.15, 8, 45.0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0


class TestBuildMetrics:

    def test_returns_all_required_keys(self):
        audio_features = {
            "energy": 0.05, "pause_ratio": 0.15,
            "pause_count": 3, "duration": 30.0
        }
        filler_data = {"total_count": 2, "frequency_pct": 3.0}
        metrics = build_metrics(0.75, 0.60, filler_data, audio_features)
        for key in ("semantic", "coverage", "fluency", "confidence",
                    "pause", "filler", "communication", "quality"):
            assert key in metrics, f"Missing key: {key}"

    def test_all_values_in_range(self):
        audio_features = {
            "energy": 0.03, "pause_ratio": 0.20,
            "pause_count": 5, "duration": 60.0
        }
        filler_data = {"total_count": 1, "frequency_pct": 1.5}
        metrics = build_metrics(0.65, 0.50, filler_data, audio_features)
        for key, val in metrics.items():
            assert 0.0 <= val <= 100.0, f"{key}={val} out of range"
