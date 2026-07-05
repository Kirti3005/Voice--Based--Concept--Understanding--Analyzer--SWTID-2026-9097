# VBCUA/tests/test_semantic_analysis.py
"""
Pytest tests for backend.semantic_analysis module.
"""
import pytest
from backend.semantic_analysis import (
    get_similarity,
    extract_filler_words,
    extract_keywords,
    analyze_coverage,
    get_understanding_level,
)


class TestGetSimilarity:

    def test_identical_texts_return_near_one(self):
        """Identical text should yield near-perfect similarity."""
        score = get_similarity("machine learning is great", "machine learning is great")
        assert score > 0.95

    def test_unrelated_texts_return_low_score(self):
        """Completely unrelated texts should yield a low similarity."""
        score = get_similarity("I love pizza", "quantum mechanics wave function")
        assert score < 0.6

    def test_returns_float(self):
        score = get_similarity("hello world", "hello there")
        assert isinstance(score, float)

    def test_score_in_valid_range(self):
        """Score must always be in [0.0, 1.0] (cosine similarity range)."""
        score = get_similarity("deep learning neural network", "Python programming language")
        assert 0.0 <= score <= 1.0

    def test_empty_user_text_returns_zero(self):
        assert get_similarity("", "reference text") == 0.0

    def test_empty_reference_text_returns_zero(self):
        assert get_similarity("user speech", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert get_similarity("", "") == 0.0


class TestExtractFillerWords:

    def test_detects_um_and_uh(self):
        result = extract_filler_words("um this is um like you know uh great")
        assert result["total_count"] >= 4

    def test_empty_text_returns_zero(self):
        result = extract_filler_words("")
        assert result["total_count"] == 0
        assert result["frequency_pct"] == 0.0

    def test_no_fillers_in_clean_text(self):
        result = extract_filler_words("Machine learning is a subset of artificial intelligence.")
        assert result["total_count"] == 0

    def test_returns_breakdown_dict(self):
        result = extract_filler_words("um basically like um")
        assert "breakdown" in result
        assert isinstance(result["breakdown"], dict)

    def test_frequency_pct_is_non_negative(self):
        result = extract_filler_words("um uh like so")
        assert result["frequency_pct"] >= 0.0

    def test_filler_count_matches_breakdown_sum(self):
        result = extract_filler_words("um um uh like")
        total_from_breakdown = sum(result["breakdown"].values())
        assert total_from_breakdown == result["total_count"]


class TestExtractKeywords:

    def test_returns_list(self):
        kws = extract_keywords("Machine learning uses algorithms")
        assert isinstance(kws, list)

    def test_removes_stopwords(self):
        kws = extract_keywords("the quick brown fox jumps over the lazy dog")
        for stopword in ("the", "over", "is", "a"):
            assert stopword not in kws

    def test_empty_text_returns_empty_list(self):
        assert extract_keywords("") == []

    def test_unique_keywords(self):
        kws = extract_keywords("machine learning machine learning")
        assert len(kws) == len(set(kws))


class TestAnalyzeCoverage:

    def test_perfect_coverage(self):
        ref = "machine learning uses algorithms and data"
        user = "machine learning uses algorithms and data"
        result = analyze_coverage(user, ref)
        assert result["coverage_score"] >= 0.8

    def test_zero_coverage(self):
        ref = "quantum entanglement superposition"
        user = "pizza delivery spaghetti"
        result = analyze_coverage(user, ref)
        assert result["coverage_score"] <= 0.2

    def test_returns_required_keys(self):
        result = analyze_coverage("test text here", "reference concept text")
        assert "coverage_score" in result
        assert "covered_keywords" in result
        assert "missing_keywords" in result

    def test_score_in_range(self):
        result = analyze_coverage("some speech", "reference definition")
        assert 0.0 <= result["coverage_score"] <= 1.0

    def test_empty_reference_returns_zero(self):
        result = analyze_coverage("user speech text", "")
        assert result["coverage_score"] == 0.0


class TestGetUnderstandingLevel:

    def test_strong_at_0_70(self):
        assert get_understanding_level(0.70) == "Strong Understanding"

    def test_strong_at_0_90(self):
        assert get_understanding_level(0.90) == "Strong Understanding"

    def test_moderate_at_0_55(self):
        assert get_understanding_level(0.55) == "Moderate Understanding"

    def test_moderate_at_0_40(self):
        assert get_understanding_level(0.40) == "Moderate Understanding"

    def test_poor_at_0_39(self):
        assert get_understanding_level(0.39) == "Poor Understanding"

    def test_poor_at_0_0(self):
        assert get_understanding_level(0.0) == "Poor Understanding"

    def test_boundary_exactly_0_70_is_strong(self):
        assert get_understanding_level(0.70) == "Strong Understanding"

    def test_boundary_exactly_0_40_is_moderate(self):
        assert get_understanding_level(0.40) == "Moderate Understanding"
