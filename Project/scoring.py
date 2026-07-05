# VBCUA/backend/scoring.py


def calculate_final_score(metrics: dict) -> float:
    """
    Computes a weighted final score from all assessment sub-metrics.
    Weights are aligned with Requirements.md scoring priorities.

    Expected metric keys:
        semantic    — Sentence-BERT cosine similarity (0-100)
        coverage    — Keyword coverage percentage (0-100)
        fluency     — Speech fluency based on filler words (0-100)
        confidence  — Vocal energy confidence proxy (0-100)
        pause       — Pause control quality (0-100)
        filler      — Filler word penalty score (0-100)
        communication — Composite communication score (0-100)
        quality     — Audio channel quality (0-100)
    """
    weights = {
        "semantic":      0.35,
        "coverage":      0.20,
        "fluency":       0.15,
        "communication": 0.12,
        "confidence":    0.08,
        "pause":         0.05,
        "filler":        0.03,
        "quality":       0.02,
    }

    final_score = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        value = metrics.get(key)
        if value is not None:
            final_score += float(value) * weight
            total_weight += weight

    # Normalize in case some metrics are missing
    if total_weight > 0 and total_weight < 1.0:
        final_score = final_score / total_weight

    return round(min(100.0, max(0.0, final_score)), 2)


def calculate_communication_score(
    fluency_score: float,
    pause_ratio: float,
    filler_frequency_pct: float
) -> float:
    """
    Composite communication score based on:
      - Fluency (filler-word-based)
      - Pause ratio (silence fraction; lower is generally better)
      - Filler frequency percentage

    Returns a 0–100 score.
    """
    # Pause penalty: >40% silence is bad, <10% is fine
    pause_penalty = max(0.0, (pause_ratio - 0.10) * 100)

    # Filler penalty: each % point of fillers costs 2 points
    filler_penalty = min(30.0, filler_frequency_pct * 2.0)

    base = (fluency_score * 0.60) + (100.0 - pause_penalty) * 0.25 + (100.0 - filler_penalty * 2) * 0.15

    return round(min(100.0, max(0.0, base)), 2)


def calculate_pause_score(pause_ratio: float, pause_count: int, duration: float) -> float:
    """
    Scores speech pause quality:
      - Ideal pause ratio: 10–30% of total speech
      - Too many pauses or too long pauses lower the score

    Returns a 0–100 score.
    """
    # Ratio score: ideal window 0.10 – 0.30
    if 0.10 <= pause_ratio <= 0.30:
        ratio_score = 100.0
    elif pause_ratio < 0.10:
        ratio_score = 70.0 + (pause_ratio / 0.10) * 30.0
    else:
        ratio_score = max(30.0, 100.0 - (pause_ratio - 0.30) * 200)

    # Pause frequency penalty (normalize by duration)
    pauses_per_minute = (pause_count / max(duration, 1)) * 60
    freq_penalty = min(30.0, max(0.0, pauses_per_minute - 10) * 2)

    return round(min(100.0, max(0.0, ratio_score - freq_penalty)), 2)


def get_understanding_level(semantic_score: float) -> str:
    """
    Maps a 0–1 semantic similarity score to a qualitative label.
    Mirrors semantic_analysis.get_understanding_level for use in scoring pipeline.
    """
    if semantic_score >= 0.70:
        return "Strong Understanding"
    elif semantic_score >= 0.40:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"


def build_metrics(
    semantic_score: float,
    coverage_score: float,
    filler_data: dict,
    audio_features: dict
) -> dict:
    """
    Convenience builder: constructs the full metrics dict from raw sub-system outputs.
    Returns a dict ready for calculate_final_score().
    """
    filler_total = filler_data.get("total_count", 0)
    filler_freq_pct = filler_data.get("frequency_pct", 0.0)
    energy = audio_features.get("energy", 0.05)
    pause_ratio = audio_features.get("pause_ratio", 0.10)
    pause_count = audio_features.get("pause_count", 2)
    duration = audio_features.get("duration", 10.0)

    fluency = max(50.0, 100.0 - (filler_total * 4))
    confidence = 85.0 if energy > 0.02 else 70.0
    filler_penalty = max(40.0, 100.0 - (filler_total * 6))
    pause_score = calculate_pause_score(pause_ratio, pause_count, duration)
    communication = calculate_communication_score(fluency, pause_ratio, filler_freq_pct)

    return {
        "semantic":      round(semantic_score * 100, 2),
        "coverage":      round(min(100.0, coverage_score * 100), 2),
        "fluency":       round(fluency, 2),
        "confidence":    round(confidence, 2),
        "pause":         round(pause_score, 2),
        "filler":        round(filler_penalty, 2),
        "communication": round(communication, 2),
        "quality":       92.0,
    }