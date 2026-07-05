# VBCUA/backend/gemini_feedback.py
import os
import google.generativeai as genai
from backend.semantic_analysis import get_understanding_level


def generate_gemini_feedback(
    user_transcript: str,
    reference_concept: str,
    semantic_score: float,
    metrics: dict,
    filler_data: dict,
    audio_features: dict,
    api_key: str = None
) -> str | None:
    """
    Calls the Google Gemini API to generate professional communication and conceptual coaching feedback.
    Returns None if the API call fails or the API key is not provided.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None

    try:
        genai.configure(api_key=key)

        semantic_pct = semantic_score * 100
        coverage_pct = metrics.get("coverage", 0.0)
        tempo = audio_features.get("tempo", 120.0)
        pause_ratio_pct = audio_features.get("pause_ratio", 0.10) * 100
        filler_count = filler_data.get("total_count", 0)
        filler_freq = filler_data.get("frequency_pct", 0.0)

        prompt = f"""You are an expert AI communication coach and educational assessor for the Voice-Based Concept Understanding Analyser (VBCUA).
Analyze the user's spoken explanation of a concept and compare it with the reference concept.

Reference Concept:
"{reference_concept}"

User's Spoken Explanation:
"{user_transcript}"

Performance Metrics:
- Semantic Similarity Match: {semantic_pct:.1f}%
- Core Keyword Coverage: {coverage_pct:.1f}%
- Speech Pace: {tempo:.1f} BPM (Ideal: 100-130 BPM)
- Pause Ratio (Silence Fraction): {pause_ratio_pct:.1f}% (Ideal: 10-30%)
- Total Filler Words: {filler_count} (Filler Frequency: {filler_freq:.2f}%)

Generate a detailed evaluation with three sections:
1. **Conceptual Understanding (Content Assessment)**: Critically assess the user's semantic understanding. Highlight what they got right, what core ideas they missed, or if they deviated from the expected definition.
2. **Delivery & Fluency (Speech Assessment)**: Assess the speech quality (pacing, silence/pauses, and filler word usage). Give specific feedback on confidence and articulation.
3. **Actionable Suggestions**: Provide 3-4 bulleted, highly specific, and actionable recommendations to help the user improve both their conceptual grasp and public speaking delivery.

Make the tone encouraging, constructive, and professional.
Keep the total response concise (around 150-250 words) so it fits nicely on a dashboard and in a 2-page PDF report. Do not use Markdown headings that are too large (prefer bold text or bullet points) to keep it clean.
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        # Print warning but allow graceful fallback
        print(f"Warning: Gemini feedback generation failed: {e}")
        return None


def get_static_feedback(understanding_level: str, final_score: float) -> str:
    """Returns fallback static feedback based on understanding level brackets."""
    if understanding_level == "Strong Understanding":
        return (f"Excellent! You demonstrated strong conceptual understanding with a score of "
                f"{final_score:.1f}/100. Your explanation closely matched the reference definition. "
                f"Continue refining your delivery for even greater clarity.")
    elif understanding_level == "Moderate Understanding":
        return (f"Good effort! Your score of {final_score:.1f}/100 indicates a moderate grasp of the concept. "
                f"Consider expanding on key terminology and reducing filler words. "
                f"Review the missing keywords below to strengthen your explanation.")
    else:
        return (f"Your score of {final_score:.1f}/100 suggests the explanation needs improvement. "
                f"Focus on using the core vocabulary of the topic, maintaining a steady pace, "
                f"and structuring your explanation more clearly around the reference definition.")


def get_qualitative_feedback(
    user_transcript: str,
    reference_concept: str,
    semantic_score: float,
    metrics: dict,
    filler_data: dict,
    audio_features: dict,
    api_key: str = None
) -> str:
    """
    High-level API: Tries to get dynamic Gemini feedback first.
    Falls back to static rule-based feedback on any failure or missing credentials.
    """
    final_score = metrics.get("final_score", 0.0)
    if not final_score and metrics:
        from backend.scoring import calculate_final_score
        final_score = calculate_final_score(metrics)

    understanding_level = get_understanding_level(semantic_score)

    feedback = generate_gemini_feedback(
        user_transcript=user_transcript,
        reference_concept=reference_concept,
        semantic_score=semantic_score,
        metrics=metrics,
        filler_data=filler_data,
        audio_features=audio_features,
        api_key=api_key
    )

    if not feedback:
        feedback = get_static_feedback(understanding_level, final_score)

    return feedback
