# VBCUA/backend/api.py
import os
import io
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# --- Backend Module Imports ---
from backend.speech_to_text import transcribe_audio
from backend.semantic_analysis import get_similarity, analyze_coverage, get_understanding_level, extract_filler_words
from backend.audio_analysis import get_audio_features
from backend.scoring import build_metrics, calculate_final_score
from backend.gemini_feedback import get_qualitative_feedback
from backend.database_manager import save_session, get_all_sessions, get_session_by_id, get_session_count, init_db
from backend.report_generator import generate_pdf_report

# Initialize DB on API startup
init_db()

app = FastAPI(
    title="🎤 VBCUA Assessment API",
    description="REST API backend for the Voice-Based Concept Understanding Analyser (VBCUA)",
    version="1.0.0"
)

# ── Reference Concept Loader ───────────────────────────────────────────────────
_CONCEPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_concepts")

def load_reference_concepts() -> dict:
    """Scans reference_concepts/ and loads all .txt files."""
    concepts = {}
    if os.path.isdir(_CONCEPTS_DIR):
        for fname in sorted(os.listdir(_CONCEPTS_DIR)):
            if fname.endswith(".txt"):
                label = fname.replace("_", " ").replace(".txt", "")
                path = os.path.join(_CONCEPTS_DIR, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        concepts[label] = f.read().strip()
                except Exception:
                    pass
    return concepts

_REFERENCE_CONCEPTS = load_reference_concepts()

# ── Pydantic Request Models ────────────────────────────────────────────────────
class AudioFeaturesModel(BaseModel):
    tempo: float
    energy: float
    duration: float
    pause_ratio: float
    pause_count: int
    longest_pause: float
    speaking_duration: float

class FillerDataModel(BaseModel):
    total_count: int
    frequency_pct: float
    breakdown: dict

class MetricsModel(BaseModel):
    semantic: float
    coverage: float
    fluency: float
    confidence: float
    pause: float
    filler: float
    communication: float
    quality: float

class SessionSaveRequest(BaseModel):
    concept: str
    language: str
    final_score: float
    understanding_level: str
    semantic_score: float
    coverage_score: float
    transcript: str
    feedback: str
    audio_features: AudioFeaturesModel
    filler_data: FillerDataModel
    metrics: MetricsModel

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Returns api status and database count."""
    try:
        count = get_session_count()
        return {"status": "healthy", "database_sessions": count}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": f"Database unavailable: {str(e)}"}
        )


@app.get("/api/concepts")
def get_concepts():
    """Returns list of preset reference concepts."""
    return {"concepts": list(_REFERENCE_CONCEPTS.keys()), "data": _REFERENCE_CONCEPTS}


@app.post("/api/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    concept_name: Optional[str] = Form(None),
    manual_concept: Optional[str] = Form(None),
    language_mode: str = Form("Auto-Detect"),
    gemini_api_key: Optional[str] = Form(None)
):
    """
    Processes audio upload, runs speech transcription, compares against reference concept,
    extracts audio features, scores performance, and generates qualitative feedback.
    """
    # 1. Validate concept input
    reference_text = ""
    if concept_name and concept_name in _REFERENCE_CONCEPTS:
        reference_text = _REFERENCE_CONCEPTS[concept_name]
    elif manual_concept and manual_concept.strip():
        reference_text = manual_concept.strip()
    else:
        raise HTTPException(
            status_code=400,
            detail="Reference concept must be provided via 'concept_name' (preset) or 'manual_concept' (manual text)."
        )

    # 2. Write UploadFile to temp file for Whisper/Librosa
    ext = audio_file.filename.split('.')[-1] if audio_file.filename else "wav"
    fd, temp_path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 3. Transcribe speech (Whisper)
        stt_result = transcribe_audio(temp_path, language_choice=language_mode)
        if not stt_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Speech transcription failed: {stt_result.get('error')}"
            )
        
        transcript = stt_result.get("text", "")
        detected_lang = stt_result.get("detected_language", "en")

        # 4. Semantic Similarity (Sentence-BERT)
        sem_score = get_similarity(transcript, reference_text)

        # 5. Filler words
        filler_data = extract_filler_words(transcript)

        # 6. Keyword Coverage (NLTK)
        coverage_data = analyze_coverage(transcript, reference_text)

        # 7. Audio Feature Extraction (Librosa)
        audio_feats = get_audio_features(temp_path)

        # 8. Scoring Engine
        metrics = build_metrics(
            semantic_score=sem_score,
            coverage_score=coverage_data.get("coverage_score", 0.0),
            filler_data=filler_data,
            audio_features=audio_feats
        )
        final_score = calculate_final_score(metrics)
        understanding_lvl = get_understanding_level(sem_score)
        metrics["final_score"] = final_score

        # 9. AI qualitative feedback (Gemini or fallback)
        feedback = get_qualitative_feedback(
            user_transcript=transcript,
            reference_concept=reference_text,
            semantic_score=sem_score,
            metrics=metrics,
            filler_data=filler_data,
            audio_features=audio_feats,
            api_key=gemini_api_key
        )

        # 10. Extract waveform samples for client visualization (downsampled to 500 points)
        import librosa
        y, _ = librosa.load(temp_path, sr=None)
        # Avoid heavy transfer payload by downsampling
        step = max(1, len(y) // 500)
        waveform_downsampled = y[::step].tolist() if len(y) > 0 else []

        return {
            "success": True,
            "transcript": transcript,
            "detected_language": detected_lang,
            "semantic_score": sem_score,
            "coverage_data": coverage_data,
            "audio_features": audio_feats,
            "filler_data": filler_data,
            "metrics": metrics,
            "final_score": final_score,
            "understanding_level": understanding_lvl,
            "feedback": feedback,
            "waveform": waveform_downsampled
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/sessions")
def save_new_session(req: SessionSaveRequest):
    """Saves evaluation session details to the SQLite database."""
    try:
        session_id = save_session(
            concept=req.concept,
            language=req.language,
            final_score=req.final_score,
            understanding_level=req.understanding_level,
            semantic_score=req.semantic_score,
            coverage_score=req.coverage_score,
            transcript=req.transcript,
            feedback=req.feedback,
            audio_features=req.audio_features.dict(),
            filler_data=req.filler_data.dict(),
            metrics=req.metrics.dict()
        )
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database save error: {str(e)}")


@app.get("/api/sessions")
def get_sessions(limit: int = 20):
    """Retrieves list of recent sessions from database."""
    try:
        sessions = get_all_sessions(limit=limit)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fetch error: {str(e)}")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: int):
    """Retrieves full details of a specific session."""
    session = get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found.")
    return session


@app.get("/api/sessions/{session_id}/pdf")
def download_session_pdf(session_id: int):
    """Generates and returns the PDF assessment report for a specific session."""
    session = get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session with ID {session_id} not found.")

    # Reconstruct PDF data structure
    eval_data = session.get("evaluation", {})
    audio_feats = session.get("audio_features", {})
    filler_data = session.get("filler_words", {})
    metrics_data = session.get("metrics_breakdown", {}).get("metrics", {})

    pdf_data = {
        "concept": session.get("concept", "N/A"),
        "language": session.get("language", "en"),
        "similarity": f"{eval_data.get('semantic_score', 0.0)*100:.2f}%",
        "score": f"{session.get('final_score', 0.0):.1f}/100",
        "tempo": f"{audio_feats.get('tempo', 0.0):.1f} BPM",
        "transcript": eval_data.get("transcript", ""),
        "metrics": metrics_data,
        "filler_words": filler_data.get("breakdown", {}),
        "pause_ratio": audio_feats.get("pause_ratio", 0.0),
        "understanding_level": session.get("understanding_level", "N/A"),
        "feedback": eval_data.get("feedback", ""),
        "waveform_y": [0.0] * 100,  # Waveform signals are not persisted, use empty array for report rendering
    }

    try:
        pdf_bytes = generate_pdf_report(pdf_data)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=VBCUA_Report_Session_{session_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")
