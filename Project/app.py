Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass# VBCUA/app.py
import streamlit as st
import os
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Backend Module Imports ─────────────────────────────────────────────────────
from backend.speech_to_text import transcribe_audio
from backend.semantic_analysis import get_similarity, extract_filler_words, analyze_coverage, get_understanding_level
from backend.scoring import calculate_final_score, build_metrics
from backend.audio_analysis import get_audio_features
from backend.database_manager import init_db, save_session, get_all_sessions, get_session_count
from backend.report_generator import generate_pdf_report

# ── DB Init on startup ─────────────────────────────────────────────────────────
init_db()

# ── Reference Concept Loader ───────────────────────────────────────────────────
_CONCEPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_concepts")

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

# ── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="VBCUA Assessment Platform",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.97) !important;
        border-right: 1px solid #3730a3;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.4);
        margin-bottom: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0,0,0,0.5);
    }
    .metric-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .level-badge-strong {
        display: inline-block;
        background: rgba(21,128,61,0.25);
        border: 1px solid #16a34a;
        color: #4ade80;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.5px;
    }
    .level-badge-moderate {
        display: inline-block;
        background: rgba(180,83,9,0.25);
        border: 1px solid #d97706;
        color: #fbbf24;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
    }
    .level-badge-poor {
        display: inline-block;
        background: rgba(220,38,38,0.2);
        border: 1px solid #dc2626;
        color: #f87171;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
    }
    .dev-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .dev-card:hover { border-color: #818cf8; }
    .transcript-box {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #22d3ee;
        padding: 16px 18px;
        border-radius: 4px;
        font-size: 15px;
        line-height: 1.7;
        color: #e2e8f0;
        margin-top: 10px;
    }
    .keyword-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .chip-covered {
        background: rgba(21,128,61,0.3);
        color: #4ade80;
        border: 1px solid #16a34a;
    }
    .chip-missing {
        background: rgba(220,38,38,0.2);
        color: #f87171;
        border: 1px solid #dc2626;
    }
    .history-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #cbd5e1;
    }
    div[data-testid="stTabs"] button {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 8px 0 18px 0;'>
        <h1 style='font-size:2.4rem; background:linear-gradient(90deg,#22d3ee,#818cf8,#c084fc);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800;'>
            🎤 Voice-Based Concept Understanding Analyzer
        </h1>
        <p style='color:#64748b; font-size:14px; margin-top:-8px;'>
            AI-Powered Speech & Semantic Assessment Platform
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
app_tabs = st.tabs(["🏠 Home Dashboard", "📊 Detailed Analytics", "👥 Developers Team"])

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛠️ Evaluation Parameters")
st.sidebar.markdown("---")

# Language selection
language_mode = st.sidebar.selectbox(
    "🌐 Language Mode",
    ["Auto-Detect", "English", "Hindi"]
)

# Reference concept — preset or manual
st.sidebar.markdown("#### 📝 Reference Concept")
concept_source = st.sidebar.radio(
    "Concept Source",
    ["Load Preset", "Enter Manually"],
    horizontal=True
)

reference_concept = ""
selected_concept_name = ""

if concept_source == "Load Preset":
    if _REFERENCE_CONCEPTS:
        selected_concept_name = st.sidebar.selectbox(
            "Select Concept",
            list(_REFERENCE_CONCEPTS.keys())
        )
        reference_concept = _REFERENCE_CONCEPTS[selected_concept_name]
        st.sidebar.markdown(
            f"<div style='background:rgba(56,189,248,0.08);border:1px solid #38bdf8;"
            f"border-radius:8px;padding:10px;font-size:12px;color:#94a3b8;'>"
            f"{reference_concept[:200]}...</div>",
            unsafe_allow_html=True
        )
    else:
        st.sidebar.warning("No preset concepts found in reference_concepts/")
        reference_concept = st.sidebar.text_area("Paste reference concept text:", height=120)
else:
    reference_concept = st.sidebar.text_area(
        "Paste reference concept text:",
        height=130,
        placeholder="Enter the reference definition/concept here..."
    )

# Audio upload
st.sidebar.markdown("#### 📁 Audio File")
uploaded_audio = st.sidebar.file_uploader(
    "Upload audio (WAV or MP3)",
    type=['wav', 'mp3']
)

st.sidebar.markdown("---")

# Run Mode Selection
st.sidebar.markdown("#### ⚙️ Execution Settings")
run_mode = st.sidebar.radio(
    "Run Mode",
    ["In-Process (Local)", "FastAPI REST Server (Localhost)"],
    index=0,
    help="In-Process runs calculations locally. FastAPI delegates to the REST backend server."
)

# Gemini API Key Input
gemini_api_key = st.sidebar.text_input(
    "🔑 Gemini API Key (Optional)",
    type="password",
    placeholder="Paste Gemini API key...",
    help="Optional: Enables dynamic AI-generated feedback. Fallback is static brackets."
)

st.sidebar.markdown("---")

# ── Session State Init ─────────────────────────────────────────────────────────
for key, default in {
    "processed": False,
    "user_transcript": "",
    "detected_lang": "en",
    "semantic_score": 0.0,
    "filler_metrics": {},
    "coverage_data": {},
    "audio_features": {},
    "waveform_y": np.array([]),
    "waveform_sr": 44100,
    "metrics": {},
    "final_score": 0.0,
    "understanding_level": "",
    "feedback": "",
    "session_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Run Pipeline ───────────────────────────────────────────────────────────────
if st.sidebar.button("⚡ Run Analysis", use_container_width=True, type="primary"):
    if not uploaded_audio:
        st.sidebar.error("❌ Please upload an audio file.")
    elif not reference_concept.strip():
        st.sidebar.error("❌ Please provide a reference concept.")
    else:
        with st.spinner("🔄 Processing — this may take a moment..."):
            ext = uploaded_audio.name.split('.')[-1]
            temp_path = f"_vbcua_temp_audio.{ext}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())

            try:
                if run_mode == "FastAPI REST Server (Localhost)":
                    import requests
                    api_url = "http://127.0.0.1:8000/api/analyze"
                    
                    with open(temp_path, "rb") as audio_file:
                        files = {"audio_file": (uploaded_audio.name, audio_file, f"audio/{ext}")}
                        form_data = {
                            "language_mode": language_mode,
                            "gemini_api_key": gemini_api_key or ""
                        }
                        if concept_source == "Load Preset":
                            form_data["concept_name"] = selected_concept_name
                        else:
                            form_data["manual_concept"] = reference_concept
                            
                        res = requests.post(api_url, files=files, data=form_data)
                        if res.status_code != 200:
                            raise Exception(f"FastAPI Server returned error ({res.status_code}): {res.text}")
                            
                        api_res = res.json()

                    st.session_state.user_transcript = api_res.get("transcript", "")
                    st.session_state.detected_lang = api_res.get("detected_language", "en")
                    st.session_state.semantic_score = api_res.get("semantic_score", 0.0)
                    st.session_state.filler_metrics = api_res.get("filler_data", {})
                    st.session_state.coverage_data = api_res.get("coverage_data", {})
                    st.session_state.audio_features = api_res.get("audio_features", {})
                    st.session_state.metrics = api_res.get("metrics", {})
                    st.session_state.final_score = api_res.get("final_score", 0.0)
                    st.session_state.understanding_level = api_res.get("understanding_level", "")
                    st.session_state.feedback = api_res.get("feedback", "")
                    
                    st.session_state.waveform_y = np.array(api_res.get("waveform", []))
                    st.session_state.waveform_sr = 44100
                    
                    # Save session using the API
                    save_url = "http://127.0.0.1:8000/api/sessions"
                    concept_name = selected_concept_name or reference_concept[:40]
                    save_payload = {
                        "concept": concept_name,
                        "language": st.session_state.detected_lang,
                        "final_score": st.session_state.final_score,
                        "understanding_level": st.session_state.understanding_level,
                        "semantic_score": st.session_state.semantic_score,
                        "coverage_score": st.session_state.coverage_data.get("coverage_score", 0.0),
                        "transcript": st.session_state.user_transcript,
                        "feedback": st.session_state.feedback,
                        "audio_features": st.session_state.audio_features,
                        "filler_data": st.session_state.filler_metrics,
                        "metrics": st.session_state.metrics
                    }
                    save_res = requests.post(save_url, json=save_payload)
                    if save_res.status_code == 200:
                        st.session_state.session_id = save_res.json().get("session_id")
                    else:
                        raise Exception(f"FastAPI Session Save failed: {save_res.text}")
                    st.session_state.processed = True

                else:
                    # In-Process (Local) Pipeline
                    # 1. Speech-to-Text (Whisper)
                    stt_result = transcribe_audio(temp_path, language_choice=language_mode)
                    st.session_state.user_transcript = stt_result.get("text", "")
                    st.session_state.detected_lang = stt_result.get("detected_language", "en")

                    # 2. Semantic Analysis (Sentence-BERT)
                    sem_score = get_similarity(st.session_state.user_transcript, reference_concept)
                    st.session_state.semantic_score = sem_score

                    # 3. Filler Word Detection
                    filler_data = extract_filler_words(st.session_state.user_transcript)
                    st.session_state.filler_metrics = filler_data

                    # 4. NLTK Coverage Analysis
                    coverage_data = analyze_coverage(st.session_state.user_transcript, reference_concept)
                    st.session_state.coverage_data = coverage_data

                    # 5. Audio Feature Extraction (Librosa)
                    audio_feats = get_audio_features(temp_path)
                    st.session_state.audio_features = audio_feats

                    # 6. Waveform for visualization
                    y, sr = librosa.load(temp_path, sr=None)
                    st.session_state.waveform_y = y[::80]
                    st.session_state.waveform_sr = sr

                    # 7. Scoring Engine
                    metrics = build_metrics(
                        semantic_score=sem_score,
                        coverage_score=coverage_data.get("coverage_score", 0.0),
                        filler_data=filler_data,
                        audio_features=audio_feats,
                    )
                    st.session_state.metrics = metrics
                    st.session_state.final_score = calculate_final_score(metrics)
                    st.session_state.understanding_level = get_understanding_level(sem_score)

                    # 8. Feedback text (Gemini integration with fallback)
                    from backend.gemini_feedback import get_qualitative_feedback
                    fb = get_qualitative_feedback(
                        user_transcript=st.session_state.user_transcript,
                        reference_concept=reference_concept,
                        semantic_score=sem_score,
                        metrics=metrics,
                        filler_data=filler_data,
                        audio_features=audio_feats,
                        api_key=gemini_api_key
                    )
                    st.session_state.feedback = fb

                    # 9. Save to SQLite
                    concept_name = selected_concept_name or reference_concept[:40]
                    sid = save_session(
                        concept=concept_name,
                        language=st.session_state.detected_lang,
                        final_score=st.session_state.final_score,
                        understanding_level=st.session_state.understanding_level,
                        semantic_score=sem_score,
                        coverage_score=coverage_data.get("coverage_score", 0.0),
                        transcript=st.session_state.user_transcript,
                        feedback=fb,
                        audio_features=audio_feats,
                        filler_data=filler_data,
                        metrics=metrics,
                    )
                    st.session_state.session_id = sid
                    st.session_state.processed = True

            except Exception as e:
                st.error(f"❌ Pipeline Error: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if st.session_state.processed:
    st.sidebar.success(f"✅ Analysis complete! Session #{st.session_state.session_id}")
    total = get_session_count()
    st.sidebar.markdown(
        f"<div style='text-align:center;color:#64748b;font-size:12px;'>📦 Total sessions stored: {total}</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with app_tabs[0]:
    if st.session_state.processed:
        # ── Understanding Level Badge ──────────────────────────────────────────
        level = st.session_state.understanding_level
        badge_cls = {
            "Strong Understanding": "level-badge-strong",
            "Moderate Understanding": "level-badge-moderate",
            "Poor Understanding": "level-badge-poor",
        }.get(level, "level-badge-moderate")
        st.markdown(
            f"<div style='text-align:center; margin-bottom:20px;'>"
            f"<span class='{badge_cls}'>🎯 {level}</span></div>",
            unsafe_allow_html=True
        )

        # ── Metric Cards Row 1 ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">⭐ Final Score</div>'
                f'<div class="metric-value">{st.session_state.final_score:.1f}%</div></div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">🧠 Semantic Match</div>'
                f'<div class="metric-value">{st.session_state.semantic_score:.1%}</div></div>',
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">📖 Concept Coverage</div>'
                f'<div class="metric-value">{st.session_state.coverage_data.get("coverage_score", 0)*100:.1f}%</div></div>',
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">🌍 Language</div>'
                f'<div class="metric-value">{st.session_state.detected_lang.upper()}</div></div>',
                unsafe_allow_html=True
            )

        # ── Metric Cards Row 2 ─────────────────────────────────────────────────
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">🎵 Speech Pace</div>'
                f'<div class="metric-value">{st.session_state.audio_features.get("tempo", 0):.1f} BPM</div></div>',
                unsafe_allow_html=True
            )
        with c6:
            pause_pct = st.session_state.audio_features.get("pause_ratio", 0) * 100
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">⏸️ Pause Ratio</div>'
                f'<div class="metric-value">{pause_pct:.1f}%</div></div>',
                unsafe_allow_html=True
            )
        with c7:
            filler_count = st.session_state.filler_metrics.get("total_count", 0)
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">💬 Filler Words</div>'
                f'<div class="metric-value">{filler_count}</div></div>',
                unsafe_allow_html=True
            )
        with c8:
            comm_score = st.session_state.metrics.get("communication", 0)
            st.markdown(
                f'<div class="glass-card"><div class="metric-title">🗣️ Communication</div>'
                f'<div class="metric-value">{comm_score:.1f}%</div></div>',
                unsafe_allow_html=True
            )

        # ── Waveform ───────────────────────────────────────────────────────────
        st.markdown("### 📈 Audio Waveform")
        wf = st.session_state.waveform_y
        if len(wf) > 0:
            fig, ax = plt.subplots(figsize=(11, 2.2), facecolor='none')
            ax.set_facecolor('none')
            ax.plot(wf, color='#22d3ee', alpha=0.85, linewidth=0.9)
            ax.axis('off')
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        # ── Audio Playback ─────────────────────────────────────────────────────
        st.markdown("### 🎧 Audio Playback")
        st.audio(uploaded_audio)

        # ── Transcript ─────────────────────────────────────────────────────────
        st.markdown("### 📝 Speech Transcript")
        if st.session_state.user_transcript:
            st.markdown(
                f'<div class="transcript-box">"{st.session_state.user_transcript}"</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="transcript-box" style="border-left-color:#ef4444;">'
                'No clear speech segments could be captured.</div>',
                unsafe_allow_html=True
            )

        # ── AI Feedback ────────────────────────────────────────────────────────
        st.markdown("### 💡 AI Feedback")
        st.info(st.session_state.feedback)

        # ── Keyword Analysis ───────────────────────────────────────────────────
        cov = st.session_state.coverage_data
        covered_kw = cov.get("covered_keywords", [])
        missing_kw = cov.get("missing_keywords", [])

        if covered_kw or missing_kw:
            st.markdown("### 🔍 Keyword Coverage")
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("**✅ Covered Keywords**")
                chips = " ".join([f'<span class="keyword-chip chip-covered">{w}</span>' for w in covered_kw[:20]])
                st.markdown(chips if chips else "_(none detected)_", unsafe_allow_html=True)
            with kw_col2:
                st.markdown("**❌ Missing Keywords**")
                chips = " ".join([f'<span class="keyword-chip chip-missing">{w}</span>' for w in missing_kw[:20]])
                st.markdown(chips if chips else "_(all covered!)_", unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='text-align:center; padding:60px 0;'>
                <div style='font-size:4rem; margin-bottom:16px;'>🎤</div>
                <h3 style='color:#94a3b8; font-weight:700;'>Ready to Analyze</h3>
                <p style='color:#64748b; font-size:15px; max-width:400px; margin:0 auto;'>
                    Upload an audio file and select a reference concept from the sidebar,
                    then click <b>Run Analysis</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DETAILED ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with app_tabs[1]:
    if st.session_state.processed:
        st.markdown("## 📊 Detailed Analytics Report")

        # ── Score Breakdown Chart ──────────────────────────────────────────────
        st.markdown("### 📉 Score Breakdown (All Metrics)")
        metrics = st.session_state.metrics
        metric_display_names = {
            "semantic": "Semantic\nSimilarity",
            "coverage": "Concept\nCoverage",
            "fluency": "Speech\nFluency",
            "confidence": "Vocal\nConfidence",
            "pause": "Pause\nControl",
            "filler": "Filler Word\nMitigation",
            "communication": "Communication\nScore",
            "quality": "Audio\nQuality",
        }
        labels = [metric_display_names.get(k, k) for k in metrics]
        values = list(metrics.values())
        bar_colors = ['#38bdf8', '#818cf8', '#c084fc', '#34d399', '#fbbf24', '#f87171', '#22d3ee', '#a78bfa']

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
        ax.set_facecolor('#1e1b4b')
        bars = ax.bar(labels, values, color=bar_colors[:len(labels)], width=0.55, edgecolor='none', zorder=3)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)", color='#94a3b8', fontsize=11)
        ax.tick_params(axis='x', colors='#94a3b8', labelsize=9)
        ax.tick_params(axis='y', colors='#94a3b8', labelsize=10)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#334155', linestyle='--', alpha=0.5, zorder=0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', color='#f8fafc', fontsize=9, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        # ── Pause Analysis ─────────────────────────────────────────────────────
        st.markdown("### ⏸️ Pause Analysis")
        af = st.session_state.audio_features
        pa1, pa2, pa3, pa4 = st.columns(4)
        pa1.metric("Total Duration", f"{af.get('duration', 0):.1f}s")
        pa2.metric("Speaking Duration", f"{af.get('speaking_duration', 0):.1f}s")
        pa3.metric("Pause Count", str(af.get('pause_count', 0)))
        pa4.metric("Longest Pause", f"{af.get('longest_pause', 0):.2f}s")

        # Pause ratio visual bar
        pause_ratio = af.get("pause_ratio", 0)
        fig2, ax2 = plt.subplots(figsize=(10, 1.2), facecolor='#0f172a')
        ax2.set_facecolor('#0f172a')
        ax2.barh(['Speaking'], [1 - pause_ratio], color='#38bdf8', height=0.5)
        ax2.barh(['Speaking'], [pause_ratio], left=[1 - pause_ratio], color='#475569', height=0.5)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Fraction of total duration", color='#94a3b8', fontsize=9)
        ax2.tick_params(colors='#94a3b8')
        for spine in ax2.spines.values():
            spine.set_color('#334155')
        ax2.text(0.02, 0, f"Speech {(1-pause_ratio)*100:.1f}%", color='#f8fafc', va='center', fontsize=9, fontweight='bold')
        ax2.text(1 - pause_ratio + 0.01, 0, f"Silence {pause_ratio*100:.1f}%", color='#94a3b8', va='center', fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

        # ── Filler Word Analysis ───────────────────────────────────────────────
        st.markdown("### 💬 Filler Word Analysis")
        filler_data = st.session_state.filler_metrics
        breakdown = filler_data.get("breakdown", {})
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Total Fillers", str(filler_data.get("total_count", 0)))
        fc2.metric("Filler Frequency", f"{filler_data.get('frequency_pct', 0):.2f}%")
        fc3.metric("Unique Filler Types", str(len(breakdown)))

        if breakdown:
            fig3, ax3 = plt.subplots(figsize=(6, 3.5), facecolor='#0f172a')
            ax3.set_facecolor('#1e1b4b')
            fwords = list(breakdown.keys())
            fcounts = list(breakdown.values())
            wedge_colors = ['#f87171', '#fbbf24', '#c084fc', '#38bdf8', '#34d399', '#818cf8']
            ax3.pie(
                fcounts,
                labels=fwords,
                autopct='%1.0f%%',
                colors=wedge_colors[:len(fwords)],
                textprops={'color': '#f8fafc', 'fontsize': 10},
                wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
            )
            ax3.set_title("Filler Word Distribution", color='#94a3b8', fontsize=11, pad=10)
            fig3.tight_layout()
            st.pyplot(fig3, clear_figure=True)
            plt.close(fig3)
        else:
            st.success("🎉 No filler words detected!")

        # ── Keyword Coverage Visual ────────────────────────────────────────────
        st.markdown("### 🔍 Concept Keyword Coverage")
        cov = st.session_state.coverage_data
        covered = cov.get("covered_keywords", [])
        missing = cov.get("missing_keywords", [])
        cov_score = cov.get("coverage_score", 0)

        fig4, ax4 = plt.subplots(figsize=(5, 3), facecolor='#0f172a')
        ax4.set_facecolor('#0f172a')
        ax4.pie(
            [max(cov_score, 0.001), max(1 - cov_score, 0.001)],
            labels=[f'Covered\n{cov_score*100:.1f}%', f'Missing\n{(1-cov_score)*100:.1f}%'],
            colors=['#34d399', '#f87171'],
            startangle=90,
            textprops={'color': '#f8fafc', 'fontsize': 10},
            wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
        )
        ax4.set_title("Coverage Ratio", color='#94a3b8', fontsize=11)
        fig4.tight_layout()
        st.pyplot(fig4, clear_figure=True)
        plt.close(fig4)

        kc1, kc2 = st.columns(2)
        with kc1:
            st.markdown("**✅ Keywords You Used:**")
            if covered:
                for kw in covered[:15]:
                    st.markdown(f'<span class="keyword-chip chip-covered">{kw}</span>', unsafe_allow_html=True)
            else:
                st.markdown("_No reference keywords matched._")
        with kc2:
            st.markdown("**❌ Keywords to Include:**")
            if missing:
                for kw in missing[:15]:
                    st.markdown(f'<span class="keyword-chip chip-missing">{kw}</span>', unsafe_allow_html=True)
            else:
                st.markdown("_All key concepts covered!_")

        # ── PDF Download ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📥 Download Assessment Report")

        pdf_bytes = generate_pdf_report({
            "concept": selected_concept_name or reference_concept[:60],
            "language": st.session_state.detected_lang,
            "similarity": f"{st.session_state.semantic_score*100:.2f}%",
            "score": f"{st.session_state.final_score:.1f}/100",
            "tempo": f"{st.session_state.audio_features.get('tempo', 0):.1f} BPM",
            "transcript": st.session_state.user_transcript,
            "metrics": st.session_state.metrics,
            "filler_words": filler_data.get("breakdown", {}),
            "pause_ratio": st.session_state.audio_features.get("pause_ratio", 0),
            "understanding_level": st.session_state.understanding_level,
            "feedback": st.session_state.feedback,
            "waveform_y": st.session_state.waveform_y,
        })
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"VBCUA_Report_Session_{st.session_state.session_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        # ── Session History ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📜 Recent Evaluation History")
        recent = get_all_sessions(limit=5)
        if recent:
            for row in recent:
                lvl_icon = {"Strong Understanding": "🟢", "Moderate Understanding": "🟡", "Poor Understanding": "🔴"}.get(
                    row.get("understanding_level", ""), "⚪"
                )
                st.markdown(
                    f'<div class="history-row">'
                    f'<b>#{row["id"]}</b> · {row["timestamp"]} &nbsp;|&nbsp; '
                    f'<b>{row.get("concept", "N/A")}</b> &nbsp;|&nbsp; '
                    f'Score: <b>{row.get("final_score", 0):.1f}%</b> &nbsp;|&nbsp; '
                    f'{lvl_icon} {row.get("understanding_level", "N/A")} &nbsp;|&nbsp; '
                    f'Lang: {row.get("language", "N/A").upper()}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No past sessions found.")

    else:
        st.warning(
            "⚠️ No analysis data available. Run the pipeline from the sidebar first."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEVELOPERS TEAM
# ══════════════════════════════════════════════════════════════════════════════
with app_tabs[2]:
    st.markdown("## 👥 VBCUA Core Engineering Team")
    st.markdown("---")

    dev_col1, dev_col2 = st.columns(2)

    with dev_col1:
        for name, role, color in [
            ("AMRITANSH DWIVEDI", "Senior AI Lead & System Architect", "#22d3ee"),
            ("KIRTI SWARNKAR", "Core NLP Research Engineer", "#c084fc"),
            ("ANUSHIKA DUTTA", "Backend Infrastructure & API Architect", "#38bdf8"),
        ]:
            st.markdown(
                f'<div class="dev-card">'
                f'<h3 style="color:{color}; margin-bottom:4px;">{name}</h3>'
                f'<p style="color:#64748b; font-size:12px; font-weight:600; '
                f'text-transform:uppercase; letter-spacing:1px;">{role}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    with dev_col2:
        for name, role, color in [
            ("SANJANA KUMARI", "Frontend & UI/UX Experience Designer", "#fb7185"),
            ("SOURAV SHARMA", "DSP Audio Signal Engineer", "#34d399"),
        ]:
            st.markdown(
                f'<div class="dev-card">'
                f'<h3 style="color:{color}; margin-bottom:4px;">{name}</h3>'
                f'<p style="color:#64748b; font-size:12px; font-weight:600; '
                f'text-transform:uppercase; letter-spacing:1px;">{role}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("""
        <div style='text-align:center; margin-top:30px; color:#475569; font-size:13px;'>
            <p>🎤 Voice-Based Concept Understanding Analyser (VBCUA) · 2026</p>
            <p>Built with: OpenAI Whisper · Sentence-BERT · Librosa · NLTK · Streamlit · ReportLab · SQLite</p>
        </div>
    """, unsafe_allow_html=True)