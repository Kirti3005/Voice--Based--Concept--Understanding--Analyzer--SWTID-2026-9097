# VBCUA/backend/speech_to_text.py
import os
import sys

# ── Inject ffmpeg into PATH BEFORE importing whisper ──────────────────────────
# Whisper calls ffmpeg via subprocess.run(["ffmpeg", ...]) at transcription time.
# We must ensure the ffmpeg binary is on PATH before the call.
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "")
_FFMPEG_KNOWN_DIRS = [
    # WinGet Gyan.FFmpeg install (bin subfolder with real exe)
    os.path.join(_LOCALAPPDATA, r"Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"),
    # WinGet Links folder (stable shim location — survives version upgrades)
    os.path.join(_LOCALAPPDATA, r"Microsoft\WinGet\Links"),
    # Common manual install locations
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for _d in _FFMPEG_KNOWN_DIRS:
    if os.path.isdir(_d) and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

# Also try imageio-ffmpeg bundled binary
try:
    import imageio_ffmpeg as _iio_ffmpeg
    _iio_dir = os.path.dirname(_iio_ffmpeg.get_ffmpeg_exe())
    if _iio_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _iio_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import whisper  # imported AFTER PATH is set

_whisper_model = None

def get_whisper_model():
    """Lazily loads and caches the Whisper 'base' model to prevent repeated loading overhead."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(audio_path: str, language_choice: str = "Auto-Detect") -> dict:
    """
    Highly resilient dynamic transcriber optimized for Hindi Devanagari script.
    Uses strict token beam constraints to prevent empty text outputs.
    Requires ffmpeg to be accessible in PATH (injected above automatically).
    """
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "detected_language": "en",
            "text": "",
            "error": f"Audio file not found at path: {audio_path}"
        }

    try:
        model = get_whisper_model()

        choice_mapping = {
            "English": "en",
            "Hindi": "hi"
        }
        forced_lang_code = choice_mapping.get(language_choice, None)

        decoding_options = {
            "task": "transcribe",
            "temperature": 0.0,
            "no_speech_threshold": 0.8,   # Relaxed: captures faint/accented speech
            "logprob_threshold": -2.0,    # Relaxed: avoids early bailout on quiet audio
        }

        # --- PASS 1: Primary Transcription ---
        if forced_lang_code:
            decoding_options["language"] = forced_lang_code
            result = model.transcribe(audio_path, **decoding_options)
            detected_lang = forced_lang_code
        else:
            result = model.transcribe(audio_path, **decoding_options)
            detected_lang = result.get("language", "en")

        transcription_text = result.get("text", "").strip()

        # --- PASS 2: Hindi Fallback if output is blank ---
        if not transcription_text:
            try:
                fallback_result = model.transcribe(audio_path, task="transcribe",
                                                   language="hi", temperature=0.2,
                                                   no_speech_threshold=0.5)
                fallback_text = fallback_result.get("text", "").strip()
                if fallback_text:
                    transcription_text = fallback_text
                    detected_lang = "hi"
            except Exception:
                pass

        if not transcription_text:
            return {
                "success": False,
                "detected_language": "en",
                "text": "No speech segment could be captured.",
                "error": "Both transcription passes returned empty output."
            }

        return {
            "success": True,
            "detected_language": detected_lang,
            "text": transcription_text,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "detected_language": "en",
            "text": "",
            "error": f"Transcription engine failure: {str(e)}"
        }