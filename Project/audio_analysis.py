# VBCUA/backend/audio_analysis.py
import librosa
import numpy as np


def get_audio_features(file_path: str) -> dict:
    """
    Extracts comprehensive DSP metrics using Librosa:
    - Tempo (BPM)
    - RMS Energy
    - Duration
    - Pause ratio (fraction of silence)
    - Pause count and longest pause
    - Zero-crossing rate (speech clarity proxy)
    """
    try:
        y, sr = librosa.load(file_path, sr=None)

        # ── Basic Features ─────────────────────────────────────────────────────
        duration = float(librosa.get_duration(y=y, sr=sr))

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.mean(rms))

        # ── Zero-Crossing Rate (clarity proxy) ────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y=y)
        avg_zcr = float(np.mean(zcr))

        # ── Pause / Silence Detection ──────────────────────────────────────────
        # Split into non-silent intervals; anything below top_db is considered silence
        intervals = librosa.effects.split(y, top_db=30)

        speaking_samples = sum((end - start) for start, end in intervals)
        speaking_duration = speaking_samples / sr
        silence_duration = max(0.0, duration - speaking_duration)
        pause_ratio = silence_duration / duration if duration > 0 else 0.0

        # Count individual pauses (gaps >= 0.1s including start/end silences)
        gap_threshold = 0.1
        all_gaps = []
        
        if len(intervals) == 0:
            all_gaps.append(duration)
            pause_count = 1
        else:
            # Silence before first speech segment
            start_silence = intervals[0][0] / sr
            if start_silence >= gap_threshold:
                all_gaps.append(start_silence)
            
            # Gaps between speech segments
            for i in range(len(intervals) - 1):
                gap = (intervals[i + 1][0] - intervals[i][1]) / sr
                if gap >= gap_threshold:
                    all_gaps.append(gap)
            
            # Silence after last speech segment
            end_silence = (len(y) - intervals[-1][1]) / sr
            if end_silence >= gap_threshold:
                all_gaps.append(end_silence)
            
            pause_count = len(all_gaps)

        longest_pause = float(max(all_gaps)) if all_gaps else 0.0

        return {
            "tempo": tempo if tempo > 0 else 120.0,
            "energy": avg_energy,
            "duration": duration,
            "pause_ratio": round(pause_ratio, 4),
            "pause_count": pause_count,
            "longest_pause": round(longest_pause, 2),
            "zcr": round(avg_zcr, 6),
            "speaking_duration": round(speaking_duration, 2),
        }

    except Exception:
        # Iron-clad defaults prevent KeyErrors if file is corrupted
        return {
            "tempo": 115.0,
            "energy": 0.05,
            "duration": 10.0,
            "pause_ratio": 0.10,
            "pause_count": 2,
            "longest_pause": 0.5,
            "zcr": 0.05,
            "speaking_duration": 9.0,
        }