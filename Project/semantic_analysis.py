# VBCUA/backend/semantic_analysis.py
import re
import nltk
from sentence_transformers import SentenceTransformer, util

# ── NLTK Downloads (first-time only) ──────────────────────────────────────────
for _pkg in ('punkt', 'stopwords', 'punkt_tab'):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Model Loading ──────────────────────────────────────────────────────────────
_sbert_model = None

def get_sbert_model():
    """Lazily loads and caches the Sentence-BERT model."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _sbert_model

# ── English stopwords set ──────────────────────────────────────────────────────
try:
    _STOP_WORDS = set(stopwords.words('english'))
except Exception:
    _STOP_WORDS = set()

# ── Hindi stopwords set ────────────────────────────────────────────────────────
_HINDI_STOP_WORDS = {
    "है", "और", "का", "की", "के", "से", "को", "में", "पर", "हैं", "था", "थी", "थे",
    "कि", "तो", "ही", "भी", "या", "इस", "उस", "यह", "वह", "जो", "कर", "करते", "करना",
    "हो", "होता", "होती", "होते", "नहीं", "ने", "गया", "गई", "गए", "द्वारा",
    "लिए", "अपने", "अपनी", "सब", "सभी", "कुछ", "एक", "दो", "तीन", "चार",
    "कम", "ज्यादा", "अधिक", "बहुत", "तथा", "एवं", "अथवा", "लेकिन", "परंतु",
    "किंतु", "अगर", "यदि", "तो", "तब", "जब", "अब", "कब", "कहाँ", "वहाँ", "यहाँ",
    "कैसे", "क्यों", "क्या", "कौन", "किसे", "किसने", "किस"
}

# ── Filler word lists (English + Hindi) ───────────────────────────────────────
_FILLER_WORDS = [
    "um", "uh", "like", "actually", "basically",
    "you know", "so", "right", "okay", "kind of",
    "sort of", "मतलब", "यानी", "जैसे कि"
]


# ──────────────────────────────────────────────────────────────────────────────
# Similarity
# ──────────────────────────────────────────────────────────────────────────────
def get_similarity(user_text: str, reference_text: str) -> float:
    """
    Computes cosine similarity between user speech and reference concept
    using Sentence-BERT embeddings.
    Returns a float in [0.0, 1.0].
    """
    if not user_text.strip() or not reference_text.strip():
        return 0.0
    model = get_sbert_model()
    embeddings = model.encode([user_text, reference_text], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score.item())


# ──────────────────────────────────────────────────────────────────────────────
# Filler Word Detection
# ──────────────────────────────────────────────────────────────────────────────
def extract_filler_words(text: str) -> dict:
    """
    Detects filler words in transcribed text.
    Returns breakdown dict, total count, and frequency %.
    """
    found_fillers = {}
    total_count = 0

    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    total_words = max(len(words), 1)

    for filler in _FILLER_WORDS:
        count = len(re.findall(r'\b' + re.escape(filler) + r'\b', clean_text))
        if count > 0:
            found_fillers[filler] = count
            total_count += count

    return {
        "breakdown": found_fillers,
        "total_count": total_count,
        "frequency_pct": round((total_count / total_words) * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# NLTK Keyword Extraction
# ──────────────────────────────────────────────────────────────────────────────
def contains_devanagari(text: str) -> bool:
    """Returns True if the text contains Devanagari characters (Hindi)."""
    return bool(re.search(r'[\u0900-\u097f]', text))


def extract_keywords(text: str) -> list[str]:
    """
    Extracts meaningful keywords from text using NLTK tokenization
    and stopword removal. Supports both English and Hindi.
    """
    if not text.strip():
        return []
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()

    is_hindi = contains_devanagari(text)
    active_stopwords = _STOP_WORDS | _HINDI_STOP_WORDS if is_hindi else _STOP_WORDS
    min_len = 1 if is_hindi else 2

    keywords = [
        t for t in tokens
        if t.isalpha() and t not in active_stopwords and len(t) > min_len
    ]
    return list(set(keywords))


# ──────────────────────────────────────────────────────────────────────────────
# Concept Coverage Analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_coverage(user_text: str, reference_text: str) -> dict:
    """
    Compares user speech keywords against reference concept keywords.
    Returns:
      - coverage_score (float 0-1): fraction of reference keywords found
      - covered_keywords (list): reference keywords found in user speech
      - missing_keywords (list): reference keywords absent from user speech
    """
    user_keywords = set(extract_keywords(user_text))
    reference_keywords = set(extract_keywords(reference_text))

    if not reference_keywords:
        return {
            "coverage_score": 0.0,
            "covered_keywords": [],
            "missing_keywords": [],
        }

    covered = user_keywords & reference_keywords
    missing = reference_keywords - user_keywords

    coverage_score = len(covered) / len(reference_keywords)

    return {
        "coverage_score": round(coverage_score, 4),
        "covered_keywords": sorted(covered),
        "missing_keywords": sorted(missing),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Understanding Level Classification
# ──────────────────────────────────────────────────────────────────────────────
def get_understanding_level(semantic_score: float) -> str:
    """
    Maps a 0–1 semantic similarity score to a qualitative understanding label.
      ≥ 0.70 → Strong Understanding
      0.40 – 0.70 → Moderate Understanding
      < 0.40 → Poor Understanding
    """
    if semantic_score >= 0.70:
        return "Strong Understanding"
    elif semantic_score >= 0.40:
        return "Moderate Understanding"
    else:
        return "Poor Understanding"