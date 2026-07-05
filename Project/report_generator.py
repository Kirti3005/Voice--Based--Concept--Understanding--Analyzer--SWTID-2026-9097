# VBCUA/backend/report_generator.py
import io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Font Registration ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
FONT_PATH = os.path.join(_ROOT_DIR, "NotoSansDevanagari-Regular.ttf")
FONT_NAME = 'Helvetica'

if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont('NotoSansDevanagari', FONT_PATH))
        FONT_NAME = 'NotoSansDevanagari'
    except Exception:
        FONT_NAME = 'Helvetica'


def _build_waveform_image(waveform_y: np.ndarray) -> io.BytesIO:
    """Renders a waveform matplotlib figure into a BytesIO PNG buffer."""
    fig, ax = plt.subplots(figsize=(6.5, 1.6))
    ax.plot(waveform_y, color='#1d70b8', linewidth=0.8, alpha=0.9)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_pdf_report(data: dict) -> bytes:
    """
    Generates a complete, 2-page binary PDF assessment report.
    Supports English and Hindi (Devanagari) transcript rendering.

    Expected keys in `data`:
        concept (str): Reference concept name
        language (str): Detected language code
        similarity (str): Formatted similarity percentage
        score (str): Overall score string
        tempo (str): BPM string
        transcript (str): Full transcribed text
        metrics (dict): All sub-metric scores
        filler_words (dict): Filler word breakdown
        pause_ratio (float): Pause ratio 0-1
        understanding_level (str): Strong / Moderate / Poor
        feedback (str): Qualitative feedback text
        waveform_y (np.ndarray | None): Downsampled waveform array
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )

    story = []
    styles = getSampleStyleSheet()

    # ── Typography Styles ──────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1d70b8'),
        spaceAfter=16,
        alignment=0
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#111827'),
        spaceBefore=16,
        spaceAfter=8
    )
    lbl_style = ParagraphStyle(
        'RowLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#111827')
    )
    val_style = ParagraphStyle(
        'RowValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151')
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8
    )
    # Transcript uses Hindi font when detected
    detected_lang = data.get('language', 'en').lower()
    transcript_font = FONT_NAME if detected_lang == 'hi' else 'Helvetica'
    transcript_style = ParagraphStyle(
        'TranscriptStyle',
        parent=styles['Normal'],
        fontName=transcript_font,
        fontSize=11,
        leading=17,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=10
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#4b5563'),
        leftIndent=15,
        spaceAfter=4
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Core Summary
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Voice-Based Concept Understanding Analyser Report", title_style))

    # Understanding Level Badge
    level = data.get('understanding_level', 'N/A')
    level_colors = {
        'Strong Understanding': '#15803d',
        'Moderate Understanding': '#b45309',
        'Poor Understanding': '#dc2626'
    }
    level_color = level_colors.get(level, '#374151')
    story.append(Paragraph(
        f"<font color='{level_color}'><b>Understanding Level: {level}</b></font>",
        ParagraphStyle('LevelBadge', parent=styles['Normal'], fontName='Helvetica-Bold',
                       fontSize=13, leading=18, spaceAfter=14)
    ))

    # Summary Table
    concept = data.get('concept', 'N/A')
    concept_display = (concept[:55] + '...') if len(concept) > 55 else concept
    summary_data = [
        [Paragraph('<b>Evaluated Concept:</b>', lbl_style), Paragraph(concept_display, val_style)],
        [Paragraph('<b>Detected Language:</b>', lbl_style), Paragraph(data.get('language', 'N/A').upper(), val_style)],
        [Paragraph('<b>Semantic Similarity:</b>', lbl_style), Paragraph(data.get('similarity', 'N/A'), val_style)],
        [Paragraph('<b>Overall Score:</b>', lbl_style), Paragraph(data.get('score', 'N/A'), val_style)],
        [Paragraph('<b>Speech Pace:</b>', lbl_style), Paragraph(data.get('tempo', 'N/A'), val_style)],
        [Paragraph('<b>Pause Ratio:</b>', lbl_style),
         Paragraph(f"{data.get('pause_ratio', 0.0)*100:.1f}%", val_style)],
    ]
    t_summary = Table(summary_data, colWidths=[155, 345])
    t_summary.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9fafb')),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 14))

    # Transcript
    story.append(Paragraph("Spoken Transcription Output", h2_style))
    transcript_text = data.get('transcript', 'No transcription available.')
    story.append(Paragraph(f'"{transcript_text}"', transcript_style))
    story.append(Spacer(1, 12))

    # Waveform Image
    waveform_y = data.get('waveform_y', None)
    if waveform_y is not None and len(waveform_y) > 0:
        story.append(Paragraph("Audio Signal Waveform", h2_style))
        wf_buf = _build_waveform_image(waveform_y)
        story.append(Image(wf_buf, width=470, height=110))

    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Detailed Metrics & Feedback
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Voice-Based Concept Understanding Analyser Report", title_style))
    story.append(Spacer(1, 5))

    # Metrics Breakdown Table
    story.append(Paragraph("Assessment Metrics Breakdown", h2_style))
    metric_labels = {
        "semantic": "Semantic Similarity Index",
        "coverage": "Conceptual Keyword Coverage",
        "fluency": "Speech Fluency",
        "confidence": "Vocal Confidence",
        "pause": "Pause Control Score",
        "filler": "Filler Word Mitigation",
        "communication": "Communication Score",
        "quality": "Audio Quality",
    }
    metrics = data.get('metrics', {})
    breakdown_data = [[
        Paragraph('<b>Metric</b>', lbl_style),
        Paragraph('<b>Score (%)</b>', lbl_style)
    ]]
    for key, val in metrics.items():
        label = metric_labels.get(key, key.capitalize())
        breakdown_data.append([
            Paragraph(f'<b>{label}</b>', lbl_style),
            Paragraph(f'{val:.2f}%', val_style)
        ])
    t_breakdown = Table(breakdown_data, colWidths=[290, 210])
    t_breakdown.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
    ]))
    story.append(t_breakdown)
    story.append(Spacer(1, 14))

    # Filler Words
    filler_words = data.get('filler_words', {})
    if filler_words:
        story.append(Paragraph("Filler Word Analysis", h2_style))
        filler_data = [[Paragraph('<b>Filler Word</b>', lbl_style), Paragraph('<b>Count</b>', lbl_style)]]
        for word, count in filler_words.items():
            filler_data.append([Paragraph(word, val_style), Paragraph(str(count), val_style)])
        t_filler = Table(filler_data, colWidths=[250, 250])
        t_filler.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fef3c7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fcd34d')),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        story.append(t_filler)
        story.append(Spacer(1, 14))

    # Qualitative Feedback
    story.append(Paragraph("Qualitative Feedback Summary", h2_style))
    feedback = data.get('feedback', 'No feedback available.')
    story.append(Paragraph(feedback, body_style))
    story.append(Spacer(1, 12))

    # Improvement Tips
    story.append(Paragraph("Targeted Improvement Suggestions", h2_style))
    story.append(Paragraph("• <b>Technical Terminology:</b> Include core topic-specific keywords in your explanation.", bullet_style))
    story.append(Paragraph("• <b>Pacing:</b> Aim for a steady speech pace (100–130 BPM) for clarity.", bullet_style))
    story.append(Paragraph("• <b>Reduce Fillers:</b> Practice pausing deliberately instead of using filler words.", bullet_style))
    story.append(Paragraph("• <b>Pause Control:</b> Use intentional pauses to emphasize key concepts.", bullet_style))
    story.append(Spacer(1, 30))

    # Footer / Signature
    sig_data = [[
        Paragraph('<b>Report ID:</b> VBCUA-2026-TRACK-X9', val_style),
        Paragraph('<b>Verified & Approved By:</b><br/>'
                  '<font color="#1d70b8"><b>AMRITANSH DWIVEDI</b></font><br/>'
                  'Lead Core System Architect, VBCUA Group', lbl_style)
    ]]
    t_sig = Table(sig_data, colWidths=[240, 260])
    t_sig.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEABOVE', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(t_sig)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()