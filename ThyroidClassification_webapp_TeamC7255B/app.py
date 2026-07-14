"""Streamlit interface for the ConvNeXt-Tiny thyroid ultrasound classifier.

Run from the project root with:
    python -m streamlit run app.py

Expected checkpoint:
    model/convnext_tiny_seed123_best.pt
"""

from __future__ import annotations

import hashlib
import html
import io
import os
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from backend.app.inference import PredictorError, ThyroidPredictor

# -----------------------------------------------------------------------------
# PAGE AND MODEL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thyroid Ultrasound Classifier",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "ConvNeXt-Tiny"
MODEL_SEED = 123

INT_TEST_SIZE = 950
INT_ROC_AUC = 0.9432176124783129
INT_ACCURACY = (231 + 573) / INT_TEST_SIZE
INT_F1 = 0.8869969040247678

EXT_TEST_SIZE = 3493
EXT_ROC_AUC = 0.7936573958435146
EXT_ACCURACY = 0.7752648153449757
EXT_SENSITIVITY = 0.5900826446280992
EXT_SPECIFICITY = 0.8734121769601402

BASE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = BASE_DIR / "model" / "model_config.json"
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "model" / "convnext_tiny_seed123_best.pt"
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT_PATH)))


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(config_path: str, checkpoint_path: str) -> ThyroidPredictor:
    return ThyroidPredictor(config_path=config_path, checkpoint_path=checkpoint_path)


try:
    PREDICTOR = load_predictor(str(MODEL_CONFIG_PATH), str(CHECKPOINT_PATH))
    MODEL_LOAD_ERROR = None
except Exception as exc:
    PREDICTOR = None
    MODEL_LOAD_ERROR = str(exc)


# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


# -----------------------------------------------------------------------------
# VISUAL STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --bg: #081321;
            --panel: #0d1b2d;
            --panel-2: #102239;
            --panel-3: #132842;
            --border: #263e5a;
            --border-soft: rgba(148, 163, 184, 0.14);
            --text: #eef6ff;
            --muted: #9aacc2;
            --quiet: #6f829b;
            --blue: #38bdf8;
            --blue-2: #60a5fa;
            --green: #4ade80;
            --red: #fb7185;
            --amber: #fbbf24;
        }

        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 0%, rgba(14, 165, 233, 0.09), transparent 31%),
                radial-gradient(circle at 96% 88%, rgba(37, 99, 235, 0.07), transparent 34%),
                var(--bg);
            color: var(--text);
        }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container {
            max-width: 1240px;
            padding-top: 2.4rem;
            padding-bottom: 2rem;
        }

        .top-line {
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .page-title {
            color: var(--text);
            font-size: clamp(2.2rem, 4vw, 3.7rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            font-weight: 700;
            margin: 0 0 0.75rem;
        }

        .page-copy {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.65;
            max-width: 650px;
            margin-bottom: 1.2rem;
        }

        .quick-note {
            background: rgba(13, 27, 45, 0.85);
            border: 1px solid var(--border-soft);
            border-left: 3px solid var(--blue);
            border-radius: 13px;
            padding: 0.9rem 1rem;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.55;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1.1rem;
        }

        .metric-card, .tech-card {
            background: rgba(13, 27, 45, 0.88);
            border: 1px solid var(--border-soft);
            border-radius: 13px;
            padding: 0.85rem 0.9rem;
            min-width: 0;
        }

        .metric-value, .tech-value {
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.18rem;
            font-weight: 600;
        }

        .metric-label, .tech-label {
            color: var(--quiet);
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.055em;
            line-height: 1.35;
            text-transform: uppercase;
            margin-top: 0.25rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(13, 27, 45, 0.94) !important;
            border: 1px solid var(--border) !important;
            border-radius: 17px !important;
            box-shadow: 0 15px 38px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.15rem 0.25rem;
        }

        .panel-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.015em;
            margin-bottom: 0.25rem;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
            margin: 0.3rem 0 0.9rem;
        }

        .step {
            background: #0a1727;
            border: 1px solid var(--border-soft);
            border-radius: 10px;
            padding: 0.55rem 0.65rem;
            color: var(--muted);
            font-size: 0.67rem;
            line-height: 1.35;
        }

        .step strong {
            color: var(--blue);
            font-family: "IBM Plex Mono", monospace;
            margin-right: 0.3rem;
        }

        .model-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            border-radius: 10px;
            padding: 0.6rem 0.7rem;
            margin-bottom: 0.8rem;
            font-size: 0.7rem;
            font-weight: 600;
        }

        .model-status.ready {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(74, 222, 128, 0.24);
        }

        .model-status.error {
            color: #fecdd3;
            background: rgba(244, 63, 94, 0.08);
            border: 1px solid rgba(251, 113, 133, 0.25);
        }

        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: currentColor;
            margin-right: 0.4rem;
        }

        .model-meta {
            color: var(--quiet);
            font-size: 0.61rem;
            font-weight: 700;
            letter-spacing: 0.045em;
            text-transform: uppercase;
        }

        [data-testid="stFileUploader"] { background: transparent !important; }

        [data-testid="stFileUploaderDropzone"] {
            background: #091625 !important;
            border: 1px dashed #35516e !important;
            border-radius: 12px !important;
            min-height: 105px;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--blue) !important;
            background: #0b1a2c !important;
        }

        [data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
        [data-testid="stFileUploaderDropzoneInstructions"] svg {
            color: var(--blue) !important;
            fill: var(--blue) !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: #091625 !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploaderFile"] * { color: var(--text) !important; }

        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(90deg, #0284c7, #2563eb) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 9px !important;
            font-weight: 700 !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: #152a43 !important;
            color: #e8f5ff !important;
            border: 1px solid #315574 !important;
            border-radius: 9px !important;
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }

        .result-kicker {
            color: var(--quiet);
            font-size: 0.63rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .result-title {
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .prediction-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 9px;
            padding: 0.55rem 0.75rem;
            font-size: 0.76rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .prediction-badge.malignant {
            color: #fecdd3;
            background: rgba(244, 63, 94, 0.10);
            border: 1px solid rgba(251, 113, 133, 0.30);
        }

        .prediction-badge.benign {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.09);
            border: 1px solid rgba(74, 222, 128, 0.28);
        }

        .score-block {
            background: #091625;
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .score-label {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 600;
        }

        .score-value {
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.55rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        .score-sub {
            color: var(--quiet);
            font-size: 0.68rem;
            line-height: 1.45;
            margin-top: 0.15rem;
        }

        .section-title {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 700;
            margin: 0.15rem 0 0.25rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }

        .explanation-box {
            background: #091625;
            border: 1px solid var(--border-soft);
            border-left: 3px solid var(--blue);
            border-radius: 12px;
            padding: 0.82rem 0.9rem;
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.6;
            margin: 0.6rem 0 0.8rem;
        }

        .explanation-box strong { color: var(--text); }

        .factor-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 0.42rem;
            margin-top: 0.25rem;
        }

        .factor-table td {
            background: #091625;
            border-top: 1px solid var(--border-soft);
            border-bottom: 1px solid var(--border-soft);
            padding: 0.62rem 0.68rem;
            color: var(--muted);
            font-size: 0.71rem;
            vertical-align: middle;
        }

        .factor-table td:first-child {
            border-left: 1px solid var(--border-soft);
            border-radius: 9px 0 0 9px;
            color: var(--text);
            font-weight: 600;
            width: 34%;
        }

        .factor-table td:last-child {
            border-right: 1px solid var(--border-soft);
            border-radius: 0 9px 9px 0;
            text-align: right;
            font-family: "IBM Plex Mono", monospace;
            color: #cdeeff;
            width: 24%;
        }

        .tech-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
        }

        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        [data-testid="stExpander"] {
            background: #091625;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {
            color: var(--muted) !important;
        }

        [data-testid="stStatusWidget"] {
            background: #091625 !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        .footer-line {
            color: var(--quiet);
            border-top: 1px solid var(--border-soft);
            font-size: 0.69rem;
            line-height: 1.5;
            margin-top: 2rem;
            padding-top: 0.9rem;
            text-align: center;
        }

        @media (max-width: 980px) {
            .metric-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .tech-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 700px) {
            .block-container { padding: 1.3rem 0.8rem 1.8rem; }
            .page-title { font-size: 2.35rem; }
            .steps { grid-template-columns: 1fr; }
            .result-header { align-items: flex-start; flex-direction: column; }
        }

        @media (max-width: 440px) {
            .metric-row, .tech-grid { grid-template-columns: 1fr; }
            .model-status { align-items: flex-start; flex-direction: column; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def open_rgb_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise PredictorError("The uploaded file is not a readable JPG or PNG image.") from exc


def colorize_heatmap(heatmap: np.ndarray) -> Image.Image:
    values = np.clip(heatmap, 0.0, 1.0)
    grayscale = Image.fromarray((values * 255).astype(np.uint8), mode="L")
    return ImageOps.colorize(
        grayscale,
        black=(9, 22, 37),
        mid=(56, 189, 248),
        white=(251, 113, 133),
    )


def make_overlay(original: Image.Image, heatmap: np.ndarray, opacity: float) -> Image.Image:
    heatmap_image = colorize_heatmap(heatmap).resize(original.size, Image.Resampling.BILINEAR)
    base = original.convert("RGB")
    return Image.blend(base, heatmap_image, alpha=opacity)


def decision_chart(score: float, threshold: float) -> alt.Chart:
    data = pd.DataFrame(
        {
            "Measure": ["Model score", "Decision threshold"],
            "Value": [score * 100, threshold * 100],
        }
    )

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=5, size=20)
        .encode(
            x=alt.X(
                "Value:Q",
                title="Percent",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f", tickCount=5),
            ),
            y=alt.Y(
                "Measure:N",
                title=None,
                sort=["Model score", "Decision threshold"],
            ),
            color=alt.Color(
                "Measure:N",
                scale=alt.Scale(
                    domain=["Model score", "Decision threshold"],
                    range=["#38bdf8", "#64748b"],
                ),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Measure:N"), alt.Tooltip("Value:Q", format=".1f")],
        )
    )

    labels = bars.mark_text(align="left", baseline="middle", dx=5, color="#eef6ff").encode(
        text=alt.Text("Value:Q", format=".1f")
    )

    return (
        (bars + labels)
        .properties(height=95, title="Score compared with the locked threshold")
        .configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_title(color="#eef6ff", fontSize=13, anchor="start", offset=10)
        .configure_axis(
            labelColor="#9aacc2",
            titleColor="#6f829b",
            gridColor="#20364f",
            domainColor="#263e5a",
            tickColor="#263e5a",
        )
    )


def benchmark_chart() -> alt.Chart:
    data = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Cohort": "Internal", "Value": INT_ACCURACY * 100},
            {"Metric": "Accuracy", "Cohort": "External", "Value": EXT_ACCURACY * 100},
            {"Metric": "ROC-AUC", "Cohort": "Internal", "Value": INT_ROC_AUC * 100},
            {"Metric": "ROC-AUC", "Cohort": "External", "Value": EXT_ROC_AUC * 100},
        ]
    )

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset="Cohort:N",
            y=alt.Y(
                "Value:Q",
                title="Score (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(tickCount=5),
            ),
            color=alt.Color(
                "Cohort:N",
                scale=alt.Scale(
                    domain=["Internal", "External"],
                    range=["#38bdf8", "#818cf8"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Cohort:N"),
                alt.Tooltip("Value:Q", format=".1f"),
            ],
        )
    )

    labels = bars.mark_text(dy=-8, color="#eef6ff", fontSize=11).encode(
        text=alt.Text("Value:Q", format=".1f")
    )

    return (
        (bars + labels)
        .properties(height=260, title="Internal and external performance")
        .configure(background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_title(color="#eef6ff", fontSize=14, anchor="start", offset=12)
        .configure_axis(
            labelColor="#9aacc2",
            titleColor="#6f829b",
            gridColor="#20364f",
            domainColor="#263e5a",
            tickColor="#263e5a",
        )
        .configure_legend(labelColor="#9aacc2")
    )


def explanation_text(result: dict[str, Any]) -> tuple[str, float, str]:
    score = float(result["malignancy_score_temperature_scaled"])
    threshold = float(result["decision_threshold_temperature_scaled_equivalent"])
    difference = abs(score - threshold) * 100
    label = str(result["prediction"]).lower()
    relation = "above" if score >= threshold else "below"
    direction = "malignant" if score >= threshold else "benign"

    text = (
        f"The calibrated malignancy score was <strong>{score*100:.1f}%</strong>, "
        f"which is <strong>{difference:.1f} percentage points {relation}</strong> the "
        f"locked threshold of {threshold*100:.1f}%. The decision therefore falls on "
        f"the <strong>{direction}</strong> side of the threshold."
    )

    if "focus_region" in result:
        region = str(result["focus_region"]).replace("-", " ")
        text += (
            f" The Grad-CAM map shows the strongest model response in the "
            f"<strong>{html.escape(region)}</strong> part of the image."
        )

    return text, difference, label


def factor_table_html(result: dict[str, Any]) -> str:
    score = float(result["malignancy_score_temperature_scaled"])
    threshold = float(result["decision_threshold_temperature_scaled_equivalent"])
    relation = "Above" if score >= threshold else "Below"

    rows = [
        ("Calibrated score", "Model output", f"{score*100:.1f}%"),
        ("Decision threshold", "Locked on validation data", f"{threshold*100:.1f}%"),
        ("Threshold result", "Determines the returned class", relation),
    ]

    if "focus_region" in result:
        rows.extend(
            [
                (
                    "Strongest response",
                    "Grad-CAM location",
                    str(result["focus_region"]).replace("-", " ").title(),
                ),
                (
                    "High-activation area",
                    "Pixels with heatmap value ≥ 0.60",
                    f"{float(result['high_activation_coverage'])*100:.1f}%",
                ),
            ]
        )

    rendered_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(description)}</td>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for name, description, value in rows
    )
    return f'<table class="factor-table">{rendered_rows}</table>'


def reset_app() -> None:
    st.session_state.analysis_result = None
    st.session_state.upload_key += 1
    st.rerun()


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
header_left, header_right = st.columns([1.18, 0.82], gap="large")

with header_left:
    st.markdown(
        """
        <div class="top-line">ConvNeXt-Tiny · Seed 123</div>
        <div class="page-title">Thyroid ultrasound classifier</div>
        <div class="page-copy">
            Upload one B-mode ultrasound image. The app returns a prediction,
            compares the score with the locked decision threshold, and shows
            where the model responded most strongly.
        </div>
        <div class="quick-note">
            Internal accuracy: 84.6% on TN5000. External accuracy: 77.5% on
            TN3K/TNCD. Research use only; the result is not a diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{INT_ACCURACY*100:.1f}%</div>
                <div class="metric-label">Internal accuracy · n={INT_TEST_SIZE}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{EXT_ACCURACY*100:.1f}%</div>
                <div class="metric-label">External accuracy · n={EXT_TEST_SIZE:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{EXT_ROC_AUC:.3f}</div>
                <div class="metric-label">External ROC-AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{EXT_SENSITIVITY*100:.1f}%</div>
                <div class="metric-label">External sensitivity</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title">Run analysis</div>
            <div class="panel-copy">Supported files: JPG, JPEG, and PNG.</div>
            <div class="steps">
                <div class="step"><strong>01</strong>Upload image</div>
                <div class="step"><strong>02</strong>Run model</div>
                <div class="step"><strong>03</strong>Review result</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if PREDICTOR is None:
            st.markdown(
                f"""
                <div class="model-status error">
                    <div><span class="status-dot"></span>Model unavailable</div>
                    <div class="model-meta">{html.escape(MODEL_LOAD_ERROR or 'Unknown error')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="model-status ready">
                    <div><span class="status-dot"></span>Model ready</div>
                    <div class="model-meta">{str(PREDICTOR.device).upper()} · {MODEL_NAME}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        uploaded_file = st.file_uploader(
            "Upload ultrasound image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            disabled=PREDICTOR is None,
            key=f"ultrasound_upload_{st.session_state.upload_key}",
        )

        generate_heatmap = st.toggle(
            "Generate explanation heatmap",
            value=True,
            help="Grad-CAM adds one extra model pass and may take longer on CPU.",
            disabled=PREDICTOR is None,
        )

        action_col, clear_col = st.columns([2.4, 1])
        analyze_clicked = action_col.button(
            "Analyze image",
            type="primary",
            use_container_width=True,
            disabled=PREDICTOR is None or uploaded_file is None,
        )
        clear_clicked = clear_col.button(
            "Clear",
            use_container_width=True,
            disabled=uploaded_file is None and st.session_state.analysis_result is None,
        )

        if clear_clicked:
            reset_app()


# -----------------------------------------------------------------------------
# RUN ANALYSIS WITH LIVE STATUS
# -----------------------------------------------------------------------------
if analyze_clicked and uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        with st.status("Running analysis…", expanded=True) as status:
            status.write("Checking the uploaded image")
            original_image = open_rgb_image(image_bytes)

            status.write("Running ConvNeXt-Tiny")
            prediction_result = PREDICTOR.predict_bytes(image_bytes) if PREDICTOR else None
            if prediction_result is None:
                raise PredictorError("The model is not loaded.")

            if generate_heatmap:
                status.write("Generating Grad-CAM explanation")
                explanation_result = PREDICTOR.explain_bytes(image_bytes)
                prediction_result.update(
                    {
                        "heatmap": explanation_result["heatmap"],
                        "focus_region": explanation_result["focus_region"],
                        "high_activation_coverage": explanation_result[
                            "high_activation_coverage"
                        ],
                        "explanation_method": explanation_result["explanation_method"],
                    }
                )

            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.analysis_result = {
            "file_hash": file_hash,
            "filename": uploaded_file.name,
            "image_bytes": image_bytes,
            "result": prediction_result,
        }
        st.success("Analysis completed successfully.")
    except PredictorError as exc:
        st.session_state.analysis_result = None
        st.error(f"Analysis failed: {exc}")
    except Exception as exc:
        st.session_state.analysis_result = None
        st.error(f"Unexpected error: {exc}")


# Hide a stale result when a different file has been selected but not analyzed.
if uploaded_file is not None and st.session_state.analysis_result is not None:
    current_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if current_hash != st.session_state.analysis_result["file_hash"]:
        st.session_state.analysis_result = None
        st.info("A new image is selected. Click “Analyze image” to run the model.")


# -----------------------------------------------------------------------------
# RESULT PANEL
# -----------------------------------------------------------------------------
analysis = st.session_state.analysis_result
if analysis is not None:
    result = analysis["result"]
    original = open_rgb_image(analysis["image_bytes"])
    label = str(result["prediction"]).upper()
    score = float(result["malignancy_score_temperature_scaled"])
    threshold = float(result["decision_threshold_temperature_scaled_equivalent"])
    badge_class = "malignant" if label == "MALIGNANT" else "benign"

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="result-header">
                <div>
                    <div class="result-kicker">Final result</div>
                    <div class="result-title">{html.escape(analysis['filename'])}</div>
                </div>
                <div class="prediction-badge {badge_class}">{label.title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        visual_col, summary_col = st.columns([1.05, 0.95], gap="large")

        with visual_col:
            heatmap = result.get("heatmap")
            if heatmap is not None:
                view_mode = st.radio(
                    "Image view",
                    ["Overlay", "Heatmap", "Original"],
                    horizontal=True,
                    label_visibility="visible",
                )
                opacity = st.slider(
                    "Overlay strength",
                    min_value=0.15,
                    max_value=0.80,
                    value=0.45,
                    step=0.05,
                    disabled=view_mode != "Overlay",
                )

                if view_mode == "Overlay":
                    display_image = make_overlay(original, heatmap, opacity)
                    caption = "Ultrasound with Grad-CAM overlay"
                elif view_mode == "Heatmap":
                    display_image = colorize_heatmap(heatmap)
                    caption = "Grad-CAM response map"
                else:
                    display_image = original
                    caption = "Original ultrasound"
            else:
                display_image = original
                caption = "Original ultrasound"

            st.image(display_image, caption=caption, use_container_width=True)

        with summary_col:
            st.markdown(
                f"""
                <div class="score-block">
                    <div class="score-label">Calibrated malignancy score</div>
                    <div class="score-value">{score*100:.1f}%</div>
                    <div class="score-sub">Decision threshold: {threshold*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.altair_chart(decision_chart(score, threshold), use_container_width=True)

            explanation, _difference, _label = explanation_text(result)
            st.markdown(
                f'<div class="explanation-box"><strong>What determined the result</strong><br>{explanation}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Explanation factors</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">These values explain the model output. They are not clinical ultrasound features or a lesion segmentation.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(factor_table_html(result), unsafe_allow_html=True)

        if result.get("heatmap") is not None:
            st.warning(
                "The heatmap shows regions that influenced this model output. "
                "It does not identify a tumor boundary and should not be used as a diagnosis."
            )

        with st.expander("Model performance chart"):
            st.altair_chart(benchmark_chart(), use_container_width=True)
            st.caption(
                "Internal results are from the locked TN5000 test split. External results "
                "are from the independent TN3K/TNCD cohort."
            )

        with st.expander("Technical details"):
            raw_score = float(result["malignancy_score_raw"])
            raw_threshold = float(result["decision_threshold_raw"])
            device_name = str(result["device"]).upper()
            input_size = f"{int(result['input_width'])} × {int(result['input_height'])}"

            st.markdown(
                f"""
                <div class="tech-grid">
                    <div class="tech-card">
                        <div class="tech-value">{raw_score:.3f}</div>
                        <div class="tech-label">Raw score</div>
                    </div>
                    <div class="tech-card">
                        <div class="tech-value">{raw_threshold:.3f}</div>
                        <div class="tech-label">Raw threshold</div>
                    </div>
                    <div class="tech-card">
                        <div class="tech-value">{device_name}</div>
                        <div class="tech-label">Inference device</div>
                    </div>
                    <div class="tech-card">
                        <div class="tech-value">{input_size}</div>
                        <div class="tech-label">Original image size</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                f"Internal: accuracy {INT_ACCURACY*100:.2f}%, ROC-AUC {INT_ROC_AUC:.3f}, "
                f"F1 {INT_F1:.3f}. External: accuracy {EXT_ACCURACY*100:.2f}%, "
                f"ROC-AUC {EXT_ROC_AUC:.3f}, sensitivity {EXT_SENSITIVITY*100:.1f}%, "
                f"specificity {EXT_SPECIFICITY*100:.1f}%."
            )

st.markdown(
    """
    <div class="footer-line">
        Research prototype for thyroid ultrasound classification. Results require clinical review.
    </div>
    """,
    unsafe_allow_html=True,
)
