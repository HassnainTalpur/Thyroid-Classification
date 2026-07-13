"""
Thyroid ultrasound nodule classifier.

Run from the project root with:
    python -m streamlit run app.py

The application loads the trained ConvNeXt-Tiny checkpoint from:
    model/convnext_tiny_seed123_best.pt
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from backend.app.inference import PredictorError, ThyroidPredictor

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thyroid Ultrasound Classifier",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "ConvNeXt-Tiny"
MODEL_SEED = 123

# Locked TN5000 internal test, deployed seed-123 checkpoint (n = 950).
INT_TEST_SIZE = 950
INT_ROC_AUC = 0.9432176124783129
INT_ACCURACY = (231 + 573) / INT_TEST_SIZE
INT_F1 = 0.8869969040247678

# Independent TN3K/TNCD external test, deployed seed-123 checkpoint (n = 3,493).
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
# VISUAL STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --bg: #091423;
            --panel: #0f1d30;
            --panel-2: #12233a;
            --border: #243b57;
            --border-soft: rgba(148, 163, 184, 0.14);
            --text: #f1f5f9;
            --muted: #91a3ba;
            --quiet: #61758f;
            --accent: #38bdf8;
            --accent-2: #60a5fa;
            --green: #4ade80;
            --red: #fb7185;
        }

        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 0%, rgba(14, 165, 233, 0.10), transparent 34%),
                radial-gradient(circle at 92% 88%, rgba(37, 99, 235, 0.08), transparent 36%),
                var(--bg);
            color: var(--text);
        }

        #MainMenu, footer, header { visibility: hidden; }

        .block-container {
            max-width: 1220px;
            padding-top: 3.2rem;
            padding-bottom: 2.2rem;
        }

        /* Main copy */
        .brand-line {
            color: var(--accent);
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2.5rem, 4vw, 4rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            font-weight: 700;
            max-width: 650px;
            margin: 0 0 1rem;
        }

        .hero-title span { color: var(--accent); }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
            max-width: 590px;
            margin-bottom: 1.4rem;
        }

        .summary-note {
            background: rgba(15, 29, 48, 0.78);
            border: 1px solid var(--border-soft);
            border-left: 3px solid var(--accent);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            max-width: 650px;
        }

        .summary-note strong {
            color: var(--text);
            display: block;
            font-size: 0.86rem;
            margin-bottom: 0.3rem;
        }

        .summary-note span {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.55;
        }

        /* Metrics */
        .stat-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.72rem;
            margin-top: 1.35rem;
            max-width: 650px;
        }

        .stat-card,
        .detail-card {
            background: rgba(15, 29, 48, 0.80);
            border: 1px solid var(--border-soft);
            border-radius: 13px;
            padding: 0.9rem 0.95rem;
            min-width: 0;
        }

        .stat-value,
        .detail-value {
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.22rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .stat-label,
        .detail-label {
            color: var(--quiet);
            font-size: 0.63rem;
            font-weight: 700;
            letter-spacing: 0.055em;
            line-height: 1.35;
            text-transform: uppercase;
            margin-top: 0.28rem;
        }

        /* Real Streamlit bordered containers. These replace the broken HTML
           wrappers that appeared as empty white bars. */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 29, 48, 0.92) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.22);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.25rem 0.3rem;
        }

        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.25rem;
        }

        .panel-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: -0.015em;
        }

        .model-chip {
            color: #b8e7ff;
            background: rgba(14, 165, 233, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.26);
            border-radius: 999px;
            padding: 0.28rem 0.58rem;
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .panel-copy {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            margin-bottom: 0.95rem;
        }

        .model-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            border-radius: 11px;
            padding: 0.62rem 0.75rem;
            margin-bottom: 0.9rem;
            font-size: 0.72rem;
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

        .use-note {
            color: var(--quiet);
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        /* File uploader — dark, integrated, no white block. */
        [data-testid="stFileUploader"] {
            background: transparent !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #0b1728 !important;
            border: 1px dashed #35506e !important;
            border-radius: 13px !important;
            min-height: 116px;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent) !important;
            background: #0d1b2e !important;
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: var(--muted) !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] svg {
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }

        [data-testid="stFileUploaderFile"] {
            background: #0b1728 !important;
            border: 1px solid var(--border) !important;
            border-radius: 11px !important;
        }

        [data-testid="stFileUploaderFile"] * {
            color: var(--text) !important;
        }

        [data-testid="stBaseButton-secondary"] {
            background: #17304d !important;
            color: #e8f5ff !important;
            border: 1px solid #315574 !important;
            border-radius: 9px !important;
        }

        [data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--accent) !important;
        }

        /* Result */
        .result-kicker {
            color: var(--quiet);
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .result-title {
            color: var(--text);
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .prediction-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 10px;
            padding: 0.6rem 0.82rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.015em;
            margin-bottom: 0.9rem;
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

        .score-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 600;
            margin: 0.15rem 0 0.35rem;
        }

        .score-value {
            color: var(--text);
            font-family: "IBM Plex Mono", monospace;
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }

        .result-note {
            color: var(--quiet);
            font-size: 0.72rem;
            line-height: 1.5;
            margin-top: 0.9rem;
        }

        .detail-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.68rem;
            margin-top: 0.25rem;
        }

        /* Progress and image */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }

        .stProgress > div > div > div {
            background: #20344d;
        }

        [data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid var(--border);
        }

        /* Expander — technical information stays available without clutter. */
        [data-testid="stExpander"] {
            background: #0b1728;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {
            color: var(--muted) !important;
        }

        .footer-line {
            color: var(--quiet);
            border-top: 1px solid var(--border-soft);
            font-size: 0.7rem;
            line-height: 1.55;
            margin-top: 2.4rem;
            padding-top: 1rem;
            text-align: center;
        }

        @media (max-width: 1050px) {
            .stat-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 760px) {
            .block-container { padding: 1.4rem 0.9rem 2rem; }
            .hero-title { font-size: 2.45rem; }
            .hero-copy { font-size: 0.94rem; }
            .stat-strip { margin-bottom: 1.5rem; }
            .panel-head { align-items: flex-start; }
        }

        @media (max-width: 450px) {
            .stat-strip,
            .detail-grid { grid-template-columns: 1fr; }
            .model-chip { display: none; }
            .model-status { align-items: flex-start; flex-direction: column; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_inference(image_bytes: bytes):
    if PREDICTOR is None:
        raise PredictorError(MODEL_LOAD_ERROR or "The model is not loaded.")

    result = PREDICTOR.predict_bytes(image_bytes)
    label = str(result["prediction"]).upper()
    calibrated_score = float(result["malignancy_score_temperature_scaled"])
    return label, calibrated_score, result


# -----------------------------------------------------------------------------
# PAGE CONTENT
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.08, 0.92], gap="large")

with left_col:
    st.markdown(
        """
        <div class="brand-line">ConvNeXt-Tiny · Seed 123</div>
        <div class="hero-title">Thyroid nodule<br><span>classification</span></div>
        <div class="hero-copy">
            Upload a B-mode ultrasound image to classify the nodule as benign or
            malignant. The model was evaluated on both an internal test split and
            an independent external cohort.
        </div>
        <div class="summary-note">
            <strong>Validation summary</strong>
            <span>
                Internal accuracy: 84.6% on TN5000. External accuracy: 77.5% on
                TN3K/TNCD. This is a research prototype and not a clinical diagnosis.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stat-strip">
            <div class="stat-card">
                <div class="stat-value">{INT_ACCURACY*100:.1f}%</div>
                <div class="stat-label">Internal accuracy · n={INT_TEST_SIZE}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{EXT_ACCURACY*100:.1f}%</div>
                <div class="stat-label">External accuracy · n={EXT_TEST_SIZE:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{INT_ROC_AUC:.3f}</div>
                <div class="stat-label">Internal ROC-AUC</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{EXT_ROC_AUC:.3f}</div>
                <div class="stat-label">External ROC-AUC</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="panel-head">
                <div class="panel-title">Analyze an ultrasound image</div>
                <div class="model-chip">{MODEL_NAME}</div>
            </div>
            <div class="panel-copy">JPG or PNG. The image is resized to 224 × 224 before inference.</div>
            """,
            unsafe_allow_html=True,
        )

        if PREDICTOR is None:
            st.markdown(
                f"""
                <div class="model-status error">
                    <div><span class="status-dot"></span>Model unavailable</div>
                    <div class="use-note">{MODEL_LOAD_ERROR}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="model-status ready">
                    <div><span class="status-dot"></span>Model ready · {str(PREDICTOR.device).upper()}</div>
                    <div class="use-note">Research use only</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        uploaded_file = st.file_uploader(
            "Upload ultrasound image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            disabled=PREDICTOR is None,
        )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        try:
            with st.spinner("Analyzing image…"):
                label, malignancy_score, inference_result = run_inference(image_bytes)
        except PredictorError as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            pil_image = None

        with st.container(border=True):
            st.markdown(
                '<div class="result-kicker">Result</div>'
                '<div class="result-title">Model prediction</div>',
                unsafe_allow_html=True,
            )

            image_col, result_col = st.columns([0.95, 1.05], gap="medium")

            with image_col:
                if pil_image is not None:
                    st.image(pil_image, use_container_width=True)

            with result_col:
                badge_class = "malignant" if label == "MALIGNANT" else "benign"
                st.markdown(
                    f'<div class="prediction-badge {badge_class}">{label.title()}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="score-label">Temperature-scaled malignancy score</div>',
                    unsafe_allow_html=True,
                )
                st.progress(float(malignancy_score))
                st.markdown(
                    f'<div class="score-value">{malignancy_score*100:.1f}%</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="result-note">The class decision uses the validation-locked raw threshold. This output is not a diagnosis.</div>',
                    unsafe_allow_html=True,
                )

            raw_score = float(inference_result["malignancy_score_raw"])
            raw_threshold = float(inference_result["decision_threshold_raw"])
            calibrated_threshold = float(
                inference_result["decision_threshold_temperature_scaled_equivalent"]
            )
            device_name = str(inference_result["device"]).upper()

            with st.expander("Technical details"):
                st.markdown(
                    f"""
                    <div class="detail-grid">
                        <div class="detail-card">
                            <div class="detail-value">{raw_score:.3f}</div>
                            <div class="detail-label">Raw score</div>
                        </div>
                        <div class="detail-card">
                            <div class="detail-value">{raw_threshold:.3f}</div>
                            <div class="detail-label">Raw threshold</div>
                        </div>
                        <div class="detail-card">
                            <div class="detail-value">{calibrated_threshold:.3f}</div>
                            <div class="detail-label">Calibrated threshold</div>
                        </div>
                        <div class="detail-card">
                            <div class="detail-value">{device_name}</div>
                            <div class="detail-label">Device</div>
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
