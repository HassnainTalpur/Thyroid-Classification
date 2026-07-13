from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import PredictorError, ThyroidPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_PATH = Path(os.getenv("MODEL_CONFIG", BASE_DIR / "model" / "model_config.json"))
CHECKPOINT_PATH = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        BASE_DIR / "model" / "convnext_tiny_seed123_best.pt",
    )
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
SUPPORTED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

app = FastAPI(
    title="ConvNeXt-Tiny Thyroid Ultrasound Research Classifier",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_predictor: ThyroidPredictor | None = None
_startup_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global _predictor, _startup_error
    try:
        _predictor = ThyroidPredictor(CONFIG_PATH, CHECKPOINT_PATH)
        _startup_error = None
    except Exception as exc:
        _predictor = None
        _startup_error = str(exc)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if _predictor is not None else "model_not_loaded",
        "model": "ConvNeXt-Tiny",
        "selected_seed": 123,
        "checkpoint": str(CHECKPOINT_PATH),
        "error": _startup_error,
    }


@app.get("/model-info")
def model_info() -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Model is not loaded.")
    return _predictor.config


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail=_startup_error or "Model is not loaded.")
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WEBP, BMP, or TIFF image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")

    try:
        return _predictor.predict_bytes(data)
    except PredictorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
