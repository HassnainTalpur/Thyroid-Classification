# Preferred UI

Use `README_STREAMLIT.md` to run the supplied Thyroid.AI Streamlit interface with real inference.

# ConvNeXt-Tiny thyroid ultrasound web deployment

This package deploys **only the ConvNeXt-Tiny model** from the supplied notebook as a single FastAPI website.

## Model selected

- Architecture: `timm` ConvNeXt-Tiny
- Seed: `123`
- Best training checkpoint epoch: `22`
- Input: RGB image resized directly to `224 × 224`
- Normalization: ImageNet mean/std
- Primary decision score: raw sigmoid output
- Locked TN5000 validation threshold: `0.9453125`
- Temperature fitted on TN5000 validation: `2.5775904655456543`

Seed 123 is the correct single-checkpoint deployment choice because it had the highest TN5000 validation ROC-AUC among the five ConvNeXt-Tiny runs (`0.962261`). It also recorded the highest ConvNeXt-Tiny external ROC-AUC (`0.793657`), but external results were not needed to justify the selection.

The notebook used seed 2024 for some manuscript figures only because its external AUC was closest to the five-seed median; that is a representative-seed rule, not a best-checkpoint deployment rule.

## Add the trained checkpoint

The uploaded notebook contains code and recorded outputs, but not the checkpoint bytes. Copy this file from Google Drive:

```text
thyroid_external_validation_outputs/
  production_training_cell13/
    convnext_tiny/
      checkpoints/
        convnext_tiny_seed123_best.pt
```

Place it here:

```text
model/convnext_tiny_seed123_best.pt
```

Do not use the TUS-DGCA-Net checkpoint. The backend checks the architecture metadata when available and loads the state dictionary strictly.

## Run without Docker

From the project root:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

Health check:

```text
http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Run with Docker

After placing the checkpoint in `model/`:

```bash
docker compose up --build
```

Open `http://localhost:8000`.

## Deploy to a server

Deploy the repository using `backend/Dockerfile`, expose port `8000`, and ensure the checkpoint is included securely at build time or mounted at runtime. For a runtime-mounted checkpoint, set:

```text
MODEL_CHECKPOINT=/absolute/path/convnext_tiny_seed123_best.pt
```

The default image is CPU-oriented. A GPU deployment requires a CUDA-compatible PyTorch image and `FORCE_CPU=0`.

## API response

`POST /predict` accepts one image and returns:

- benign/malignant research classification
- raw malignancy model score
- temperature-scaled score
- raw decision threshold
- model name and seed

The classification follows the notebook's primary operating point: **raw score ≥ 0.9453125**. Temperature scaling is returned for calibration display but does not alter the primary decision.

## Recorded performance for seed 123

TN5000 validation:

- ROC-AUC: `0.962261`
- Sensitivity: `0.886111`
- Specificity: `0.922414`
- F1: `0.927326`

External TN3K/TNCD, 3,493 images:

- ROC-AUC: `0.793657`
- PR-AUC: `0.712238`
- Accuracy: `0.775265`
- Sensitivity: `0.590083`
- Specificity: `0.873412`
- F1: `0.645278`

## Safety and privacy

This is a research prototype, not a medical device or diagnosis. At the locked threshold, external sensitivity was about 59%, so malignant cases can be missed. Do not retain uploads by default, do not log patient-identifiable filenames, use HTTPS in production, and validate on a new untouched local cohort before any consequential use.
