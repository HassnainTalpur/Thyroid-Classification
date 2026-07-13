# Thyroid.AI Streamlit UI with real ConvNeXt-Tiny inference

This package uses the supplied Streamlit interface and replaces its simulated
hash-based score with the real ConvNeXt-Tiny seed-123 inference pipeline.

## 1. Add the checkpoint

Copy:

```text
convnext_tiny_seed123_best.pt
```

into:

```text
model/convnext_tiny_seed123_best.pt
```

The file should come from:

```text
thyroid_external_validation_outputs/
  production_training_cell13/
    convnext_tiny/
      checkpoints/
        convnext_tiny_seed123_best.pt
```

## 2. Set up on Windows

Open PowerShell in the project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

## 3. Start the UI

```powershell
.
un_windows.ps1
```

Or manually:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Streamlit normally opens the site automatically. Otherwise open:

```text
http://localhost:8501
```

## Real inference behavior

- Architecture: `timm` ConvNeXt-Tiny
- Checkpoint: seed 123
- Input: RGB, direct resize to 224 × 224
- Normalization: ImageNet mean/std
- Primary decision: raw sigmoid score >= 0.9453125
- Temperature: 2.5775904655456543
- Calibrated equivalent threshold: 0.7513148919051769

The displayed temperature-scaled malignancy score is not thresholded at 0.5.
The classification follows the validation-locked raw threshold from the study.

## Important

This is a research prototype, not a medical diagnosis or medical device. Its
external sensitivity at the locked threshold was approximately 59%, so it can
miss malignant cases.

## CPU performance

The predictor caps PyTorch CPU inference at up to four threads by default to
avoid severe oversubscription on Windows laptops. Override it when needed:

```powershell
$env:TORCH_NUM_THREADS = "2"
.\run_windows.ps1
```
