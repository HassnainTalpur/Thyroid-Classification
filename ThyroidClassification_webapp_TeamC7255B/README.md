# ConvNeXt-Tiny thyroid ultrasound classifier

A local Streamlit interface for the trained ConvNeXt-Tiny seed-123 checkpoint.

## Included

- Real ConvNeXt-Tiny inference.
- ImageNet preprocessing at 224 × 224.
- Temperature-scaled malignancy score.
- Validation-locked decision threshold.
- Internal and external performance display.
- Score-versus-threshold chart.
- Internal-versus-external benchmark chart.
- Grad-CAM heatmap and overlay.
- Concise natural-language explanation.
- Upload, analysis, clear, view, and opacity controls.
- Loading, success, and error status messages.

## Checkpoint

Copy this trained checkpoint into the project:

```text
model/convnext_tiny_seed123_best.pt
```

## Windows setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
.\run_windows.ps1
```

Then open:

```text
http://localhost:8501
```

## Manual start

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Do not start the Streamlit file with `python app.py`.

## Model decision

The class decision is based on the raw sigmoid score and the validation-locked raw threshold stored in `model/model_config.json`. The interface displays the mathematically equivalent temperature-scaled score and calibrated threshold for readability.

## Explainability

The Grad-CAM module uses the final spatial convolution of ConvNeXt-Tiny. The resulting image indicates model response, not a medical lesion boundary. It is intended only to help inspect the model output.

## Warning

This project is a research prototype. It must not be used as a standalone diagnosis or clinical decision system.
