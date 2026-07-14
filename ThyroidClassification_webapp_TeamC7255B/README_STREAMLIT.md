# Streamlit interface

## 1. Add the checkpoint

Place the trained file at:

```text
model/convnext_tiny_seed123_best.pt
```

## 2. Install on Windows

Open PowerShell in the project folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

## 3. Start the app

```powershell
.\run_windows.ps1
```

Open:

```text
http://localhost:8501
```

## Interface flow

1. Upload one JPG, JPEG, or PNG ultrasound image.
2. Leave **Generate explanation heatmap** enabled to create Grad-CAM.
3. Select **Analyze image**.
4. Review the final class, calibrated score, threshold chart, explanation view, and decision-factor table.
5. Use the image controls to switch between **Overlay**, **Heatmap**, and **Original**.

## Displayed model results

- Internal TN5000 test accuracy: 84.63%.
- Internal TN5000 test ROC-AUC: 0.943.
- External TN3K/TNCD accuracy: 77.53%.
- External TN3K/TNCD ROC-AUC: 0.794.
- External sensitivity: 59.0%.

The internal and external results are shown separately because the external cohort better represents generalization to a different dataset.

## Explainability warning

Grad-CAM displays areas that influenced the network output. It does not segment a nodule, identify a tumor boundary, or provide a medical diagnosis.
