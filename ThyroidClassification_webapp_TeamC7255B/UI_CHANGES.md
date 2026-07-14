# Visual UI and explainability update

## Main workflow

The interface now follows one clear path:

1. Upload a JPG or PNG ultrasound image.
2. Choose whether to generate a Grad-CAM explanation.
3. Click **Analyze image**.
4. Review the prediction, threshold comparison, explanation image, and concise decision factors.

## Visual additions

- Live step-by-step analysis status.
- Score-versus-threshold bar chart.
- Internal-versus-external performance chart.
- Original, heatmap, and overlay image views.
- Overlay-strength slider.
- Highlighted explanation-factor table.
- Clear success, error, loading, and model-ready messages.
- Responsive card and chart layout for desktop, tablet, and mobile widths.

## Explainability

The app now generates Grad-CAM from the final ConvNeXt spatial convolution. The map shows regions that most influenced the returned class. It also reports:

- calibrated score;
- locked threshold;
- whether the score is above or below the threshold;
- strongest response region;
- percentage of the image with high activation.

The heatmap is not a tumor segmentation mask and is not a clinical diagnosis.

## Inference

Checkpoint loading, image preprocessing, temperature scaling, and the validation-locked class threshold remain unchanged.
