# Model card: ConvNeXt-Tiny thyroid ultrasound classifier

## Intended use

Research demonstration of image-level benign-versus-malignant classification for thyroid ultrasound images.

## Model

- `timm.create_model("convnext_tiny", pretrained=False, num_classes=1, in_chans=3)`
- Selected checkpoint: seed 123, best epoch 22
- Selection: highest TN5000 validation ROC-AUC among the five ConvNeXt-Tiny seeds
- Input size: 224 × 224 RGB
- Output: one logit converted with sigmoid

## Operating point

The primary class decision uses the raw sigmoid score and the TN5000-validation Youden threshold `0.9453125`. Temperature scaling (`T = 2.5775904655456543`) is exposed only as an additional calibrated score.

## External evaluation

On TN3K/TNCD (3,493 images), seed 123 recorded ROC-AUC 0.793657, sensitivity 0.590083, specificity 0.873412, and F1 0.645278.

## Limitations

Dataset shift reduced performance relative to TN5000 internal testing. The locked threshold missed 496 of 1,210 malignant external images in the recorded evaluation. The model must not be used as an autonomous diagnostic tool.
