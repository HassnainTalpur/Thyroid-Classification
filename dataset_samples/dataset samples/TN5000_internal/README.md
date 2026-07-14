# TN5000 internal-dataset samples

This folder contains five representative ultrasound images from the TN5000 dataset used for internal model development and evaluation.

All five supplied examples have class ID `1`, corresponding to **malignant** in the project label mapping.

Only the images and a compact manifest are included here. The original Pascal VOC XML annotations and complete train/validation/test list files were intentionally excluded because they are not required for the repository sample gallery.

## Files

- `images/` — five ultrasound examples
- `manifest.csv` — filename, class label, original split, and bounding-box coordinates

## Preview

<p>
  <img src="./images/004996.jpg" width="240" alt="TN5000 sample 004996">
  <img src="./images/004997.jpg" width="240" alt="TN5000 sample 004997">
  <img src="./images/004998.jpg" width="240" alt="TN5000 sample 004998">
</p>

<p>
  <img src="./images/004999.jpg" width="240" alt="TN5000 sample 004999">
  <img src="./images/005000.jpg" width="240" alt="TN5000 sample 005000">
</p>

These images are included only as small visual examples. The full dataset is not redistributed in this repository. Consult the original TN5000 source and its usage terms before reuse.
