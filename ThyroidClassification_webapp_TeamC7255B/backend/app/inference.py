from __future__ import annotations

import io
import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile, UnidentifiedImageError

from .model_def import create_model

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000


class PredictorError(RuntimeError):
    """Raised when the model or an uploaded image cannot be processed safely."""


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class ThyroidPredictor:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.config_path.exists():
            raise PredictorError(f"Model config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise PredictorError(f"Checkpoint not found: {self.checkpoint_path}")

        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1"
            else "cpu"
        )

        if self.device.type == "cpu":
            default_threads = min(4, os.cpu_count() or 1)
            cpu_threads = max(1, int(os.getenv("TORCH_NUM_THREADS", default_threads)))
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        self.model = create_model().to(self.device)
        checkpoint = _safe_torch_load(self.checkpoint_path, self.device)

        if isinstance(checkpoint, dict):
            checkpoint_model_name = checkpoint.get("model_name")
            if checkpoint_model_name and checkpoint_model_name != "convnext_tiny":
                raise PredictorError(
                    "Wrong checkpoint architecture: "
                    f"expected convnext_tiny, found {checkpoint_model_name}."
                )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise PredictorError(
                "Checkpoint does not match the ConvNeXt-Tiny architecture. "
                "Use convnext_tiny_seed123_best.pt from production_training_cell13."
            ) from exc

        self.model.eval()
        self._model_lock = threading.RLock()

        image_size = int(self.config["image_size"])
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=self.config["imagenet_mean"],
                    std=self.config["imagenet_std"],
                ),
            ]
        )

        self.raw_threshold = float(self.config["raw_probability_threshold"])
        self.temperature = float(self.config["temperature"])
        self.calibrated_threshold = float(
            self.config["calibrated_probability_threshold_equivalent"]
        )

    def _decode_image(self, data: bytes) -> tuple[Image.Image, int, int]:
        if not data:
            raise PredictorError("The uploaded image is empty.")

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                width, height = image.size
                image_copy = image.copy()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise PredictorError("The uploaded file is not a supported readable image.") from exc

        return image_copy, width, height

    def _format_result(self, logit: float, width: int, height: int) -> dict[str, Any]:
        raw_score = _sigmoid(logit)
        calibrated_score = _sigmoid(logit / self.temperature)
        predicted_label = int(raw_score >= self.raw_threshold)
        label = self.config["label_mapping"][str(predicted_label)]

        return {
            "model": self.config["display_name"],
            "model_name": self.config["model_name"],
            "selected_seed": int(self.config["selected_seed"]),
            "prediction": label,
            "predicted_label": predicted_label,
            "malignancy_score_raw": raw_score,
            "malignancy_score_temperature_scaled": calibrated_score,
            "decision_threshold_raw": self.raw_threshold,
            "decision_threshold_temperature_scaled_equivalent": self.calibrated_threshold,
            "input_width": width,
            "input_height": height,
            "device": str(self.device),
            "research_warning": self.config["warning"],
        }

    @torch.inference_mode()
    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        image, width, height = self._decode_image(data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with self._model_lock:
            logit = float(self.model(tensor).reshape(-1)[0].item())

        return self._format_result(logit, width, height)

    def _gradcam_target_layer(self) -> torch.nn.Module:
        """Return the final spatial convolution used for Grad-CAM."""
        try:
            return self.model.stages[-1].blocks[-1].conv_dw
        except (AttributeError, IndexError):
            conv_layers = [
                module
                for module in self.model.modules()
                if isinstance(module, torch.nn.Conv2d)
            ]
            if not conv_layers:
                raise PredictorError("No convolutional layer is available for Grad-CAM.")
            return conv_layers[-1]

    def explain_bytes(self, data: bytes) -> dict[str, Any]:
        """Generate a Grad-CAM map supporting the model's predicted class.

        The heatmap is an explanation aid. It is not a nodule segmentation mask
        and must not be interpreted as a clinical boundary.
        """
        image, width, height = self._decode_image(data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        target_layer = self._gradcam_target_layer()

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def capture_activation(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

        hook = target_layer.register_forward_hook(capture_activation)

        try:
            with self._model_lock, torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                output = self.model(tensor).reshape(-1)[0]
                logit = float(output.detach().item())
                result = self._format_result(logit, width, height)

                # Explain evidence supporting the returned class. For a benign
                # prediction, negative logit evidence is used.
                objective = output if result["predicted_label"] == 1 else -output
                objective.backward()

                if "value" not in activations or "value" not in gradients:
                    raise PredictorError("Grad-CAM hooks did not capture model features.")

                feature_map = activations["value"]
                feature_grad = gradients["value"]
                if feature_map.ndim != 4 or feature_grad.ndim != 4:
                    raise PredictorError("Unexpected feature-map shape for Grad-CAM.")

                weights = feature_grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))

                if float(cam.max().detach().item()) <= 1e-12:
                    cam = torch.abs((weights * feature_map).sum(dim=1, keepdim=True))

                cam = cam - cam.amin(dim=(2, 3), keepdim=True)
                denominator = cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
                cam = cam / denominator
                cam = F.interpolate(
                    cam,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                heatmap = cam.detach().cpu().numpy().astype(np.float32)
        finally:
            hook.remove()
            self.model.zero_grad(set_to_none=True)

        # Summarize the heatmap without claiming clinical features.
        high_activation = heatmap >= 0.60
        coverage = float(high_activation.mean())
        focus_mask = heatmap >= float(np.quantile(heatmap, 0.80))
        focus_weights = heatmap * focus_mask
        total_weight = float(focus_weights.sum())

        if total_weight > 0:
            yy, xx = np.indices(heatmap.shape)
            cx = float((xx * focus_weights).sum() / total_weight) / max(width - 1, 1)
            cy = float((yy * focus_weights).sum() / total_weight) / max(height - 1, 1)
        else:
            cx = cy = 0.5

        horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
        focus_region = f"{vertical}-{horizontal}"

        return {
            **result,
            "heatmap": heatmap,
            "focus_region": focus_region,
            "high_activation_coverage": coverage,
            "explanation_method": "Grad-CAM",
        }
