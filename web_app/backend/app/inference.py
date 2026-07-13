from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
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

        # Large default thread counts can make single-image CPU inference much
        # slower on laptops and shared servers due to oversubscription.
        if self.device.type == "cpu":
            default_threads = min(4, os.cpu_count() or 1)
            cpu_threads = max(1, int(os.getenv("TORCH_NUM_THREADS", default_threads)))
            torch.set_num_threads(cpu_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch only allows this setting before parallel work starts.
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

    @torch.inference_mode()
    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        if not data:
            raise PredictorError("The uploaded image is empty.")

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image = image.convert("RGB")
                width, height = image.size
                tensor = self.transform(image).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise PredictorError("The uploaded file is not a supported readable image.") from exc

        logit = float(self.model(tensor).reshape(-1)[0].item())
        raw_score = _sigmoid(logit)
        calibrated_score = _sigmoid(logit / self.temperature)

        # The notebook's primary operating point used the raw probability with
        # a Youden threshold selected only on TN5000 validation data.
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
