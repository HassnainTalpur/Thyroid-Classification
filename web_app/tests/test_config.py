import json
import math
from pathlib import Path


def test_equivalent_thresholds() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "model" / "model_config.json").read_text())
    raw = config["raw_probability_threshold"]
    temperature = config["temperature"]
    expected = 1 / (1 + math.exp(-(math.log(raw / (1 - raw)) / temperature)))
    assert abs(expected - config["calibrated_probability_threshold_equivalent"]) < 1e-12
