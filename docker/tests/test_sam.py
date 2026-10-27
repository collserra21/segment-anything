import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import pytest

from app import load_predictor, make_demo_image, save_results, segment


@pytest.fixture(scope="module")
def predictor():
    return load_predictor()


def test_demo_image_is_generated(tmp_path):
    path = tmp_path / "demo.png"
    make_demo_image(path)
    image = Image.open(path)
    assert image.size == (512, 384)
    assert image.mode == "RGB"


def test_sam_creates_nonempty_mask_and_visible_artifacts(predictor, tmp_path):
    image_path = tmp_path / "demo.png"
    make_demo_image(image_path)
    image = Image.open(image_path).convert("RGB")
    mask, score = segment(predictor, image, 256, 185)

    assert mask.shape == (image.height, image.width)
    assert 0 < int(mask.sum()) < mask.size
    assert 0.0 <= score <= 1.0

    output = Path(os.getenv("OUTPUT_DIR", "/app/output"))
    metadata = save_results(image, mask, score, (256, 185), output)
    assert metadata["mask_pixels"] == int(mask.sum())
    assert Image.open(output / "mask.png").size == image.size
    assert Image.open(output / "overlay.png").size == image.size
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["mask_pixels"] > 0
