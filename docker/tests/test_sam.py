import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import pytest

from app import (
    intersection_over_union,
    load_predictor,
    make_demo_image,
    make_demo_reference_mask,
    save_results,
    segment,
)


@pytest.fixture(scope="module")
def predictor():
    return load_predictor()


def test_iou_matches_identical_masks_and_handles_disjoint_masks():
    first = np.array([[1, 1], [0, 0]], dtype=bool)
    second = np.array([[1, 0], [0, 1]], dtype=bool)
    assert intersection_over_union(first, first) == 1.0
    assert intersection_over_union(first, second) == pytest.approx(1 / 3)


def test_demo_image_and_reference_mask_are_generated(tmp_path):
    image_path = tmp_path / "demo.png"
    reference_path = tmp_path / "reference_mask.png"
    make_demo_image(image_path)
    make_demo_reference_mask(reference_path)
    image = Image.open(image_path)
    reference = Image.open(reference_path)
    assert image.size == (512, 384)
    assert image.mode == "RGB"
    assert reference.size == image.size
    assert np.asarray(reference).sum() > 0


def test_sam_creates_mask_with_minimum_reference_iou_and_visible_artifacts(predictor, tmp_path):
    image_path = tmp_path / "demo.png"
    reference_path = tmp_path / "reference_mask.png"
    make_demo_image(image_path)
    make_demo_reference_mask(reference_path)
    image = Image.open(image_path).convert("RGB")
    reference = np.asarray(Image.open(reference_path).convert("L")) > 0
    mask, score = segment(predictor, image, 256, 185)
    iou = intersection_over_union(mask, reference)

    assert mask.shape == (image.height, image.width)
    assert 0 < int(mask.sum()) < mask.size
    assert 0.0 <= score <= 1.0
    assert iou >= 0.60, f"SAM reference-mask IoU too low: {iou:.3f}"

    output = Path(os.getenv("OUTPUT_DIR", "/app/output"))
    metadata = save_results(image, mask, score, (256, 185), output, reference)
    assert metadata["mask_pixels"] == int(mask.sum())
    assert metadata["reference_iou"] == pytest.approx(iou)
    assert Image.open(output / "mask.png").size == image.size
    assert Image.open(output / "overlay.png").size == image.size
    assert Image.open(output / "reference_mask.png").size == image.size
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["mask_pixels"] > 0
    assert result["reference_iou"] >= 0.60

