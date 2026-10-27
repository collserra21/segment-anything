import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from segment_anything import SamPredictor, sam_model_registry

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = Path(os.getenv("SAM_HOME", "/models")) / "sam_vit_b_01ec64.pth"


def make_demo_image(path: Path) -> None:
    """Create a deterministic scene with a clear foreground object."""
    width, height = 512, 384
    im = Image.new("RGB", (width, height), "#90c7ee")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 235, width, height), fill="#4f9150")
    # A high-contrast ball is the prompted object; click at its center.
    d.ellipse((175, 105, 337, 267), fill="#e54837", outline="#97291f", width=5)
    d.arc((190, 120, 323, 253), start=205, end=335, fill="#ffc1a9", width=8)
    d.ellipse((215, 137, 250, 172), fill="#ffd4c5")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def make_demo_reference_mask(path: Path) -> None:
    """Write the ground-truth silhouette of the demo ball."""
    mask = Image.new("L", (512, 384), 0)
    ImageDraw.Draw(mask).ellipse((175, 105, 337, 267), fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)


def intersection_over_union(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Compute binary-mask IoU, rejecting incompatible mask dimensions."""
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError(f"Mask shapes differ: {predicted.shape} != {reference.shape}")
    union = np.logical_or(predicted, reference).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted, reference).sum() / union)

def get_checkpoint() -> Path:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.exists():
        temp = CHECKPOINT_PATH.with_suffix(".download")
        urllib.request.urlretrieve(CHECKPOINT_URL, temp)
        temp.replace(CHECKPOINT_PATH)
    return CHECKPOINT_PATH


def load_predictor():
    model = sam_model_registry["vit_b"](checkpoint=str(get_checkpoint()))
    model.eval()
    model.to(device="cpu")
    return SamPredictor(model)


def segment(predictor, image: Image.Image, x: int, y: int):
    rgb = np.asarray(image.convert("RGB"))
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def save_results(image: Image.Image, mask: np.ndarray, score: float, point, output: Path, reference_mask=None):
    output.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(output / "mask.png")

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    overlay = base.copy()
    tint = np.zeros_like(overlay)
    tint[:, :] = (255, 45, 45)
    overlay[mask] = (0.55 * base[mask] + 0.45 * tint[mask]).astype(np.uint8)
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)
    x, y = point
    draw.ellipse((x-6, y-6, x+6, y+6), fill="#00ff70", outline="black", width=2)
    result.save(output / "overlay.png")

    metadata = {
        "model": "facebookresearch/segment-anything vit_b",
        "device": "cpu",
        "prompt_point_xy": [int(x), int(y)],
        "predicted_iou": score,
        "mask_pixels": int(mask.sum()),
        "image_size": list(image.size),
    }
    if reference_mask is not None:
        reference = np.asarray(reference_mask, dtype=bool)
        if reference.shape != mask.shape:
            raise ValueError(f"Reference mask shape {reference.shape} does not match prediction {mask.shape}")
        metadata["reference_iou"] = intersection_over_union(mask, reference)
        Image.fromarray(reference.astype(np.uint8) * 255, mode="L").save(output / "reference_mask.png")
    (output / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="SAM point-prompt image segmentation")
    parser.add_argument("--image", default="/app/data/demo.png")
    parser.add_argument("--output", default="/app/output")
    parser.add_argument("--point-x", type=int, default=256)
    parser.add_argument("--point-y", type=int, default=185)
    parser.add_argument("--make-demo-image", action="store_true")
    parser.add_argument("--reference-mask", help="Optional ground-truth mask image for IoU evaluation")
    args = parser.parse_args()
    image_path = Path(args.image)
    if args.make_demo_image or not image_path.exists():
        make_demo_image(image_path)
        reference_path = Path(args.output) / "reference_mask.png"
        make_demo_reference_mask(reference_path)
    else:
        reference_path = Path(args.reference_mask) if args.reference_mask else None
    image = Image.open(image_path).convert("RGB")
    if not (0 <= args.point_x < image.width and 0 <= args.point_y < image.height):
        parser.error("Prompt point must be inside the image")
    predictor = load_predictor()
    mask, score = segment(predictor, image, args.point_x, args.point_y)
    reference_mask = None
    if reference_path is not None:
        reference_mask = np.asarray(Image.open(reference_path).convert("L")) > 0
    result = save_results(image, mask, score, (args.point_x, args.point_y), Path(args.output), reference_mask)
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}/overlay.png, mask.png, result.json")


if __name__ == "__main__":
    main()





