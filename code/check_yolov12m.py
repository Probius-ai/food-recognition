#!/usr/bin/env python3
"""Quick sanity check that YOLOv12m weights load and run a forward pass."""

import sys
import torch
from ultralytics import YOLO


MODEL_NAME = "yolo12m.pt"
IMG_SIZE = 640


def _describe_shapes(obj) -> str:
    """Recursively collect tensor-like shapes for logging."""
    shapes = []

    def _collect(item):
        if torch.is_tensor(item):
            shapes.append(str(tuple(item.shape)))
        elif hasattr(item, "shape"):
            shapes.append(str(tuple(item.shape)))
        elif isinstance(item, (list, tuple)):
            for sub in item:
                _collect(sub)
        else:
            shapes.append(type(item).__name__)

    _collect(obj)
    return ", ".join(shapes)


def main() -> None:
    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load '{MODEL_NAME}'. Ensure network access is enabled so Ultralytics can "
            f"download the weights, or place the file locally. Original error: {exc}"
        ) from exc

    print(f"Successfully loaded '{MODEL_NAME}'")

    dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    try:
        outputs = model.model(dummy_input)
    except Exception as exc:
        print(f"Forward pass on dummy input failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Forward pass successful. Output shapes: {_describe_shapes(outputs)}")


if __name__ == "__main__":
    main()
