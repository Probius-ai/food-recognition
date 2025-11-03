# Requirements Reference

This note summarizes the Python dependencies defined in `requirements.txt` so future iterations (including LLM copilots) can reason about the toolchain without re-reading the raw list.

## Reading `requirements.txt`
- The file uses the default PyPI resolver. The only pinned constraint is `jupyterlab>3.0`; all other packages track their latest compatible versions, so reproducibility may require a lock file or container image in the future.
- GPU-enabled builds of `torch`, `torchvision`, and `torchaudio` should match the target CUDA/cuDNN stack. Consider installing from the official PyTorch index rather than plain `pip install -r requirements.txt` when CUDA matters.

## Package Quick Reference

| Package | Role in Project | Notes |
| --- | --- | --- |
| `jupyterlab>3.0` | Interactive experimentation | Lower-bound keeps access to modern notebook UX and debugger. |
| `torch` | Core deep learning framework | Ensure CUDA build matches host drivers; pairs with `torchvision`/`torchaudio`. |
| `torchvision` | Vision I/O and models | Supplies datasets, augmentations, and pretrained CNN backbones. |
| `torchaudio` | Audio utilities | Included for completeness; verify if audio modality is required. |
| `numpy` | Numeric base layer | Backbone for tensor conversions and some augmentations. |
| `pandas` | Tabular data handling | Useful for experiment metadata and CSV annotations. |
| `ultralytics` | YOLO pipeline | Provides task-specific training/inference loops for detection. |
| `opencv-python` | Computer vision toolkit | Image decoding, resizing, drawing overlays. |
| `tqdm` | Progress visualization | Wraps training/evaluation loops with live progress bars. |
| `matplotlib` | Plotting | Baseline visualization for metrics/qualitative checks. |
| `pycocotools` | COCO metrics & API | Needed for detection dataset parsing and evaluation. |
| `albumentations` | Data augmentation | High-performance augmentations with OpenCV integration. |
| `scikit-learn` | Classical ML utilities | Metrics (precision/recall), model selection helpers. |
| `Pillow` | Image manipulation | Backing library for PIL image operations, used by torchvision. |
| `seaborn` | Statistical plots | Higher-level metric visualization built atop Matplotlib. |
| `tensorboard` | Training logs | PyTorch integration via `SummaryWriter`; ensure log dir writable. |
| `onnx` | Model export format | Needed to serialize trained models for inference portability. |
| `onnxruntime` | ONNX inference engine | Allows local validation of exported ONNX models. |
| `thop` | FLOP/parameter profiling | Handy for reporting model complexity. |
| `torchsummary` | Model summary utility | Prints layer-wise shapes/parameter counts. |
| `optuna` | Hyperparameter tuning | Supports automated search; integrate with training loops. |

## Maintenance Tips
- Consider generating a lock file (e.g., `pip-tools` or `poetry export`) once versions are validated to stabilize training reproducibility.
- Keep CUDA/cuDNN version notes alongside this file so future installations choose compatible PyTorch wheels.
- If `ultralytics` updates introduce breaking changes, pin the major version used to train released models.
- Monitor binary-heavy packages (`torch`, `onnxruntime`, `opencv-python`) for platform-specific wheels when deploying.
