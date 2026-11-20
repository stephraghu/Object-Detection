# YOLOv8 COCO Object Detection

A Google Colab workflow for training, evaluating, and running image and browser-webcam object detection using **Ultralytics YOLOv8 (`yolov8n.pt`)** on the **COCO (`coco.yaml`)** dataset.

## Overview

This notebook demonstrates how to set up a GPU-accelerated environment, train a `yolov8n.pt` object detection model on COCO, evaluate validation metrics, run inference on sample images, back up weights to Google Drive, and stream live webcam frames for real-time detection inside Colab.

## Notebook Workflow

1. **Install & Import Dependencies:** Installs `ultralytics` (`8.4.184`) and imports `torch` and `YOLO`.
2. **GPU Check:** Verifies CUDA availability (`torch.cuda.is_available()`) and inspects GPU details with `nvidia-smi` (configured on an **NVIDIA Tesla T4** GPU).
3. **Baseline Training Run (1 Epoch):** Runs `yolo detect train` with `data=coco.yaml`, `model=yolov8n.pt`, `epochs=1`, `imgsz=640`, and `batch=16`, automatically downloading COCO labels and images into `/content/datasets` and saving results to `/content/runs/detect/train`.
4. **Multi-Epoch Training Run (12 Epochs, 50% Dataset Fraction):** Uses the Python API (`model.train(...)`) on `coco.yaml` with `epochs=12`, `imgsz=640`, `batch=16`, `device=0`, `cache=True`, and `fraction=0.5`, saving outputs under `project="coco_scratch_project"` and `name="yolov8n_scratch"` (`/content/runs/detect/coco_scratch_project/yolov8n_scratch`).
5. **Validation & Image Inference:** Evaluates a trained checkpoint with `model.val(data="coco.yaml")` and runs inference on an image URL using `PIL.Image` and `result.plot()`.
6. **Optional Backup & Live Webcam Stream (Marked as Optional for Workshop):** Mounts Google Drive (`/content/drive`) to copy trained weights and registers a Colab kernel callback (`notebook.SendFrame`) with JavaScript `getUserMedia` to stream annotated webcam frames until `Q` is pressed.

## Setup & Usage

### Install Ultralytics & Check GPU
```python
!pip install ultralytics
import torch
from ultralytics import YOLO

print("GPU Available:", torch.cuda.is_available())
!nvidia-smi
