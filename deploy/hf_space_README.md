---
title: Steel Surface Defect Detector
emoji: 🔩
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

Demo of a steel-surface defect **detector** (NEU-DET, 6 classes).
YOLO26 fine-tuned with PyTorch (transfer learning from COCO), exported to an NMS-free ONNX graph and served with ONNX Runtime.
