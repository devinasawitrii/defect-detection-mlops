"""Gradio demo, also the entrypoint for the Hugging Face Space."""
from pathlib import Path

import gradio as gr

from serving.inference import Detector, draw_detections

detector = Detector()
EXAMPLES = sorted(
    [str(p), 0.25] for p in Path("examples").glob("*") if p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"}
)


def detect(image, conf):
    if image is None:
        return None, {}
    result = detector.predict(image, conf)
    counts: dict[str, int] = {}
    for d in result["detections"]:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    return draw_detections(image, result["detections"], detector.class_names), counts


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="Steel surface image"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[gr.Image(label="Detections"), gr.JSON(label="Defects found")],
    title="Steel Surface Defect Detector (YOLO26)",
    description=(
        "Upload a hot-rolled steel strip image (NEU-DET style). YOLO26 fine-tuned with transfer learning "
        "localises and classifies 6 defect types, served via ONNX Runtime."
    ),
    examples=EXAMPLES or None,
)

if __name__ == "__main__":
    demo.launch()
