"""Publish the Gradio demo to a Hugging Face Space.

Needs env vars HF_TOKEN (write token) and HF_SPACE_ID (e.g. "username/steel-defect-demo").
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

space_id = os.environ["HF_SPACE_ID"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)

uploads = {
    "app.py": "app.py",
    "serving/__init__.py": "serving/__init__.py",
    "serving/inference.py": "serving/inference.py",
    "models/model.onnx": "models/model.onnx",
    "models/classes.json": "models/classes.json",
    "deploy/hf_requirements.txt": "requirements.txt",
    "deploy/hf_space_README.md": "README.md",
}
for local, remote in uploads.items():
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=space_id, repo_type="space")

if Path("examples").exists():
    api.upload_folder(folder_path="examples", path_in_repo="examples", repo_id=space_id, repo_type="space")

print(f"Deployed: https://huggingface.co/spaces/{space_id}")
