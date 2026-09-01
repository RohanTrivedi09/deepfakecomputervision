"""Shared inference utilities: load the checkpoint once, predict on a single
image, and produce a Grad-CAM overlay. Used by app/streamlit_app.py.

Loads whatever notebooks/deepfake_training.ipynb saved: a raw state_dict at
models/checkpoints/deepfake_resnet18.pt plus models/checkpoints/metadata.json
(class_names, img_size, normalize mean/std). The Grad-CAM implementation below
mirrors that notebook's own hook-based GradCAM class exactly, so inference
matches what the notebook showed during evaluation.
"""
import json
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.common.paths import MODELS_DIR
from src.training.model import build_model, get_device


class GradCAM:
    """Minimal Grad-CAM via forward/backward hooks on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, img_size, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (img_size, img_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (1 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0, 1)


@lru_cache(maxsize=1)
def load_inference_bundle():
    device = get_device()
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())

    model = build_model(num_classes=metadata["num_classes"]).to(device)
    state_dict = torch.load(MODELS_DIR / "deepfake_resnet18.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cam = GradCAM(model, target_layer=model.layer4[-1])
    transform = transforms.Compose([
        transforms.Resize((metadata["img_size"], metadata["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(metadata["normalize_mean"], metadata["normalize_std"]),
    ])
    return model, cam, transform, metadata, device


def predict(image: Image.Image) -> dict:
    model, cam, transform, metadata, device = load_inference_bundle()
    classes = metadata["class_names"]
    img_size = metadata["img_size"]

    pil_img = image.convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    grayscale_cam, _ = cam.generate(input_tensor, img_size, class_idx=pred_idx)
    rgb_img = np.array(pil_img.resize((img_size, img_size))).astype(np.float32) / 255.0
    overlay = (overlay_heatmap(rgb_img, grayscale_cam) * 255).astype(np.uint8)

    return {
        "label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "gradcam_overlay": overlay,
    }
