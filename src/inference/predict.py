"""Shared inference utilities: load checkpoint once, predict on a single image,
and produce a Grad-CAM overlay. Used by app/streamlit_app.py.
"""
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.common.paths import MODELS_DIR
from src.preprocessing.dataset import build_transforms
from src.training.model import build_model, get_device


@lru_cache(maxsize=1)
def load_inference_bundle():
    device = get_device()
    checkpoint = torch.load(MODELS_DIR / "best_model.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    return model, cam, checkpoint["classes"], device


def predict(image: Image.Image) -> dict:
    model, cam, classes, device = load_inference_bundle()
    transform = build_transforms(train=False)

    pil_img = image.convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return {
        "label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "gradcam_overlay": overlay,
    }
