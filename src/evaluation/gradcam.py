"""Generate Grad-CAM overlays for sample correct and incorrect test predictions.

Saves a handful of each (default 4/4) to reports/figures/gradcam/ for the report.
"""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.common.paths import DATA_PROCESSED, FIGURES_DIR, MODELS_DIR
from src.preprocessing.dataset import build_transforms
from src.training.model import build_model, get_device

N_SAMPLES_PER_CATEGORY = 4


def load_model():
    device = get_device()
    checkpoint = torch.load(MODELS_DIR / "best_model.pth", map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["classes"], device


def main() -> None:
    model, classes, device = load_model()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    transform = build_transforms(train=False)
    out_dir = FIGURES_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_saved, incorrect_saved = 0, 0
    test_dir = DATA_PROCESSED / "test"

    for cls_dir in sorted(test_dir.iterdir()):
        if correct_saved >= N_SAMPLES_PER_CATEGORY and incorrect_saved >= N_SAMPLES_PER_CATEGORY:
            break
        true_label = classes.index(cls_dir.name)

        for img_path in sorted(cls_dir.glob("*.*")):
            if correct_saved >= N_SAMPLES_PER_CATEGORY and incorrect_saved >= N_SAMPLES_PER_CATEGORY:
                break

            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_label = model(input_tensor).argmax(1).item()
            is_correct = pred_label == true_label

            if is_correct and correct_saved >= N_SAMPLES_PER_CATEGORY:
                continue
            if not is_correct and incorrect_saved >= N_SAMPLES_PER_CATEGORY:
                continue

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            rgb_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            tag = "correct" if is_correct else "incorrect"
            fname = f"{tag}_{classes[true_label]}_pred-{classes[pred_label]}_{img_path.stem}.png"
            Image.fromarray(overlay).save(out_dir / fname)

            if is_correct:
                correct_saved += 1
            else:
                incorrect_saved += 1

    print(f"Saved {correct_saved} correct and {incorrect_saved} incorrect Grad-CAM overlays to {out_dir}")


if __name__ == "__main__":
    main()
