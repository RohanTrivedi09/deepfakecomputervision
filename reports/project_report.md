# Deepfake / Face-Swap Detection — Project Report

*Skeleton — fill in after training completes on Colab (see `notebooks/colab_train.ipynb`).*

## 1. Problem Statement

(see `deepfake-detection-plan.md` — Problem Statement)

## 2. Dataset

- 140k Real and Fake Faces (Kaggle `xhlulu/140k-real-and-fake-faces`)
- Subset used: ___ images/class (train ___ / valid ___ / test ___)

## 3. Approach

- Model: ResNet18 (ImageNet-pretrained), fine-tuned, binary classification head
- Augmentation: horizontal flip, ±10° rotation, color jitter
- Explainability: Grad-CAM on the final conv block (`layer4`)

## 4. Results

- Accuracy / Precision / Recall / F1: *(fill from `reports/metrics.json`)*
- Confusion matrix: *(see `reports/figures/confusion_matrix.png`)*
- Grad-CAM samples (correct + incorrect): *(see `reports/figures/gradcam/`)*
- Notable failure patterns: ___

## 5. Limitations

- Cross-GAN generalization not evaluated — model trained only on StyleGAN-generated
  fakes and may not flag deepfakes from other generators.
- Static images only; no temporal/video deepfake detection.
- No sub-typing of which generation method produced a fake.

## 6. Ethical Note

Dataset uses consented/synthetic faces (FFHQ real faces + StyleGAN-generated
fakes) — no real individuals were impersonated without consent.
