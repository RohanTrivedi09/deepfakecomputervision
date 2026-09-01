# Deepfake / Face-Swap Detection Platform

## Build Notes (read this first)

This is a **college computer-vision course project** (Sem 7, complements CV lab work on image transforms/homography). It runs alongside a parallel project (an adversarial-ML research paper on evasion attacks vs. NIDS), so scope below is locked to a v1 achievable solo in roughly **25–35 hours**. Do not expand scope beyond what's marked "v1" without asking first.

**Why this topic over alternatives considered:**
- Dataset is freely downloadable, no license request, no gatekeeping (verified — see Data Source).
- Genuine CV problem: real, learnable visual artifacts (GAN blending seams, texture/frequency inconsistencies), unlike QR-code "malicious visual" detection which is fundamentally a decoded-URL problem, not an image problem.
- Reuses the same binary-classification + transfer-learning + Grad-CAM + Streamlit template already scoped for the earlier industrial-defect plan — low re-plan risk.
- Ties directly into the user's other Sem 7 project (adversarial ML / security) — coherent portfolio narrative rather than two unrelated projects.

**v1 locked decisions:**
- **Dataset: 140k Real and Fake Faces** (Kaggle, `xhlulu/140k-real-and-fake-faces`) — 70k real faces (Flickr/FFHQ via NVIDIA), 70k GAN-generated fake faces (StyleGAN, via Bojan Tunguz's "1 Million Fake Faces"), pre-split train/val/test, 224×224. No registration/license agreement required.
- **Static images only.** This is image-based manipulation detection, not video deepfake detection — no frame extraction, no temporal modeling. (Video deepfakes are a much bigger lift — face tracking, temporal consistency models — explicitly deferred.)
- **Binary classification only**: real vs. fake. No further sub-typing (e.g. which GAN generated it).
- **Streamlit demo**, with both file upload and `st.camera_input` (webcam snapshot) as input methods — same pipeline either way.
- Grad-CAM stays in — it's the demo payoff: shows *which facial region* (eyes, mouth, blending boundary) triggered the "fake" verdict.
- Docker is optional polish, not a v1 requirement.

## Project Overview

A binary image classifier that distinguishes authentic human face photos from GAN-generated fake faces, with visual explainability (Grad-CAM) showing which facial regions drove the prediction, delivered through a Streamlit demo supporting both file upload and live webcam capture.

## Problem Statement

Synthetic/GAN-generated faces are now visually convincing enough to fool casual inspection, creating risk for identity fraud, misinformation, and fake profile creation. There's a need for an accessible tool that can flag a face image as likely real or fake, and show *why* — not just a black-box verdict.

## Objectives (v1)

- build a binary real-vs-fake face classifier using transfer learning,
- train and evaluate on the 140k Real and Fake Faces dataset,
- generate Grad-CAM overlays highlighting the regions driving each prediction,
- present results through a Streamlit demo (upload + webcam snapshot),
- document the pipeline, results, and limitations for the course report/viva.

## Scope

### In Scope (v1)

- binary (real/fake) classification on static face images,
- preprocessing and light augmentation,
- transfer-learning model training and evaluation,
- Grad-CAM visualization of predictions,
- Streamlit web demo: file upload + webcam snapshot capture.

### Out of Scope (v1)

- video-based / temporal deepfake detection,
- face-swap localization or segmentation (i.e. exactly which pixels were swapped),
- multi-class sub-typing (which generation method produced the fake),
- detecting deepfakes the model wasn't trained on generalizing to (cross-GAN generalization is a known hard research problem — acknowledge as a limitation, don't try to solve it),
- full production deployment, mobile/edge inference,
- audio deepfake detection.

## Tech Stack

### Core Languages and Frameworks

- Python
- PyTorch (pick one framework, don't dual-build in TF)
- OpenCV (face crop/alignment preprocessing if needed)
- scikit-learn (metrics only)

### Model Approach (v1)

- CNN classification via transfer learning (e.g. ResNet18 or EfficientNet-B0, pretrained on ImageNet, fine-tuned on the 140k dataset),
- Grad-CAM for explainability.
- *(Frequency-domain artifact analysis, e.g. spectral/FFT-based detectors, is a strong v2 upgrade — not v1.)*

### Application Layer

- Streamlit, with both `st.file_uploader` (upload) and `st.camera_input` (webcam snapshot) as input methods,
- Matplotlib for result/metric visualization.

### Supporting Tools

- GitHub
- Jupyter Notebook (experimentation only — final pipeline should be scripts, not notebooks)
- Kaggle API for dataset download

## Data Source (v1)

**140k Real and Fake Faces** — Kaggle dataset `xhlulu/140k-real-and-fake-faces`.
- 70,000 real faces (Flickr/FFHQ, via NVIDIA)
- 70,000 fake faces (StyleGAN-generated, via Bojan Tunguz's "1 Million Fake Faces")
- Already resized/split into train/validation/test — minimal preprocessing overhead.
- Direct download via Kaggle (`kaggle datasets download -d xhlulu/140k-real-and-fake-faces`), no license form.
- For v1 timeline, use a stratified subset (e.g. ~10–15k images per class) rather than the full 140k — full-set training is unnecessary for a course-scope demo and saves compute/time.

## System Architecture

1. Download and subsample the dataset (stratified real/fake subset).
2. Preprocessing: resize, normalize; optional face alignment via OpenCV if source images are inconsistently cropped.
3. Light augmentation (flips, small rotations, color jitter) for robustness.
4. Binary classifier trained via transfer learning.
5. Evaluation on the held-out test split (accuracy, precision, recall, F1, confusion matrix).
6. Grad-CAM generated for sample predictions (both correct and incorrect, for the report).
7. Inference exposed through the Streamlit app (upload or webcam capture → prediction + confidence + Grad-CAM overlay).

### Logical Layers

- dataset layer,
- preprocessing layer,
- training layer,
- evaluation layer,
- inference + visualization layer (combined for v1).

## Key Features (v1)

- real vs. fake face detection,
- automated preprocessing pipeline,
- light augmentation,
- confidence score per prediction,
- Grad-CAM explainability overlay (shows *which* facial region looked synthetic),
- Streamlit demo supporting both file upload and live webcam snapshot capture,
- evaluation report (metrics + confusion matrix) for the course write-up.

## Implementation Phases (v1)

### Phase 1: Setup

- download dataset via Kaggle API,
- build stratified subset,
- confirm binary label structure, set eval metrics.

### Phase 2: Data Preparation

- load and resize images,
- normalize,
- confirm/adjust train/val/test split,
- apply light augmentation.

### Phase 3: Model Development

- build baseline transfer-learning classifier,
- fine-tune,
- checkpoint the trained model.

### Phase 4: Evaluation and Explainability

- compute accuracy/precision/recall/F1/confusion matrix,
- generate Grad-CAM overlays on a handful of correct + incorrect predictions,
- note false positives/negatives and any visible failure patterns for the report.

### Phase 5: Streamlit Demo

- build UI with both file upload and `st.camera_input` webcam snapshot,
- show prediction, confidence, Grad-CAM overlay,
- polish for presentation.

### Phase 6: Wrap-up

- clean code/repo,
- write short report/README covering approach, results, limitations (esp. generalization limits — flag that model is trained on one GAN family and may not generalize to other deepfake methods),
- prep demo walkthrough.

## Estimated Timeline (v1)

- **Total:** ~26–36 hours over 3–4 weeks.

| Week | Focus | Hours |
|---|---|---|
| 1 | Dataset setup + preprocessing pipeline | 6–8 |
| 2 | Baseline model + training | 8–10 |
| 3 | Evaluation + Grad-CAM | 5–7 |
| 4 | Streamlit demo (upload + camera snapshot) + report/polish | 7–11 |

## Milestones

- dataset subset prepared, binary labels confirmed,
- preprocessing pipeline working,
- baseline model trained and checkpointed,
- evaluation metrics + Grad-CAM working,
- Streamlit demo functional end-to-end (upload + webcam),
- report/README ready.

## Risks

- **cross-GAN generalization**: model trained on one GAN family (StyleGAN) may not flag deepfakes from other generators — call this out explicitly as a known limitation, don't oversell the model's real-world robustness in the report,
- class imbalance shouldn't be an issue here (dataset is balanced 70k/70k), but confirm after subsampling,
- Grad-CAM overlays can be subtle on high-quality fakes — pick a few clearly illustrative examples for the demo rather than relying on live results looking perfect every time,
- ethical framing: dataset uses consented/synthetic faces only (no real individuals impersonated without consent) — worth a one-line note in the report,
- scope creep toward video/temporal detection — resist; that's v2.

## Deferred / Future Scope (explicitly NOT v1 — do not build yet)

- video-based deepfake detection (temporal models, frame consistency),
- face-swap region localization/segmentation,
- cross-GAN generalization improvements (domain adaptation, multi-dataset training),
- frequency-domain/spectral artifact detectors,
- audio deepfake detection,
- mobile/edge deployment,
- Docker packaging.

## Resume Value

Demonstrates: practical computer vision implementation, transfer learning, data preprocessing/augmentation, evaluation and explainability (Grad-CAM), and a working end-to-end demo — in a security-relevant domain (synthetic media detection) that pairs naturally with adversarial-ML/security work. Relevant for computer vision, applied AI/security, and ML engineering roles.

## Repository Structure (v1)

```text
deepfake-detection-platform/
├── data/
│   ├── raw/             # 140k Real and Fake Faces subset
│   └── processed/
├── notebooks/
│   └── experiments/      # exploration only, not the final pipeline
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── models/
│   └── checkpoints/
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   └── project_report.md
├── requirements.txt
├── README.md
└── .gitignore
```
