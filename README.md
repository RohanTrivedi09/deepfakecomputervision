# Deepfake / Face-Swap Detection Platform

Binary real-vs-fake face classifier (ResNet18, transfer learning) with Grad-CAM
explainability, served through a Streamlit demo (file upload + webcam snapshot).
Sem 7 computer vision course project — see `deepfake-detection-plan.md` for the
full scope, rationale, and locked v1 decisions.

Trained on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
dataset (70k real / 70k StyleGAN-fake), using a stratified subset.

## How this repo is organized

Training is compute-heavy and this repo is developed on a machine without a
dedicated GPU, so **training runs on Google Colab**, not locally:

- `notebooks/deepfake_training.ipynb` is a single self-contained notebook that
  covers the whole training pipeline — data prep, ResNet18 fine-tuning,
  evaluation, and Grad-CAM — with no dependency on cloning this repo. Upload
  it to Colab and run it there on a free GPU.
- `app/streamlit_app.py` and `src/inference/predict.py` run **locally** — they
  only need the checkpoint the notebook produces
  (`models/checkpoints/deepfake_resnet18.pt` + `metadata.json`), not the
  dataset or a GPU.

## Running the training pipeline (Colab)

1. In a browser, go to the [dataset page](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
   and click **Download** (needs a Kaggle account login — no API token needed).
2. Upload the downloaded zip to your Google Drive.
3. Upload `notebooks/deepfake_training.ipynb` to [Colab](https://colab.research.google.com)
   (File → Upload notebook), or open it directly from this GitHub repo.
4. Runtime → Change runtime type → GPU (T4).
5. In the "Get the Dataset" cell, update `DRIVE_ZIP_PATH` if your zip isn't at
   the root of My Drive, then Runtime → Run all. It mounts Drive, extracts the
   dataset onto the Colab VM's local disk, builds a stratified subset, trains,
   evaluates, and generates Grad-CAM samples.
6. The last cell zips `models/checkpoints/` and `reports/figures/` and
   downloads them through the browser — unzip into the same paths in your
   local clone.

## Running the demo (local)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

Requires `models/checkpoints/deepfake_resnet18.pt` and `metadata.json` to
exist (brought back from Colab per above). Open the local URL Streamlit
prints, then use either the "Upload image" or "Webcam snapshot" tab.

## Repository structure

```text
notebooks/
  deepfake_training.ipynb  # self-contained training pipeline (runs on Colab)
src/
  common/           # shared path constants (checkpoint location)
  training/         # ResNet18 architecture builder (mirrors the notebook's)
  inference/        # predict() + Grad-CAM, used by the Streamlit app
models/checkpoints/ # deepfake_resnet18.pt + metadata.json (from Colab)
app/
  streamlit_app.py  # demo UI: upload / webcam -> prediction + Grad-CAM
reports/
  figures/                 # confusion matrix, Grad-CAM samples (from Colab)
  project_report.md        # course write-up (approach, results, limitations)
```

## Known limitations (v1)

- Trained on one GAN family (StyleGAN) — may not generalize to deepfakes from
  other generators (a known hard research problem, not attempted here).
- Static images only, no video/temporal deepfake detection.
- Binary real-vs-fake only, no sub-typing of which method generated a fake.

See `reports/project_report.md` for the full write-up once trained.
