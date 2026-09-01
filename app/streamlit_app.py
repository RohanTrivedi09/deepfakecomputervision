"""Streamlit demo: upload or webcam capture -> real/fake prediction + Grad-CAM overlay."""
import streamlit as st
from PIL import Image

from src.inference.predict import predict

st.set_page_config(page_title="Deepfake Face Detector", layout="centered")
st.title("Deepfake / Face-Swap Detector")
st.caption(
    "Binary real-vs-fake face classifier (ResNet18, fine-tuned on the 140k Real "
    "and Fake Faces dataset) with Grad-CAM explainability. Trained on one GAN "
    "family (StyleGAN) — may not generalize to other generators."
)

tab_upload, tab_camera = st.tabs(["Upload image", "Webcam snapshot"])

image = None
with tab_upload:
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)

with tab_camera:
    snapshot = st.camera_input("Take a snapshot")
    if snapshot is not None:
        image = Image.open(snapshot)

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Running inference..."):
        result = predict(image)

    label = result["label"]
    confidence = result["confidence"]
    verdict = "FAKE" if label == "fake" else "REAL"
    color = "red" if label == "fake" else "green"

    with col2:
        st.image(result["gradcam_overlay"], caption="Grad-CAM overlay", use_container_width=True)

    st.markdown(f"### Verdict: :{color}[{verdict}]  (confidence {confidence:.1%})")
    st.bar_chart(result["probabilities"])
else:
    st.info("Upload an image or take a webcam snapshot to get a prediction.")
