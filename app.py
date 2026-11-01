"""Pneumonia detection demo: upload a chest X-ray and see the predicted class and probability.

Academic decision-support prototype. Not a clinical diagnostic tool.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom
import streamlit as st
from tensorflow import keras

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Pneumonia Detection", page_icon=":lungs:", layout="centered")


@st.cache_resource
def load_model_and_config():
    config = json.loads((MODEL_DIR / "model_config.json").read_text())
    model = keras.models.load_model(MODEL_DIR / config["model_file"], compile=False)
    return model, config


def read_gray(uploaded_file):
    """Return an 8-bit grayscale image, handling DICOM, PNG and JPEG uploads."""
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".dcm"):
        pixels = pydicom.dcmread(io.BytesIO(data)).pixel_array.astype(np.float32)
        if pixels.min() < 0 or pixels.max() > 255:           # rescale 12/16-bit DICOMs to 0-255
            pixels = (pixels - pixels.min()) / max(pixels.max() - pixels.min(), 1e-6) * 255.0
        return pixels.astype(np.uint8)
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("The file could not be read as an image.")
    return gray


def preprocess(gray, size):
    """Same steps as training: resize to size x size, scale to [0, 1], add batch and channel axes."""
    small = cv2.resize(gray, (size, size))
    return (small.astype("float32") / 255.0)[None, ..., None]


model, config = load_model_and_config()

st.title("Pneumonia Detection from Chest X-rays")
st.warning("Academic prototype for decision support only. It is not a medical diagnosis. "
           "A qualified clinician must review every result.")

threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, float(config["threshold"]), 0.01,
                              help="Images with a pneumonia probability at or above this value are flagged.")
st.sidebar.markdown(f"**Model:** {config['model_name']}")
st.sidebar.markdown(f"**Default threshold:** {config['threshold']:.2f} (chosen on validation data "
                    f"for recall of at least {config['target_recall']:.0%})")

uploaded = st.file_uploader("Upload a chest X-ray (DICOM, PNG or JPEG)", type=["dcm", "png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        gray = read_gray(uploaded)
    except Exception as error:
        st.error(f"Could not read this file: {error}")
        st.stop()

    st.image(gray, caption="Uploaded image (grayscale)", use_container_width=True, clamp=True)
    prob = float(model.predict(preprocess(gray, config["img_size"]), verbose=0)[0, 0])
    label = config["class_names"][int(prob >= threshold)]

    st.subheader(f"Predicted class: {label}")
    st.metric("Probability of pneumonia", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Threshold used: {threshold:.2f}")
    if label == "Pneumonia":
        st.info("Flagged for clinician review. False alarms are possible, especially with other lung abnormalities.")
    else:
        st.info("Not flagged. This does not rule out pneumonia.")
else:
    st.caption("Upload an image to see the prediction.")
