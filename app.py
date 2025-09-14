import os
import zipfile
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Classification using DenseNet121",
    page_icon="🧠",
    layout="centered"
)

MODEL_FILE = "brain_tumor_densenet_original_working_prototype.h5"
HF_REPO_URL = "https://huggingface.co/shomeshwarren/brain_tumor/resolve/main/brain_tumor_densenet_original_working_prototype.h5"

# -----------------------------
# DOWNLOAD MODEL IF MISSING
# -----------------------------
if not os.path.exists(MODEL_FILE):
    st.warning("📥 Model not found locally. Downloading from Hugging Face...")
    try:
        response = requests.get(HF_REPO_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(img):
    img = img.resize((224, 224))  # resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 Brain Tumor Classification using DenseNet121")
st.write("Upload an MRI image to classify the tumor type.")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔮 Predict"):
        if model is None:
            st.error("⚠️ Model not loaded. Please check the model file.")
        else:
            with st.spinner("🔎 Analyzing MRI..."):
                processed_img = preprocess(img)
                preds = model.predict(processed_img)
                predicted_class = CLASS_NAMES[np.argmax(preds)]
                confidence = np.max(preds) * 100

            st.markdown(
                f"<h3 style='text-align:center;'>✅ Predicted Tumor: <b>{predicted_class}</b> ({confidence:.2f}%)</h3>",
                unsafe_allow_html=True
            )
else:
    st.info("👆 Please upload an MRI image to start prediction.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>Built with ❤️ using Streamlit & DenseNet121</p>",
    unsafe_allow_html=True
)
