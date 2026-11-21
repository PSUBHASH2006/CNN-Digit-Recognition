import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="centered")
MODEL_PATH = Path(__file__).parent / "digit_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload an image containing one handwritten digit (0–9). The CNN model will predict the digit.")

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
except Exception as exc:
    st.error("The model could not be loaded. Check that digit_model.keras is in the same folder as app.py.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded:
    image = Image.open(uploaded).convert("L")
    st.image(image, caption="Uploaded image", width=220)
    if st.button("Predict digit", type="primary"):
        # Matches the notebook preprocessing: grayscale, resize to 28x28,
        # convert to float32 and scale pixel values to [0, 1].
        resized = image.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        probabilities = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])
        st.success(f"Predicted digit: **{digit}**")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.bar_chart({"Probability": probabilities}, x_label="Digit", y_label="Probability")
        st.caption("Tip: use a clear, centered image of a single dark handwritten digit on a light background, similar to the MNIST examples.")
else:
    st.info("Upload a clear image to get started.")
