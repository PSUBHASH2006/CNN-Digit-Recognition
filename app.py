from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="centered")
MODEL_PATH = Path(__file__).parent / "digit_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute an Otsu threshold without requiring OpenCV."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    best_variance = -1.0
    threshold = 127
    for value in range(256):
        weight_background += hist[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * hist[value]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value
    return threshold


def preprocess_digit(image: Image.Image):
    """Convert a photo/drawing into MNIST-like white-on-black 28x28 input."""
    gray_image = ImageOps.exif_transpose(image).convert("L")
    # Limit huge phone photos before processing.
    gray_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    gray = np.asarray(gray_image, dtype=np.uint8)

    # Determine ink polarity from the image border: paper is usually bright,
    # while an MNIST-style black background is dark.
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) > 127:
        gray = 255 - gray

    threshold = otsu_threshold(gray)
    # Keep only pixels clearly belonging to the foreground digit.
    mask = gray > max(20, min(threshold, 220))
    ys, xs = np.where(mask)
    if len(xs) < 8:
        raise ValueError("I couldn't detect a clear digit. Use a darker, larger digit with a plain background.")

    # Crop around the digit, preserving its aspect ratio.
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    cropped = gray[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    # Remove faint background/noise while retaining grayscale stroke edges.
    digit = np.where(cropped_mask, cropped, 0).astype(np.uint8)

    # Fit digit inside a 20x20 box (MNIST digits occupy roughly this area).
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit_img = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit_img, ((28 - new_w) // 2, (28 - new_h) // 2))
    processed = np.asarray(canvas, dtype=np.float32) / 255.0
    return processed.reshape(1, 28, 28, 1), canvas


st.title("✍️ Handwritten Digit Recognition")
st.write("Upload one handwritten digit (0–9). The app prepares it in an MNIST-like format before prediction.")

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded. Make sure digit_model.keras is beside app.py.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is not None:
    try:
        original = Image.open(uploaded)
        st.image(original, caption="Your uploaded image", width=260)
        if st.button("Predict digit", type="primary"):
            model_input, preview = preprocess_digit(original)
            probabilities = model.predict(model_input, verbose=0)[0]
            predicted = int(np.argmax(probabilities))
            order = np.argsort(probabilities)[::-1]

            st.image(preview.resize((280, 280)), caption="Image actually sent to the model (28 × 28, enlarged)", width=280)
            st.success(f"Predicted digit: {predicted}")
            st.metric("Model confidence", f"{float(probabilities[predicted]):.1%}")
            st.subheader("Top predictions")
            for idx in order[:3]:
                st.write(f"**{int(idx)}** — {float(probabilities[idx]):.1%}")
            st.bar_chart({"Probability": probabilities}, x_label="Digit", y_label="Probability")
            st.caption("If the processed preview does not look like a clear white digit on a black background, try a sharper image with one large digit and no surrounding text.")
    except Exception as exc:
        st.error(str(exc))
else:
    st.info("For best results, upload one large, clear digit on a plain background. Avoid shadows, grids, and multiple digits.")
